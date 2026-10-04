"""RAG knowledge base storage (CLAUDE.md §7 Phase 5a).

One SQLite file with the `sqlite-vec` extension:

    sources     one row per indexed item — a migrated /note screenshot, or a
                notebook page (whose photo is a Fernet-encrypted file, via
                storage.encrypted_files, the same primitive as document scans)
    chunks      the text pieces of each source
    vec_chunks  vec0 virtual table: one embedding per chunk, cosine distance

Per-user isolation is structural, not a post-filter: `telegram_id` is a vec0
PARTITION KEY, so a KNN query is answered from that user's partition only, and
every public read method takes `telegram_id` as a required argument and repeats
it in the SQL join. There is no method that searches across users.

The index is pinned to one embedding model (`meta.embedding_model`): vectors
from different models are not comparable, so opening the store with another
model id fails loudly instead of returning garbage similarities.

Why sqlite-vec over Chroma: it is a single prebuilt loadable extension per
platform (pip wheel, no compiler, no admin, no server) on top of the stdlib
sqlite3 we already use; Chroma pulls onnxruntime, hnswlib and a server-ish
stack — the native-build pain we hit with SQLCipher/tessdata.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
import sqlite_vec

from storage.encrypted_files import EncryptedFileStore

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id    INTEGER NOT NULL,
    kind           TEXT NOT NULL,          -- 'note' (migrated /note) | 'page' (notebook photo)
    title          TEXT NOT NULL,
    origin         TEXT,                   -- e.g. 'note:42'; makes migration idempotent
    photo_filename TEXT,                   -- encrypted page photo, NULL for notes
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (origin)
);

CREATE INDEX IF NOT EXISTS idx_sources_owner ON sources (telegram_id, id);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    telegram_id INTEGER NOT NULL,
    seq         INTEGER NOT NULL,
    text        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_owner ON chunks (telegram_id, source_id);
"""


class KnowledgeStoreError(RuntimeError):
    """The store cannot be opened as configured (extension, model mismatch)."""


@dataclass(slots=True)
class KnowledgeSource:
    id: int
    telegram_id: int
    kind: str
    title: str
    origin: str | None
    photo_filename: str | None
    created_at: str


@dataclass(slots=True)
class ChunkHit:
    chunk_id: int
    text: str
    score: float  # cosine similarity, 1.0 = identical direction
    source: KnowledgeSource


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class KnowledgeStore:
    def __init__(
        self,
        *,
        db_path: Path,
        page_dir: Path,
        encryption_key: bytes,
        embedding_model: str,
        dimensions: int,
    ) -> None:
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.pages = EncryptedFileStore(page_dir, encryption_key)
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("KnowledgeStore.connect() has not been called")
        return self._db

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pages.ensure_dir()
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        try:
            await self._db.enable_load_extension(True)
            await self._db.load_extension(sqlite_vec.loadable_path())
            await self._db.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError) as exc:
            await self.close()
            raise KnowledgeStoreError(
                "could not load the sqlite-vec extension. This Python's sqlite3 was "
                "built without extension loading (common with pyenv builds) — use "
                f"python.org/Debian/uv Python, or see README. ({exc})"
            ) from exc

        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(SCHEMA)
        await self._db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id    INTEGER PRIMARY KEY,
                telegram_id INTEGER PARTITION KEY,
                embedding   FLOAT[{self.dimensions}] DISTANCE_METRIC=cosine
            )
            """
        )
        await self._check_model()
        await self._db.commit()
        version = await self._scalar("SELECT vec_version()")
        logger.info(
            "Knowledge store ready at %s (sqlite-vec %s, %s)",
            self.db_path,
            version,
            self.embedding_model,
        )

    async def _check_model(self) -> None:
        stored = await self._scalar("SELECT value FROM meta WHERE key = 'embedding_model'")
        if stored is None:
            await self.db.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_model', ?)",
                (self.embedding_model,),
            )
        elif stored != self.embedding_model:
            await self.close()
            raise KnowledgeStoreError(
                f"{self.db_path} was indexed with {stored}, but the configured embedding "
                f"model is {self.embedding_model}. Vectors from different models are not "
                "comparable: restore EMBEDDING_MODEL/EMBEDDING_DIM, or move the file aside "
                "to rebuild the index (staging notes re-migrate automatically)."
            )

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _scalar(self, sql: str, params: tuple = ()) -> object | None:
        async with self.db.execute(sql, params) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None

    # -- writes ---------------------------------------------------------------

    async def add_source(
        self,
        *,
        telegram_id: int,
        kind: str,
        title: str,
        chunks: list[str],
        embeddings: list[list[float]],
        origin: str | None = None,
        photo: bytes | None = None,
    ) -> KnowledgeSource:
        if len(chunks) != len(embeddings) or not chunks:
            raise ValueError("need one embedding per chunk, and at least one chunk")
        if any(len(vector) != self.dimensions for vector in embeddings):
            raise ValueError(f"embeddings must have {self.dimensions} dimensions")

        photo_filename = await self.pages.write(photo) if photo is not None else None
        try:
            cursor = await self.db.execute(
                "INSERT INTO sources (telegram_id, kind, title, origin, photo_filename) "
                "VALUES (?, ?, ?, ?, ?)",
                (telegram_id, kind, title, origin, photo_filename),
            )
            source_id = cursor.lastrowid
            assert source_id is not None
            for seq, (text, vector) in enumerate(zip(chunks, embeddings, strict=True)):
                cursor = await self.db.execute(
                    "INSERT INTO chunks (source_id, telegram_id, seq, text) VALUES (?, ?, ?, ?)",
                    (source_id, telegram_id, seq, text),
                )
                await self.db.execute(
                    "INSERT INTO vec_chunks (chunk_id, telegram_id, embedding) VALUES (?, ?, ?)",
                    (cursor.lastrowid, telegram_id, _serialize(vector)),
                )
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            if photo_filename:
                await self.pages.delete(photo_filename)
            raise
        source = await self.get_source(telegram_id=telegram_id, source_id=source_id)
        assert source is not None
        return source

    # -- reads ------------------------------------------------------------------

    async def has_origin(self, origin: str) -> bool:
        return await self._scalar("SELECT 1 FROM sources WHERE origin = ?", (origin,)) is not None

    async def search(self, *, telegram_id: int, embedding: list[float], k: int) -> list[ChunkHit]:
        """Top-`k` chunks for this user only, most similar first."""
        async with self.db.execute(
            """
            SELECT v.chunk_id, v.distance, c.text,
                   s.id AS source_id, s.telegram_id, s.kind, s.title, s.origin,
                   s.photo_filename, s.created_at
            FROM vec_chunks AS v
            JOIN chunks  AS c ON c.id = v.chunk_id AND c.telegram_id = ?
            JOIN sources AS s ON s.id = c.source_id AND s.telegram_id = ?
            WHERE v.embedding MATCH ? AND v.k = ? AND v.telegram_id = ?
            ORDER BY v.distance
            """,
            (telegram_id, telegram_id, _serialize(embedding), k, telegram_id),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            ChunkHit(
                chunk_id=row["chunk_id"],
                text=row["text"],
                score=1.0 - float(row["distance"]),
                source=KnowledgeSource(
                    id=row["source_id"],
                    telegram_id=row["telegram_id"],
                    kind=row["kind"],
                    title=row["title"],
                    origin=row["origin"],
                    photo_filename=row["photo_filename"],
                    created_at=row["created_at"],
                ),
            )
            for row in rows
        ]

    async def get_source(self, *, telegram_id: int, source_id: int) -> KnowledgeSource | None:
        async with self.db.execute(
            "SELECT * FROM sources WHERE id = ? AND telegram_id = ?", (source_id, telegram_id)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return KnowledgeSource(
            id=row["id"],
            telegram_id=row["telegram_id"],
            kind=row["kind"],
            title=row["title"],
            origin=row["origin"],
            photo_filename=row["photo_filename"],
            created_at=row["created_at"],
        )

    async def load_photo(self, *, telegram_id: int, source_id: int) -> bytes | None:
        """Decrypted page photo — only if `source_id` belongs to `telegram_id`."""
        source = await self.get_source(telegram_id=telegram_id, source_id=source_id)
        if source is None or source.photo_filename is None:
            return None
        return await self.pages.read(source.photo_filename)

    async def count_sources(self, *, telegram_id: int) -> int:
        value = await self._scalar(
            "SELECT COUNT(*) FROM sources WHERE telegram_id = ?", (telegram_id,)
        )
        return int(value or 0)
