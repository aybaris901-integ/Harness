"""Encrypted document archive storage (CLAUDE.md §5, §7 Phase 3).

A separate SQLite database from `storage/db.py` — the CLAUDE.md architecture
draws a hard line between the plain relational DB (users, chat history) and
the "encrypted local DB (documents, PII)". Two things are encrypted at rest
with Fernet (AES-128-CBC + HMAC-SHA256, authenticated) using a key that comes
from `Settings` / `.env` and is never logged:

- the scanned image, as a standalone `<uuid>.enc` file under `scan_dir`
- the extracted field values and the raw OCR text, as BLOB columns

`document_type` and the set of field *names* (not values) are stored in the
clear — they are not PII themselves and are what natural-language search
matches against (`tools.document_fields.score_query`) without ever decrypting
another user's — or an unmatched document's — PII to do it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from cryptography.fernet import Fernet, InvalidToken

from tools.document_fields import score_query

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id   INTEGER NOT NULL,
    document_type TEXT NOT NULL,
    field_names   TEXT NOT NULL DEFAULT '',
    fields_enc    BLOB NOT NULL,
    raw_text_enc  BLOB NOT NULL,
    scan_filename TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents (telegram_id, id);
"""


class DocumentStorageError(RuntimeError):
    """A stored document could not be decrypted (wrong/rotated key, corruption)."""


@dataclass(slots=True)
class DocumentRecord:
    id: int
    telegram_id: int
    document_type: str
    fields: dict[str, str]
    raw_text: str
    scan_path: Path


@dataclass(slots=True)
class DocumentMatch:
    record: DocumentRecord
    matched_fields: list[str]
    score: int


class DocumentStore:
    """Thin async wrapper over one SQLite connection + one directory of encrypted scans."""

    def __init__(self, *, db_path: Path, scan_dir: Path, encryption_key: bytes) -> None:
        self.db_path = db_path
        self.scan_dir = scan_dir
        self._fernet = Fernet(encryption_key)
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("DocumentStore.connect() has not been called")
        return self._db

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info("Document store ready at %s (scans in %s)", self.db_path, self.scan_dir)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # -- writes ---------------------------------------------------------------

    async def save(
        self,
        *,
        telegram_id: int,
        document_type: str,
        fields: dict[str, str],
        raw_text: str,
        image_bytes: bytes,
    ) -> DocumentRecord:
        scan_filename = f"{uuid.uuid4().hex}.enc"
        scan_path = self.scan_dir / scan_filename
        encrypted_image = self._fernet.encrypt(image_bytes)
        await asyncio.to_thread(scan_path.write_bytes, encrypted_image)

        fields_enc = self._fernet.encrypt(json.dumps(fields, ensure_ascii=False).encode("utf-8"))
        raw_text_enc = self._fernet.encrypt(raw_text.encode("utf-8"))
        field_names = ",".join(sorted(fields.keys()))

        cursor = await self.db.execute(
            """
            INSERT INTO documents
                (telegram_id, document_type, field_names, fields_enc, raw_text_enc, scan_filename)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (telegram_id, document_type, field_names, fields_enc, raw_text_enc, scan_filename),
        )
        await self.db.commit()
        assert cursor.lastrowid is not None
        return DocumentRecord(
            id=cursor.lastrowid,
            telegram_id=telegram_id,
            document_type=document_type,
            fields=fields,
            raw_text=raw_text,
            scan_path=scan_path,
        )

    # -- reads ------------------------------------------------------------------

    async def search(self, *, telegram_id: int, query: str) -> DocumentMatch | None:
        """Best-matching document for a natural-language query, or None.

        Scoring/matching is pure keyword heuristics (`score_query`) over the
        plaintext `document_type` / `field_names` columns — no PII value is
        decrypted until after a document has already been picked as the match.
        """
        async with self.db.execute(
            "SELECT id, document_type, field_names FROM documents "
            "WHERE telegram_id = ? ORDER BY id DESC",
            (telegram_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        best: tuple[int, int, list[str]] | None = None  # (score, doc_id, matched_fields)
        for row in rows:
            field_names = [name for name in row["field_names"].split(",") if name]
            score, matched_fields = score_query(
                query, document_type=row["document_type"], field_names=field_names
            )
            if score > 0 and (best is None or score > best[0]):
                best = (score, row["id"], matched_fields)

        if best is None:
            return None
        score, doc_id, matched_fields = best
        record = await self._load(doc_id)
        return DocumentMatch(record=record, matched_fields=matched_fields, score=score)

    async def _load(self, document_id: int) -> DocumentRecord:
        async with self.db.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise DocumentStorageError(f"document {document_id} vanished mid-query")
        return self._decrypt_row(row)

    def _decrypt_row(self, row: aiosqlite.Row) -> DocumentRecord:
        try:
            fields = json.loads(self._fernet.decrypt(row["fields_enc"]).decode("utf-8"))
            raw_text = self._fernet.decrypt(row["raw_text_enc"]).decode("utf-8")
        except InvalidToken as exc:
            raise DocumentStorageError(
                f"could not decrypt document {row['id']} — wrong DOCUMENTS_ENCRYPTION_KEY?"
            ) from exc
        return DocumentRecord(
            id=row["id"],
            telegram_id=row["telegram_id"],
            document_type=row["document_type"],
            fields=fields,
            raw_text=raw_text,
            scan_path=self.scan_dir / row["scan_filename"],
        )

    async def load_scan(self, record: DocumentRecord) -> bytes:
        encrypted = await asyncio.to_thread(record.scan_path.read_bytes)
        try:
            return self._fernet.decrypt(encrypted)
        except InvalidToken as exc:
            raise DocumentStorageError(
                f"could not decrypt scan for document {record.id} — wrong DOCUMENTS_ENCRYPTION_KEY?"
            ) from exc
