"""Staging table for Phase 4 notes (CLAUDE.md §7 Phase 4, "ties into Phase 5's
vector store").

Screenshots are the explicit non-sensitive case (CLAUDE.md §5) — unlike
`storage/documents.py`, nothing here is encrypted. This table is deliberately
NOT a search index: Phase 5's vector store is what will make these notes
actually queryable. Until Phase 5 exists, this only exists so a formatted note
isn't lost between now and then — no search/query method is provided here on
purpose (CLAUDE.md §7 Phase 4: "Don't build ad-hoc search for it now").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id  INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    title        TEXT NOT NULL,
    content_md   TEXT NOT NULL,
    content_json TEXT NOT NULL,
    tags         TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_notes_owner ON notes (telegram_id, id);
"""


@dataclass(slots=True)
class NoteRecord:
    id: int
    telegram_id: int
    kind: str
    title: str
    content_md: str
    content_json: str
    tags: list[str] = field(default_factory=list)


class NoteStore:
    """Thin async wrapper over one plain SQLite connection. No search — see
    the module docstring for why."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("NoteStore.connect() has not been called")
        return self._db

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info("Note staging store ready at %s", self.db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def save(
        self,
        *,
        telegram_id: int,
        kind: str,
        title: str,
        content_md: str,
        content_json: str,
        tags: list[str],
    ) -> NoteRecord:
        cursor = await self.db.execute(
            """
            INSERT INTO notes (telegram_id, kind, title, content_md, content_json, tags)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (telegram_id, kind, title, content_md, content_json, ",".join(tags)),
        )
        await self.db.commit()
        assert cursor.lastrowid is not None
        return NoteRecord(
            id=cursor.lastrowid,
            telegram_id=telegram_id,
            kind=kind,
            title=title,
            content_md=content_md,
            content_json=content_json,
            tags=tags,
        )

    async def count(self, *, telegram_id: int) -> int:
        """Only used for the self-check roundtrip and startup logging — not a
        query API; see the module docstring."""
        async with self.db.execute(
            "SELECT COUNT(*) AS n FROM notes WHERE telegram_id = ?", (telegram_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["n"]) if row else 0
