"""SQLite storage: users and per-user chat history.

Phase 1 only needs these two tables. Encrypted document storage (Phase 3) and the
vector store (Phase 5) get their own modules later — this one stays plain SQLite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from providers.base import ChatMessage

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    telegram_id   INTEGER PRIMARY KEY,
    username      TEXT,
    first_name    TEXT,
    language_code TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    chat_id     INTEGER NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (telegram_id, chat_id, id);
"""


@dataclass(slots=True)
class UserProfile:
    telegram_id: int
    username: str | None
    first_name: str | None
    language_code: str | None


class Storage:
    """Thin async wrapper over one SQLite connection."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Storage.connect() has not been called")
        return self._db

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info("SQLite ready at %s", self.db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def upsert_user(self, profile: UserProfile) -> None:
        await self.db.execute(
            """
            INSERT INTO users (telegram_id, username, first_name, language_code)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(telegram_id) DO UPDATE SET
                username      = excluded.username,
                first_name    = excluded.first_name,
                language_code = excluded.language_code,
                last_seen_at  = datetime('now')
            """,
            (profile.telegram_id, profile.username, profile.first_name, profile.language_code),
        )
        await self.db.commit()

    async def add_message(self, *, telegram_id: int, chat_id: int, role: str, content: str) -> None:
        """Append one turn. The user row must exist (see `UserTrackingMiddleware`)."""
        await self.db.execute(
            "INSERT INTO messages (telegram_id, chat_id, role, content) VALUES (?, ?, ?, ?)",
            (telegram_id, chat_id, role, content),
        )
        await self.db.commit()

    async def recent_messages(
        self, *, telegram_id: int, chat_id: int, limit: int
    ) -> list[ChatMessage]:
        """The last `limit` turns, oldest first (the order an LLM expects)."""
        async with self.db.execute(
            """
            SELECT role, content FROM messages
            WHERE telegram_id = ? AND chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (telegram_id, chat_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [ChatMessage(role=row["role"], content=row["content"]) for row in reversed(rows)]

    async def clear_history(self, *, telegram_id: int, chat_id: int) -> int:
        cursor = await self.db.execute(
            "DELETE FROM messages WHERE telegram_id = ? AND chat_id = ?",
            (telegram_id, chat_id),
        )
        await self.db.commit()
        return cursor.rowcount

    async def message_count(self, *, telegram_id: int, chat_id: int) -> int:
        async with self.db.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE telegram_id = ? AND chat_id = ?",
            (telegram_id, chat_id),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["n"]) if row else 0
