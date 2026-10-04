"""Flashcards + spaced repetition + quiz state (CLAUDE.md §7 Phase 5b).

Plain SQLite (cards are study material, not PII — the PII gate runs before
any source text reaches the card generator). Everything the scheduler needs is
in here, nothing in memory: "what is due" is a query over `cards.due_at`,
"has today's quiz gone out" is `quiz_settings.last_daily_date`, and every
question already sent is a `quiz_items` row, so inline buttons keep working
across restarts.

Per-user isolation: every public method takes `telegram_id` and repeats it in
the SQL. A card/item id belonging to someone else behaves exactly like a
missing one — there is no method that reads, reviews or deletes across users.

Times are naive UTC, stored as "YYYY-MM-DD HH:MM:SS" text (sorts correctly).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiosqlite

from tools.quiz_mode import ANSWER_MAX_CHARS
from tools.sm2 import CardState

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id      INTEGER NOT NULL,
    front            TEXT NOT NULL,
    back             TEXT NOT NULL,
    front_key        TEXT NOT NULL,          -- normalized front, for de-duplication
    explanation      TEXT,                   -- optional, shown after the answer
    options_json     TEXT NOT NULL DEFAULT '[]',  -- generated wrong options (validated)
    source_key       TEXT NOT NULL,          -- 'kb:<source id>' | 'pdf:<sha256>'
    source_title     TEXT NOT NULL,
    ease             REAL NOT NULL DEFAULT 2.5,
    interval_days    INTEGER NOT NULL DEFAULT 0,
    repetitions      INTEGER NOT NULL DEFAULT 0,
    lapses           INTEGER NOT NULL DEFAULT 0,
    due_at           TEXT NOT NULL,
    last_reviewed_at TEXT,
    created_at       TEXT NOT NULL,
    UNIQUE (telegram_id, front_key)
);
CREATE INDEX IF NOT EXISTS idx_cards_due ON cards (telegram_id, due_at);

CREATE TABLE IF NOT EXISTS card_sources (
    telegram_id INTEGER NOT NULL,
    source_key  TEXT NOT NULL,
    title       TEXT NOT NULL,
    card_count  INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (telegram_id, source_key)
);

CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id     INTEGER NOT NULL,
    telegram_id INTEGER NOT NULL,
    grade       INTEGER NOT NULL,
    mode        TEXT NOT NULL,
    reviewed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_owner ON reviews (telegram_id, reviewed_at);

CREATE TABLE IF NOT EXISTS quiz_settings (
    telegram_id        INTEGER PRIMARY KEY,
    chat_id            INTEGER NOT NULL,
    quiz_time          TEXT NOT NULL,          -- local "HH:MM"
    utc_offset_minutes INTEGER NOT NULL,
    daily_cap          INTEGER NOT NULL,
    enabled            INTEGER NOT NULL DEFAULT 1,
    last_daily_date    TEXT                    -- local date the daily quiz last went out
);

CREATE TABLE IF NOT EXISTS quiz_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER NOT NULL,
    chat_id     INTEGER NOT NULL,
    kind        TEXT NOT NULL,                 -- 'daily' | 'manual'
    card_limit  INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quiz_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL,
    telegram_id   INTEGER NOT NULL,
    card_id       INTEGER NOT NULL,
    mode          TEXT NOT NULL,               -- 'choice' | 'reveal'
    choices_json  TEXT NOT NULL DEFAULT '[]',
    correct_index INTEGER,
    sent_at       TEXT NOT NULL,
    revealed_at   TEXT,
    answered_at   TEXT,
    grade         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_items_session ON quiz_items (session_id);
"""


def to_db(moment: datetime) -> str:
    return moment.replace(microsecond=0, tzinfo=None).isoformat(sep=" ")


def from_db(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(slots=True)
class Card:
    id: int
    telegram_id: int
    front: str
    back: str
    source_key: str
    source_title: str
    state: CardState
    due_at: datetime
    last_reviewed_at: datetime | None
    created_at: datetime
    explanation: str | None = None
    options: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class NewCard:
    front: str
    back: str
    front_key: str
    explanation: str = ""
    options: tuple[str, ...] = ()


@dataclass(slots=True)
class QuizSettings:
    telegram_id: int
    chat_id: int
    quiz_time: str
    utc_offset_minutes: int
    daily_cap: int
    enabled: bool
    last_daily_date: str | None


@dataclass(slots=True)
class QuizSession:
    id: int
    telegram_id: int
    chat_id: int
    kind: str
    card_limit: int


@dataclass(slots=True)
class QuizItem:
    id: int
    session_id: int
    telegram_id: int
    card_id: int
    mode: str
    choices: list[str]
    correct_index: int | None
    revealed: bool
    answered: bool


@dataclass(slots=True)
class CardStats:
    total: int
    due_now: int
    new: int
    learned: int  # reviewed successfully at least twice in a row
    reviewed_today: int
    next_due_at: datetime | None


def _card(row: aiosqlite.Row) -> Card:
    return Card(
        id=row["id"],
        telegram_id=row["telegram_id"],
        front=row["front"],
        back=row["back"],
        source_key=row["source_key"],
        source_title=row["source_title"],
        state=CardState(
            ease=row["ease"],
            interval_days=row["interval_days"],
            repetitions=row["repetitions"],
            lapses=row["lapses"],
        ),
        due_at=datetime.fromisoformat(row["due_at"]),
        last_reviewed_at=from_db(row["last_reviewed_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        explanation=row["explanation"] or None,
        options=json.loads(row["options_json"] or "[]"),
    )


def _settings(row: aiosqlite.Row) -> QuizSettings:
    return QuizSettings(
        telegram_id=row["telegram_id"],
        chat_id=row["chat_id"],
        quiz_time=row["quiz_time"],
        utc_offset_minutes=row["utc_offset_minutes"],
        daily_cap=row["daily_cap"],
        enabled=bool(row["enabled"]),
        last_daily_date=row["last_daily_date"],
    )


def _item(row: aiosqlite.Row) -> QuizItem:
    return QuizItem(
        id=row["id"],
        session_id=row["session_id"],
        telegram_id=row["telegram_id"],
        card_id=row["card_id"],
        mode=row["mode"],
        choices=json.loads(row["choices_json"]),
        correct_index=row["correct_index"],
        revealed=row["revealed_at"] is not None,
        answered=row["answered_at"] is not None,
    )


class FlashcardStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("FlashcardStore.connect() has not been called")
        return self._db

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(SCHEMA)
        await self._migrate()
        await self._db.commit()
        logger.info("Flashcard store ready at %s", self.db_path)

    async def _migrate(self) -> None:
        """Phase 5b.1: cards gained `explanation` and `options_json`.

        Existing cards are NOT rewritten: an old long answer (> 60 chars, the
        "answer + explanation in one" shape) is copied into `explanation`
        verbatim and its answer is left as it was — such a card stays in
        show-answer mode (logged as answer_too_long) until the user deletes it
        or its source is re-carded."""
        columns = {row["name"] for row in await self._all("PRAGMA table_info(cards)", ())}
        if "explanation" not in columns:
            await self.db.execute("ALTER TABLE cards ADD COLUMN explanation TEXT")
            cursor = await self.db.execute(
                "UPDATE cards SET explanation = back WHERE length(trim(back)) > ?",
                (ANSWER_MAX_CHARS,),
            )
            logger.info(
                "flashcards migration: added explanation; %d long answer(s) copied into it",
                cursor.rowcount,
            )
        if "options_json" not in columns:
            await self.db.execute(
                "ALTER TABLE cards ADD COLUMN options_json TEXT NOT NULL DEFAULT '[]'"
            )

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _one(self, sql: str, params: tuple) -> aiosqlite.Row | None:
        async with self.db.execute(sql, params) as cursor:
            return await cursor.fetchone()

    async def _all(self, sql: str, params: tuple) -> list[aiosqlite.Row]:
        async with self.db.execute(sql, params) as cursor:
            return list(await cursor.fetchall())

    # -- cards --------------------------------------------------------------------

    async def has_source(self, *, telegram_id: int, source_key: str) -> bool:
        row = await self._one(
            "SELECT 1 FROM card_sources WHERE telegram_id = ? AND source_key = ?",
            (telegram_id, source_key),
        )
        return row is not None

    async def add_cards(
        self,
        *,
        telegram_id: int,
        source_key: str,
        source_title: str,
        cards: list[NewCard],
        now: datetime,
    ) -> list[Card]:
        """Insert new cards (due immediately); duplicates of an existing front
        for this user are skipped. Records the source as done either way."""
        created: list[Card] = []
        stamp = to_db(now)
        for new in cards:
            cursor = await self.db.execute(
                "INSERT OR IGNORE INTO cards (telegram_id, front, back, front_key, explanation, "
                "options_json, source_key, source_title, due_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    telegram_id,
                    new.front,
                    new.back,
                    new.front_key,
                    new.explanation or None,
                    json.dumps(list(new.options), ensure_ascii=False),
                    source_key,
                    source_title,
                    stamp,
                    stamp,
                ),
            )
            if cursor.rowcount == 1 and cursor.lastrowid is not None:
                card = await self.get_card(telegram_id=telegram_id, card_id=cursor.lastrowid)
                if card is not None:
                    created.append(card)
        await self.db.execute(
            "INSERT OR REPLACE INTO card_sources (telegram_id, source_key, title, card_count, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (telegram_id, source_key, source_title, len(created), stamp),
        )
        await self.db.commit()
        return created

    async def get_card(self, *, telegram_id: int, card_id: int) -> Card | None:
        row = await self._one(
            "SELECT * FROM cards WHERE id = ? AND telegram_id = ?", (card_id, telegram_id)
        )
        return _card(row) if row else None

    async def delete_card(self, *, telegram_id: int, card_id: int) -> Card | None:
        card = await self.get_card(telegram_id=telegram_id, card_id=card_id)
        if card is None:
            return None
        await self.db.execute(
            "DELETE FROM quiz_items WHERE card_id = ? AND telegram_id = ?", (card_id, telegram_id)
        )
        await self.db.execute(
            "DELETE FROM reviews WHERE card_id = ? AND telegram_id = ?", (card_id, telegram_id)
        )
        await self.db.execute(
            "DELETE FROM cards WHERE id = ? AND telegram_id = ?", (card_id, telegram_id)
        )
        await self.db.commit()
        return card

    async def save_review(
        self,
        *,
        telegram_id: int,
        card_id: int,
        state: CardState,
        due_at: datetime,
        grade: int,
        mode: str,
        now: datetime,
    ) -> bool:
        cursor = await self.db.execute(
            "UPDATE cards SET ease = ?, interval_days = ?, repetitions = ?, lapses = ?, "
            "due_at = ?, last_reviewed_at = ? WHERE id = ? AND telegram_id = ?",
            (
                state.ease,
                state.interval_days,
                state.repetitions,
                state.lapses,
                to_db(due_at),
                to_db(now),
                card_id,
                telegram_id,
            ),
        )
        if cursor.rowcount != 1:
            await self.db.rollback()
            return False
        await self.db.execute(
            "INSERT INTO reviews (card_id, telegram_id, grade, mode, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (card_id, telegram_id, grade, mode, to_db(now)),
        )
        await self.db.commit()
        return True

    async def next_due_card(
        self, *, telegram_id: int, now: datetime, session_id: int | None = None
    ) -> Card | None:
        """Most overdue card, skipping cards already asked in this session."""
        row = await self._one(
            "SELECT * FROM cards WHERE telegram_id = ? AND due_at <= ? AND id NOT IN "
            "(SELECT card_id FROM quiz_items WHERE session_id = ? AND telegram_id = ?) "
            "ORDER BY due_at, id LIMIT 1",
            (telegram_id, to_db(now), session_id or 0, telegram_id),
        )
        return _card(row) if row else None

    async def answer_pool(self, *, telegram_id: int) -> list[tuple[int, str, str]]:
        """(card id, answer, source key) of this user's cards — the fallback
        distractor pool, after each card's own stored options."""
        rows = await self._all(
            "SELECT id, back, source_key FROM cards WHERE telegram_id = ?", (telegram_id,)
        )
        return [(row["id"], row["back"], row["source_key"]) for row in rows]

    async def all_cards(self, *, telegram_id: int) -> list[Card]:
        return [
            _card(row)
            for row in await self._all(
                "SELECT * FROM cards WHERE telegram_id = ? ORDER BY id", (telegram_id,)
            )
        ]

    async def stats(self, *, telegram_id: int, now: datetime, day_start: datetime) -> CardStats:
        row = await self._one(
            "SELECT COUNT(*) AS total, "
            "SUM(due_at <= ?) AS due_now, "
            "SUM(last_reviewed_at IS NULL) AS new, "
            "SUM(repetitions >= 2) AS learned, "
            "MIN(CASE WHEN due_at > ? THEN due_at END) AS next_due "
            "FROM cards WHERE telegram_id = ?",
            (to_db(now), to_db(now), telegram_id),
        )
        reviewed = await self._one(
            "SELECT COUNT(*) AS n FROM reviews WHERE telegram_id = ? AND reviewed_at >= ?",
            (telegram_id, to_db(day_start)),
        )
        assert row is not None and reviewed is not None
        return CardStats(
            total=row["total"] or 0,
            due_now=row["due_now"] or 0,
            new=row["new"] or 0,
            learned=row["learned"] or 0,
            reviewed_today=reviewed["n"] or 0,
            next_due_at=from_db(row["next_due"]),
        )

    # -- quiz settings -----------------------------------------------------------------

    async def get_settings(self, *, telegram_id: int) -> QuizSettings | None:
        row = await self._one("SELECT * FROM quiz_settings WHERE telegram_id = ?", (telegram_id,))
        return _settings(row) if row else None

    async def ensure_settings(
        self, *, telegram_id: int, chat_id: int, quiz_time: str, utc_offset_minutes: int,
        daily_cap: int,
    ) -> QuizSettings:  # fmt: skip
        """Create default settings on first use; always refresh chat_id."""
        await self.db.execute(
            "INSERT INTO quiz_settings (telegram_id, chat_id, quiz_time, utc_offset_minutes, "
            "daily_cap) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (telegram_id) DO UPDATE SET chat_id = excluded.chat_id",
            (telegram_id, chat_id, quiz_time, utc_offset_minutes, daily_cap),
        )
        await self.db.commit()
        settings = await self.get_settings(telegram_id=telegram_id)
        assert settings is not None
        return settings

    async def update_settings(
        self,
        *,
        telegram_id: int,
        quiz_time: str | None = None,
        utc_offset_minutes: int | None = None,
        daily_cap: int | None = None,
        enabled: bool | None = None,
        last_daily_date: str | None = None,
    ) -> None:
        updates = {
            "quiz_time": quiz_time,
            "utc_offset_minutes": utc_offset_minutes,
            "daily_cap": daily_cap,
            "enabled": None if enabled is None else int(enabled),
            "last_daily_date": last_daily_date,
        }
        updates = {key: value for key, value in updates.items() if value is not None}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        await self.db.execute(
            f"UPDATE quiz_settings SET {assignments} WHERE telegram_id = ?",
            (*updates.values(), telegram_id),
        )
        await self.db.commit()

    async def enabled_settings(self) -> list[QuizSettings]:
        """Scheduler only: every user with the daily quiz on."""
        return [
            _settings(r)
            for r in await self._all("SELECT * FROM quiz_settings WHERE enabled = 1", ())
        ]

    async def claim_daily(self, *, telegram_id: int, local_date: str) -> bool:
        """Atomically mark today's daily quiz as sent; False if it already was.
        Claimed *before* sending, so a failing send can't repeat every tick."""
        cursor = await self.db.execute(
            "UPDATE quiz_settings SET last_daily_date = ? WHERE telegram_id = ? "
            "AND (last_daily_date IS NULL OR last_daily_date != ?)",
            (local_date, telegram_id, local_date),
        )
        await self.db.commit()
        return cursor.rowcount == 1

    # -- quiz sessions and items ----------------------------------------------------

    async def create_session(
        self, *, telegram_id: int, chat_id: int, kind: str, card_limit: int, now: datetime
    ) -> QuizSession:
        cursor = await self.db.execute(
            "INSERT INTO quiz_sessions (telegram_id, chat_id, kind, card_limit, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (telegram_id, chat_id, kind, card_limit, to_db(now)),
        )
        await self.db.commit()
        assert cursor.lastrowid is not None
        return QuizSession(cursor.lastrowid, telegram_id, chat_id, kind, card_limit)

    async def get_session(self, *, telegram_id: int, session_id: int) -> QuizSession | None:
        row = await self._one(
            "SELECT * FROM quiz_sessions WHERE id = ? AND telegram_id = ?",
            (session_id, telegram_id),
        )
        if row is None:
            return None
        return QuizSession(
            row["id"], row["telegram_id"], row["chat_id"], row["kind"], row["card_limit"]
        )

    async def session_progress(self, *, telegram_id: int, session_id: int) -> tuple[int, int]:
        """(items asked, items answered correctly — grade >= 3) in a session."""
        row = await self._one(
            "SELECT COUNT(*) AS asked, SUM(grade >= 3) AS good FROM quiz_items "
            "WHERE session_id = ? AND telegram_id = ?",
            (session_id, telegram_id),
        )
        assert row is not None
        return row["asked"] or 0, row["good"] or 0

    async def create_item(
        self,
        *,
        session_id: int,
        telegram_id: int,
        card_id: int,
        mode: str,
        choices: list[str],
        correct_index: int | None,
        now: datetime,
    ) -> QuizItem:
        cursor = await self.db.execute(
            "INSERT INTO quiz_items (session_id, telegram_id, card_id, mode, choices_json, "
            "correct_index, sent_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                telegram_id,
                card_id,
                mode,
                json.dumps(choices, ensure_ascii=False),
                correct_index,
                to_db(now),
            ),
        )
        await self.db.commit()
        assert cursor.lastrowid is not None
        item = await self.get_item(telegram_id=telegram_id, item_id=cursor.lastrowid)
        assert item is not None
        return item

    async def get_item(self, *, telegram_id: int, item_id: int) -> QuizItem | None:
        row = await self._one(
            "SELECT * FROM quiz_items WHERE id = ? AND telegram_id = ?", (item_id, telegram_id)
        )
        return _item(row) if row else None

    async def mark_revealed(self, *, telegram_id: int, item_id: int, now: datetime) -> None:
        await self.db.execute(
            "UPDATE quiz_items SET revealed_at = COALESCE(revealed_at, ?) "
            "WHERE id = ? AND telegram_id = ?",
            (to_db(now), item_id, telegram_id),
        )
        await self.db.commit()

    async def claim_answer(
        self, *, telegram_id: int, item_id: int, grade: int, now: datetime
    ) -> bool:
        """Atomically mark an item answered; False if it already was (double tap,
        or a second device) — so one question can never be graded twice."""
        cursor = await self.db.execute(
            "UPDATE quiz_items SET answered_at = ?, grade = ? "
            "WHERE id = ? AND telegram_id = ? AND answered_at IS NULL",
            (to_db(now), grade, item_id, telegram_id),
        )
        await self.db.commit()
        return cursor.rowcount == 1
