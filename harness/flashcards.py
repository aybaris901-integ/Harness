"""Flashcards + spaced repetition + daily quiz (CLAUDE.md §7 Phase 5b).

    knowledge-base source / text PDF
        -> text (storage.knowledge.source_text | tools.pdf_text; scanned PDF -> /page hint)
        -> tools.flashcards.segments
        -> PII gate per segment: document_fields.has_pii_signals fires -> the
           segment is SKIPPED, never sent to a cloud LLM (CLAUDE.md §5)
        -> tools.flashcards.generate_cards (llm_router, §9 prompt + JSON schema)
        -> storage.flashcards (per user, due immediately)

    quiz session: one card at a time, up to `card_limit`; the next card is sent
    when the previous one is answered. Multiple choice when the user has enough
    other cards to draw distractors from, else "show answer -> rate recall".
    Every answer is an SM-2 review (tools.sm2).

    daily quiz: `daily_due` is a pure function of the DB and the clock — one
    session per user per LOCAL day, at or after their quiz time. A missed day
    (bot down) is not replayed: the next start sends at most today's single
    capped session, so downtime never turns into a backlog flood.

No Telegram here — the bot layer renders `QuizPrompt`s and calls back in.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import strings
from llm_router import AllProvidersFailedError, LLMRouter
from storage.flashcards import Card, CardStats, FlashcardStore, NewCard, QuizItem, QuizSettings
from storage.knowledge import KnowledgeStore
from tools import document_fields, flashcards, pdf_text, quiz_clock, quiz_mode, sm2

logger = logging.getLogger(__name__)

# Inline-button grades for "show answer -> rate": Again / Hard / Good / Easy.
RATING_GRADES = (1, 3, 4, 5)
CHOICE_CORRECT_GRADE = 4
CHOICE_WRONG_GRADE = 1
# "/cards make": subcommand words, not part of the user's language.
_CARDS_OPTIONS = frozenset({"make"})


class FlashcardError(RuntimeError):
    """A user-presentable failure (message is a strings.py text)."""


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def local_now(settings: QuizSettings, now: datetime) -> datetime:
    return quiz_clock.to_local(now, settings.utc_offset_minutes)


def daily_due(settings: QuizSettings, now: datetime) -> bool:
    return quiz_clock.daily_due(
        now_utc=now,
        quiz_time=settings.quiz_time,
        utc_offset_minutes=settings.utc_offset_minutes,
        enabled=settings.enabled,
        last_daily_date=settings.last_daily_date,
    )


@dataclass(slots=True)
class GenerationReport:
    cards: list[Card] = field(default_factory=list)
    sources_done: int = 0
    sources_remaining: int = 0
    segments_skipped_pii: int = 0
    cards_dropped_pii: int = 0  # generated cards whose own text tripped the gate
    segments_failed: int = 0
    truncated_pdf: bool = False


@dataclass(slots=True)
class QuizPrompt:
    item: QuizItem
    card: Card
    position: int  # 1-based within the session
    card_limit: int


@dataclass(slots=True)
class AnswerResult:
    card: Card
    grade: int
    correct: bool | None  # None for self-rated (reveal) answers
    chosen: str | None
    next_due_at: datetime
    next_due_local: str  # "dd.mm.yyyy" in the user's own UTC offset
    next_prompt: QuizPrompt | None
    session_done: bool
    session_asked: int
    session_good: int


class FlashcardService:
    def __init__(
        self,
        *,
        store: FlashcardStore,
        router: LLMRouter,
        knowledge_store: KnowledgeStore | None = None,
        default_quiz_time: str = "20:00",
        default_utc_offset_minutes: int = 300,
        default_daily_cap: int = 10,
        max_sources_per_run: int = 5,
        max_segments_per_source: int = 4,
        pdf_max_mb: float = 20.0,
    ) -> None:
        self.store = store
        self.router = router
        self.knowledge_store = knowledge_store
        self.default_quiz_time = default_quiz_time
        self.default_utc_offset_minutes = default_utc_offset_minutes
        self.default_daily_cap = default_daily_cap
        self.max_sources_per_run = max_sources_per_run
        self.max_segments_per_source = max_segments_per_source
        self.pdf_max_mb = pdf_max_mb

    def describe(self) -> str:
        return (
            f"Flashcards: daily quiz default {self.default_quiz_time} "
            f"{quiz_clock.format_utc_offset(self.default_utc_offset_minutes)}, "
            f"cap {self.default_daily_cap}/day"
        )

    async def settings(self, *, telegram_id: int, chat_id: int) -> QuizSettings:
        return await self.store.ensure_settings(
            telegram_id=telegram_id,
            chat_id=chat_id,
            quiz_time=self.default_quiz_time,
            utc_offset_minutes=self.default_utc_offset_minutes,
            daily_cap=self.default_daily_cap,
        )

    # -- generation -------------------------------------------------------------------

    async def make_from_knowledge(
        self, *, telegram_id: int, chat_id: int, user_message: str
    ) -> GenerationReport:
        """Cards from this user's knowledge-base sources that have none yet,
        at most `max_sources_per_run` per call (bounded LLM spend per command).

        `user_message` is the raw "/cards make ..." text. "make" is command
        syntax, not the user's language, so it is dropped here before the
        reply language is picked (tools.language strips "/cards" itself)."""
        user_message = " ".join(
            word for word in user_message.split() if word.lower() not in _CARDS_OPTIONS
        )
        if self.knowledge_store is None:
            raise FlashcardError(strings.KB_NOT_CONFIGURED)
        sources = await self.knowledge_store.list_sources(telegram_id=telegram_id)
        pending = [
            source
            for source in sources
            if not await self.store.has_source(
                telegram_id=telegram_id, source_key=f"kb:{source.id}"
            )
        ]
        if not sources:
            raise FlashcardError(strings.CARDS_NO_SOURCES)
        if not pending:
            raise FlashcardError(strings.CARDS_ALL_SOURCES_DONE)

        report = GenerationReport()
        for source in pending[: self.max_sources_per_run]:
            text = await self.knowledge_store.source_text(
                telegram_id=telegram_id, source_id=source.id
            )
            await self._generate(
                report,
                telegram_id=telegram_id,
                text=text,
                source_key=f"kb:{source.id}",
                source_title=source.title,
                user_message=user_message,
            )
            report.sources_done += 1
        report.sources_remaining = max(0, len(pending) - self.max_sources_per_run)
        await self.settings(telegram_id=telegram_id, chat_id=chat_id)
        return report

    async def make_from_pdf(
        self, *, telegram_id: int, chat_id: int, data: bytes, filename: str, user_message: str
    ) -> GenerationReport:
        source_key = "pdf:" + hashlib.sha256(data).hexdigest()
        if await self.store.has_source(telegram_id=telegram_id, source_key=source_key):
            raise FlashcardError(strings.CARDS_PDF_ALREADY_DONE)
        try:
            extracted = await pdf_text.extract_text(data)
        except pdf_text.PdfError as exc:
            raise FlashcardError(strings.CARDS_PDF_UNREADABLE.format(error=exc)) from exc
        if extracted.scanned:
            raise FlashcardError(strings.CARDS_PDF_SCANNED)

        report = GenerationReport(truncated_pdf=extracted.truncated)
        await self._generate(
            report,
            telegram_id=telegram_id,
            text=extracted.text,
            source_key=source_key,
            source_title=filename,
            user_message=user_message,
        )
        report.sources_done = 1
        await self.settings(telegram_id=telegram_id, chat_id=chat_id)
        return report

    async def _generate(
        self,
        report: GenerationReport,
        *,
        telegram_id: int,
        text: str,
        source_key: str,
        source_title: str,
        user_message: str,
    ) -> None:
        drafts: list[flashcards.CardDraft] = []
        parts = flashcards.segments(text)
        if len(parts) > self.max_segments_per_source:
            report.truncated_pdf = True
            parts = parts[: self.max_segments_per_source]
        for part in parts:
            # CLAUDE.md §5: nothing that looks like a personal document goes to
            # a cloud LLM. Checked per segment, so one flagged page of a long
            # PDF doesn't block the rest — and is itself never sent.
            if fired := document_fields.pii_signals(part):
                report.segments_skipped_pii += 1
                logger.warning(
                    "flashcards for user %s: segment of %r NOT sent to the LLM (%s)",
                    telegram_id,
                    source_title,
                    document_fields.describe_pii_signals(fired),
                )
                continue
            try:
                drafts.extend(
                    await flashcards.generate_cards(self.router, part, user_message=user_message)
                )
            except (AllProvidersFailedError, flashcards.FlashcardFormatError) as exc:
                report.segments_failed += 1
                logger.error("flashcard generation failed for %r: %s", source_title, exc)

        if report.segments_failed and not drafts and not report.segments_skipped_pii:
            # Nothing usable came back and nothing was withheld: don't record
            # the source as done, so the user can simply retry later.
            raise FlashcardError(strings.CARDS_GENERATION_FAILED)

        # Same PII gate on what the model produced: card text and every
        # generated wrong option (CLAUDE.md §5) — an option the model invented
        # must not smuggle an IIN/passport number into the plain cards table.
        new_cards: list[NewCard] = []
        for draft in drafts:
            if fired := document_fields.pii_signals(
                f"{draft.front}\n{draft.back}\n{draft.explanation}"
            ):
                report.cards_dropped_pii += 1
                logger.warning(
                    "flashcards for user %s: generated card dropped (%s)",
                    telegram_id,
                    document_fields.describe_pii_signals(fired),
                )
                continue
            options = tuple(
                [o for o in draft.options if not document_fields.has_pii_signals(o)][
                    : quiz_mode.OPTIONS_NEEDED
                ]
            )
            new_cards.append(
                NewCard(
                    front=draft.front,
                    back=draft.back,
                    front_key=flashcards.normalize_front(draft.front),
                    explanation=draft.explanation,
                    options=options,
                )
            )
        created = await self.store.add_cards(
            telegram_id=telegram_id,
            source_key=source_key,
            source_title=source_title,
            cards=new_cards,
            now=utcnow(),
        )
        report.cards.extend(created)
        logger.info(
            "flashcards for user %s from %r: %d new card(s), %d segment(s) withheld (PII), "
            "%d failed",
            telegram_id,
            source_title,
            len(created),
            report.segments_skipped_pii,
            report.segments_failed,
        )

    # -- quiz ----------------------------------------------------------------------------

    async def start_session(
        self, *, telegram_id: int, chat_id: int, kind: str, now: datetime | None = None
    ) -> QuizPrompt | None:
        """First card of a new session, or None if nothing is due."""
        now = now or utcnow()
        settings = await self.settings(telegram_id=telegram_id, chat_id=chat_id)
        if await self.store.next_due_card(telegram_id=telegram_id, now=now) is None:
            return None
        session = await self.store.create_session(
            telegram_id=telegram_id,
            chat_id=chat_id,
            kind=kind,
            card_limit=settings.daily_cap,
            now=now,
        )
        return await self._next_prompt(telegram_id=telegram_id, session_id=session.id, now=now)

    async def _next_prompt(
        self, *, telegram_id: int, session_id: int, now: datetime
    ) -> QuizPrompt | None:
        session = await self.store.get_session(telegram_id=telegram_id, session_id=session_id)
        if session is None:
            return None
        asked, _ = await self.store.session_progress(telegram_id=telegram_id, session_id=session_id)
        if asked >= session.card_limit:
            return None
        card = await self.store.next_due_card(
            telegram_id=telegram_id, now=now, session_id=session_id
        )
        if card is None:
            return None

        decision = await self._decide_mode(telegram_id, card)
        # One line per card asked — ids, lengths and counts only, no card text.
        logger.info(
            "quiz mode user=%s card=%d mode=%s reason=%s answer_len=%d stored_options=%d "
            "candidates=%d",
            telegram_id,
            card.id,
            decision.mode,
            decision.reason,
            decision.answer_len,
            decision.stored_options,
            decision.candidates,
        )
        choices: list[str] = []
        correct_index: int | None = None
        mode = "reveal"
        if decision.mode == "multiple_choice":
            mode = "choice"
            choices = [*decision.options, card.back.strip()]
            random.shuffle(choices)
            correct_index = choices.index(card.back.strip())
        item = await self.store.create_item(
            session_id=session_id,
            telegram_id=telegram_id,
            card_id=card.id,
            mode=mode,
            choices=choices,
            correct_index=correct_index,
            now=now,
        )
        return QuizPrompt(item=item, card=card, position=asked + 1, card_limit=session.card_limit)

    async def _decide_mode(
        self,
        telegram_id: int,
        card: Card,
        pool: list[tuple[int, str, str]] | None = None,
    ) -> quiz_mode.ModeDecision:
        """Own stored options first, then same-source answers, then the user's
        other answers (each group shuffled so options vary between reviews)."""
        if pool is None:
            pool = await self.store.answer_pool(telegram_id=telegram_id)
        same = [
            back for card_id, back, key in pool if card_id != card.id and key == card.source_key
        ]
        other = [
            back for card_id, back, key in pool if card_id != card.id and key != card.source_key
        ]
        random.shuffle(same)
        random.shuffle(other)
        return quiz_mode.choose_mode(card.back, card.options, same, other)

    async def multiple_choice_count(self, *, telegram_id: int) -> int:
        """How many of this user's cards would be asked as multiple choice —
        the same decision the quiz makes, card by card."""
        cards = await self.store.all_cards(telegram_id=telegram_id)
        pool = [(c.id, c.back, c.source_key) for c in cards]
        count = 0
        for card in cards:
            decision = await self._decide_mode(telegram_id, card, pool)
            count += decision.mode == "multiple_choice"
        return count

    async def reveal(self, *, telegram_id: int, item_id: int) -> tuple[QuizItem, Card] | None:
        item = await self.store.get_item(telegram_id=telegram_id, item_id=item_id)
        if item is None:
            return None
        card = await self.store.get_card(telegram_id=telegram_id, card_id=item.card_id)
        if card is None:
            return None
        await self.store.mark_revealed(telegram_id=telegram_id, item_id=item_id, now=utcnow())
        return item, card

    async def answer_choice(
        self, *, telegram_id: int, item_id: int, choice: int, now: datetime | None = None
    ) -> AnswerResult | None:
        item = await self.store.get_item(telegram_id=telegram_id, item_id=item_id)
        if item is None or item.mode != "choice" or not 0 <= choice < len(item.choices):
            return None
        correct = choice == item.correct_index
        grade = CHOICE_CORRECT_GRADE if correct else CHOICE_WRONG_GRADE
        return await self._grade(
            telegram_id, item, grade, correct=correct, chosen=item.choices[choice], now=now
        )

    async def rate(
        self, *, telegram_id: int, item_id: int, grade: int, now: datetime | None = None
    ) -> AnswerResult | None:
        if grade not in RATING_GRADES:
            return None
        item = await self.store.get_item(telegram_id=telegram_id, item_id=item_id)
        if item is None:
            return None
        return await self._grade(telegram_id, item, grade, correct=None, chosen=None, now=now)

    async def _grade(
        self,
        telegram_id: int,
        item: QuizItem,
        grade: int,
        *,
        correct: bool | None,
        chosen: str | None,
        now: datetime | None,
    ) -> AnswerResult | None:
        now = now or utcnow()
        card = await self.store.get_card(telegram_id=telegram_id, card_id=item.card_id)
        if card is None:
            return None
        if not await self.store.claim_answer(
            telegram_id=telegram_id, item_id=item.id, grade=grade, now=now
        ):
            raise FlashcardError(strings.QUIZ_ALREADY_ANSWERED)
        state = sm2.review(card.state, grade)
        due = sm2.due_after(now, state)
        await self.store.save_review(
            telegram_id=telegram_id,
            card_id=card.id,
            state=state,
            due_at=due,
            grade=grade,
            mode=item.mode,
            now=now,
        )
        next_prompt = await self._next_prompt(
            telegram_id=telegram_id, session_id=item.session_id, now=now
        )
        asked, good = await self.store.session_progress(
            telegram_id=telegram_id, session_id=item.session_id
        )
        settings = await self.store.get_settings(telegram_id=telegram_id)
        offset = settings.utc_offset_minutes if settings else self.default_utc_offset_minutes
        return AnswerResult(
            card=card,
            grade=grade,
            correct=correct,
            chosen=chosen,
            next_due_at=due,
            next_due_local=quiz_clock.to_local(due, offset).strftime("%d.%m.%Y"),
            next_prompt=next_prompt,
            session_done=next_prompt is None,
            session_asked=asked,
            session_good=good,
        )

    # -- daily scheduling ----------------------------------------------------------------

    async def due_daily_sessions(
        self, now: datetime | None = None
    ) -> list[tuple[QuizSettings, QuizPrompt | None]]:
        """Claim and start today's daily quiz for every user it is due for.

        Claimed in the DB *before* anything is sent: a crash or a failed send
        never re-sends on the next tick, and a user with nothing due is simply
        marked done for the day (prompt None, nothing to send)."""
        now = now or utcnow()
        started: list[tuple[QuizSettings, QuizPrompt | None]] = []
        for settings in await self.store.enabled_settings():
            if not daily_due(settings, now):
                continue
            local_date = local_now(settings, now).date().isoformat()
            if not await self.store.claim_daily(
                telegram_id=settings.telegram_id, local_date=local_date
            ):
                continue
            prompt = await self.start_session(
                telegram_id=settings.telegram_id, chat_id=settings.chat_id, kind="daily", now=now
            )
            started.append((settings, prompt))
        return started

    # -- stats, settings, deletion -----------------------------------------------------

    async def stats(
        self, *, telegram_id: int, chat_id: int, now: datetime | None = None
    ) -> tuple[CardStats, QuizSettings]:
        now = now or utcnow()
        settings = await self.settings(telegram_id=telegram_id, chat_id=chat_id)
        local = local_now(settings, now)
        local_midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start = local_midnight - timedelta(minutes=settings.utc_offset_minutes)
        stats = await self.store.stats(telegram_id=telegram_id, now=now, day_start=day_start)
        return stats, settings

    async def get_card(self, *, telegram_id: int, card_id: int) -> Card | None:
        return await self.store.get_card(telegram_id=telegram_id, card_id=card_id)

    async def delete_card(self, *, telegram_id: int, card_id: int) -> Card | None:
        card = await self.store.delete_card(telegram_id=telegram_id, card_id=card_id)
        if card is not None:
            logger.info("deleted card %d for user %s", card_id, telegram_id)
        return card
