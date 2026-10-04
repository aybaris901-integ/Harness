"""Quiz rendering + the daily-quiz scheduler loop (CLAUDE.md §7 Phase 5b).

Callback data (≤64 bytes), all prefixed "fc:":
    fc:a:<item>:<choice>   multiple-choice answer
    fc:s:<item>            show answer (reveal mode)
    fc:r:<item>:<grade>    self-rating after reveal (SM-2 grade)
    fc:d:<card>:y|n        /delcard confirmation
Ids are DB ids; the handlers resolve them with the presser's own Telegram id,
so a forwarded button can never touch someone else's card or question.

The scheduler is a plain asyncio loop that asks the service "who is due now?"
every minute. That answer comes from SQLite alone (quiz settings + last sent
date + card due dates), so there is no in-memory job state to lose on a
restart — the reason this is not APScheduler.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import strings
from bot.formatting import escape
from harness.flashcards import AnswerResult, FlashcardService, QuizPrompt
from storage.flashcards import Card

logger = logging.getLogger(__name__)

TICK_SECONDS = 60


def _question_html(prompt: QuizPrompt) -> str:
    card = prompt.card
    return (
        escape(
            strings.QUIZ_CARD_HEADER.format(
                card_id=card.id, position=prompt.position, limit=prompt.card_limit
            )
        )
        + f"\n\n<b>{escape(card.front)}</b>\n\n<i>"
        + escape(strings.QUIZ_SOURCE.format(title=card.source_title))
        + "</i>"
    )


def _explanation_html(card: Card) -> str:
    """The optional explanation, shown after the answer — skipped when it just
    repeats the answer (migrated cards keep their old long answer in both)."""
    explanation = (card.explanation or "").strip()
    if not explanation or explanation.casefold() == card.back.strip().casefold():
        return ""
    return "\n" + escape(strings.QUIZ_EXPLANATION.format(explanation=explanation))


def render_prompt(prompt: QuizPrompt) -> tuple[str, InlineKeyboardMarkup]:
    item = prompt.item
    if item.mode == "choice":
        text = _question_html(prompt) + "\n\n" + escape(strings.QUIZ_CHOOSE)
        rows = [
            [InlineKeyboardButton(text=choice, callback_data=f"fc:a:{item.id}:{index}")]
            for index, choice in enumerate(item.choices)
        ]
    else:
        text = _question_html(prompt) + "\n\n" + escape(strings.QUIZ_THINK)
        rows = [
            [InlineKeyboardButton(text=strings.QUIZ_SHOW_ANSWER, callback_data=f"fc:s:{item.id}")]
        ]
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def render_revealed(card: Card, item_id: int) -> tuple[str, InlineKeyboardMarkup]:
    text = (
        f"<b>{escape(card.front)}</b>\n\n<i>"
        + escape(strings.QUIZ_SOURCE.format(title=card.source_title))
        + "</i>\n\n"
        + escape(strings.QUIZ_ANSWER.format(back=card.back))
        + _explanation_html(card)
        + "\n\n"
        + escape(strings.QUIZ_RATE)
    )
    row = [
        InlineKeyboardButton(text=label, callback_data=f"fc:r:{item_id}:{grade}")
        for grade, label in strings.QUIZ_RATING_LABELS.items()
    ]
    return text, InlineKeyboardMarkup(inline_keyboard=[row])


def render_result(result: AnswerResult) -> str:
    card = result.card
    lines = [f"<b>{escape(card.front)}</b>", ""]
    if result.correct is True:
        lines.append(escape(strings.QUIZ_CORRECT))
    elif result.correct is False:
        lines.append(escape(strings.QUIZ_WRONG.format(chosen=result.chosen or "")))
    else:
        label = strings.QUIZ_RATING_LABELS.get(result.grade, str(result.grade))
        lines.append(escape(strings.QUIZ_RATED.format(label=label)))
    lines.append(escape(strings.QUIZ_ANSWER.format(back=card.back)) + _explanation_html(card))
    lines.append(escape(strings.QUIZ_NEXT_REVIEW.format(when=result.next_due_local)))
    return "\n".join(lines)


async def send_prompt(bot: Bot, chat_id: int, prompt: QuizPrompt) -> None:
    text, keyboard = render_prompt(prompt)
    await bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=keyboard)


class QuizScheduler:
    """Every TICK_SECONDS: start today's daily quiz for whoever is due."""

    def __init__(self, bot: Bot, service: FlashcardService) -> None:
        self.bot = bot
        self.service = service
        self._task: asyncio.Task[None] | None = None

    async def tick(self, now: datetime | None = None) -> int:
        """One pass; returns how many quizzes were sent. Safe to call any time."""
        sent = 0
        for settings, prompt in await self.service.due_daily_sessions(now):
            if prompt is None:
                continue  # nothing due today: day claimed, nothing sent
            try:
                await self.bot.send_message(settings.chat_id, strings.QUIZ_DAILY_INTRO)
                await send_prompt(self.bot, settings.chat_id, prompt)
                sent += 1
            except TelegramAPIError as exc:
                # The day is already claimed in the DB, so a blocked bot or a
                # deleted chat is not retried every minute.
                logger.warning(
                    "daily quiz for user %s not delivered: %s", settings.telegram_id, exc
                )
        if sent:
            logger.info("daily quiz sent to %d user(s)", sent)
        return sent

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("quiz scheduler tick failed")
            await asyncio.sleep(TICK_SECONDS)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="quiz-scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
