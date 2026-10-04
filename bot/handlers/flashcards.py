"""Flashcards, spaced repetition, quizzes (CLAUDE.md §7 Phase 5b).

    /cards            stats + daily quiz settings
    /cards make       cards from knowledge-base sources that have none yet
    a PDF document    cards from its text layer (scanned PDF -> /page hint)
    /quiz             start reviewing due cards now
    /quiztime ...     daily quiz time / UTC offset / per-day cap / off
    /delcard <id>     delete a card, after an inline Yes/No confirmation
    fc:* callbacks    quiz answers, reveal, self-rating, delete confirmation

`flashcards` is None when the feature is not wired (it never is without an
LLM router, which the bot can't start without, so in practice always set).
"""

from __future__ import annotations

import logging

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.utils.chat_action import ChatActionSender

import strings
from bot.quiz import render_result, render_revealed, send_prompt
from harness.flashcards import AnswerResult, FlashcardError, FlashcardService, GenerationReport
from tools import quiz_clock

logger = logging.getLogger(__name__)

router = Router(name="flashcards")

_MAX_DAILY_CAP = 50
_CARDS_LISTED = 5


def _report_text(report: GenerationReport, quiz_time: str, offset_minutes: int) -> str:
    lines = (
        [strings.CARDS_CREATED.format(count=len(report.cards), sources=report.sources_done)]
        if report.cards
        else [strings.CARDS_CREATED_NONE]
    )
    for card in report.cards[:_CARDS_LISTED]:
        lines.append(f"• #{card.id} {card.front}")
    if len(report.cards) > _CARDS_LISTED:
        lines.append("…")
    if report.cards_dropped_pii:
        lines.append(strings.CARDS_PII_DROPPED.format(count=report.cards_dropped_pii))
    if report.segments_skipped_pii:
        lines.append(strings.CARDS_PII_SKIPPED.format(count=report.segments_skipped_pii))
    if report.segments_failed:
        lines.append(strings.CARDS_PART_FAILED.format(count=report.segments_failed))
    if report.truncated_pdf:
        lines.append(strings.CARDS_TRUNCATED)
    if report.sources_remaining:
        lines.append(strings.CARDS_MORE_SOURCES.format(count=report.sources_remaining))
    if report.cards:
        lines.append(
            strings.CARDS_FIRST_QUIZ_HINT.format(
                time=quiz_time, offset=quiz_clock.format_utc_offset(offset_minutes)
            )
        )
    return "\n".join(lines)


# -- /cards --------------------------------------------------------------------


@router.message(Command("cards"))
async def handle_cards(
    message: Message, command: CommandObject, flashcards: FlashcardService | None
) -> None:
    assert message.from_user is not None
    if flashcards is None:
        await message.answer(strings.CARDS_GENERIC_ERROR)
        return
    telegram_id, chat_id = message.from_user.id, message.chat.id
    args = (command.args or "").split()

    if args and args[0].lower() == "make":
        status = await message.answer(strings.CARDS_GENERATING)
        try:
            async with ChatActionSender.typing(bot=message.bot, chat_id=chat_id):
                report = await flashcards.make_from_knowledge(
                    telegram_id=telegram_id, chat_id=chat_id, user_message=message.text or ""
                )
        except FlashcardError as exc:
            await status.edit_text(str(exc))
            return
        except Exception:
            logger.exception("Unexpected failure making cards from the knowledge base")
            await status.edit_text(strings.CARDS_GENERIC_ERROR)
            return
        settings = await flashcards.settings(telegram_id=telegram_id, chat_id=chat_id)
        await status.edit_text(
            _report_text(report, settings.quiz_time, settings.utc_offset_minutes)
        )
        return

    stats, settings = await flashcards.stats(telegram_id=telegram_id, chat_id=chat_id)
    if stats.total == 0:
        await message.answer(strings.CARDS_STATS_EMPTY)
        return
    next_due = ""
    if stats.next_due_at is not None:
        when = quiz_clock.to_local(stats.next_due_at, settings.utc_offset_minutes)
        next_due = strings.CARDS_NEXT_DUE.format(when=when.strftime("%d.%m.%Y %H:%M"))
    quiz = (
        strings.QUIZ_TIME_ON.format(
            time=settings.quiz_time,
            offset=quiz_clock.format_utc_offset(settings.utc_offset_minutes),
            cap=settings.daily_cap,
        )
        if settings.enabled
        else strings.QUIZ_TIME_OFF
    )
    await message.answer(
        strings.CARDS_STATS.format(
            total=stats.total,
            due=stats.due_now,
            new=stats.new,
            learned=stats.learned,
            today=stats.reviewed_today,
            mc=await flashcards.multiple_choice_count(telegram_id=telegram_id),
            next_due=next_due,
            quiz=quiz,
        )
    )


# -- PDF documents ---------------------------------------------------------------


def _is_pdf(message: Message) -> bool:
    document = message.document
    if document is None:
        return False
    name = (document.file_name or "").lower()
    return document.mime_type == "application/pdf" or name.endswith(".pdf")


@router.message(F.document, F.func(_is_pdf))
async def handle_pdf(message: Message, flashcards: FlashcardService | None) -> None:
    assert message.from_user is not None and message.bot is not None
    assert message.document is not None
    if flashcards is None:
        await message.answer(strings.CARDS_GENERIC_ERROR)
        return
    size_mb = (message.document.file_size or 0) / 1e6
    if size_mb > flashcards.pdf_max_mb:
        await message.answer(
            strings.CARDS_PDF_TOO_LARGE.format(size=size_mb, limit=flashcards.pdf_max_mb)
        )
        return

    telegram_id, chat_id = message.from_user.id, message.chat.id
    status = await message.answer(strings.CARDS_PDF_RECEIVED)
    data = (await message.bot.download(message.document)).read()
    try:
        async with ChatActionSender.typing(bot=message.bot, chat_id=chat_id):
            report = await flashcards.make_from_pdf(
                telegram_id=telegram_id,
                chat_id=chat_id,
                data=data,
                filename=message.document.file_name or "PDF",
                # The caption (if any) decides the card language, per §8.
                user_message=message.caption or "",
            )
    except FlashcardError as exc:
        await status.edit_text(str(exc))
        return
    except Exception:
        logger.exception("Unexpected failure making cards from a PDF")
        await status.edit_text(strings.CARDS_GENERIC_ERROR)
        return
    settings = await flashcards.settings(telegram_id=telegram_id, chat_id=chat_id)
    await status.edit_text(_report_text(report, settings.quiz_time, settings.utc_offset_minutes))


# -- /quiz ----------------------------------------------------------------------------


@router.message(Command("quiz"))
async def handle_quiz(message: Message, flashcards: FlashcardService | None) -> None:
    assert message.from_user is not None and message.bot is not None
    if flashcards is None:
        await message.answer(strings.CARDS_GENERIC_ERROR)
        return
    telegram_id, chat_id = message.from_user.id, message.chat.id
    prompt = await flashcards.start_session(telegram_id=telegram_id, chat_id=chat_id, kind="manual")
    if prompt is not None:
        await send_prompt(message.bot, chat_id, prompt)
        return
    stats, settings = await flashcards.stats(telegram_id=telegram_id, chat_id=chat_id)
    if stats.total == 0:
        await message.answer(strings.QUIZ_NO_CARDS)
        return
    next_due = ""
    if stats.next_due_at is not None:
        when = quiz_clock.to_local(stats.next_due_at, settings.utc_offset_minutes)
        next_due = strings.QUIZ_NOTHING_DUE_NEXT.format(when=when.strftime("%d.%m.%Y %H:%M"))
    await message.answer(strings.QUIZ_NOTHING_DUE.format(next_due=next_due))


# -- /quiztime -------------------------------------------------------------------------


@router.message(Command("quiztime"))
async def handle_quiztime(
    message: Message, command: CommandObject, flashcards: FlashcardService | None
) -> None:
    assert message.from_user is not None
    if flashcards is None:
        await message.answer(strings.CARDS_GENERIC_ERROR)
        return
    telegram_id, chat_id = message.from_user.id, message.chat.id
    settings = await flashcards.settings(telegram_id=telegram_id, chat_id=chat_id)
    args = (command.args or "").split()
    store = flashcards.store

    if not args:
        current = (
            strings.QUIZ_TIME_ON.format(
                time=settings.quiz_time,
                offset=quiz_clock.format_utc_offset(settings.utc_offset_minutes),
                cap=settings.daily_cap,
            )
            if settings.enabled
            else strings.QUIZ_TIME_OFF
        )
        await message.answer(strings.QUIZTIME_USAGE.format(current=current))
        return

    if args[0].lower() == "off":
        await store.update_settings(telegram_id=telegram_id, enabled=False)
        await message.answer(strings.QUIZTIME_OFF)
        return

    if args[0].lower() == "cap":
        if len(args) != 2 or not args[1].isdigit() or not 1 <= int(args[1]) <= _MAX_DAILY_CAP:
            await message.answer(strings.QUIZTIME_BAD_CAP.format(max=_MAX_DAILY_CAP))
            return
        await store.update_settings(telegram_id=telegram_id, daily_cap=int(args[1]))
        await message.answer(strings.QUIZTIME_CAP_SET.format(cap=int(args[1])))
        return

    quiz_time = quiz_clock.parse_quiz_time(args[0])
    offset = quiz_clock.parse_utc_offset(args[1]) if len(args) > 1 else settings.utc_offset_minutes
    if quiz_time is None or offset is None or len(args) > 2:
        await message.answer(strings.QUIZTIME_BAD)
        return
    # last_daily_date is left untouched: a time change never produces a second
    # quiz on a day that already had one (and one not yet sent today fires at
    # the new time).
    await store.update_settings(
        telegram_id=telegram_id, quiz_time=quiz_time, utc_offset_minutes=offset, enabled=True
    )
    await message.answer(
        strings.QUIZTIME_SET.format(time=quiz_time, offset=quiz_clock.format_utc_offset(offset))
    )


# -- /delcard ----------------------------------------------------------------------------


@router.message(Command("delcard"))
async def handle_delcard(
    message: Message, command: CommandObject, flashcards: FlashcardService | None
) -> None:
    assert message.from_user is not None
    if flashcards is None:
        await message.answer(strings.CARDS_GENERIC_ERROR)
        return
    raw = (command.args or "").strip().lstrip("#")
    if not raw.isdigit():
        await message.answer(strings.DELCARD_USAGE)
        return
    card = await flashcards.get_card(telegram_id=message.from_user.id, card_id=int(raw))
    if card is None:
        await message.answer(strings.DELCARD_NOT_FOUND.format(id=int(raw)))
        return
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=strings.DELCARD_YES, callback_data=f"fc:d:{card.id}:y"),
                InlineKeyboardButton(text=strings.DELCARD_NO, callback_data=f"fc:d:{card.id}:n"),
            ]
        ]
    )
    await message.answer(
        strings.DELCARD_CONFIRM.format(id=card.id, front=card.front, back=card.back),
        reply_markup=keyboard,
    )


# -- callbacks ------------------------------------------------------------------------------


async def _edit(callback: CallbackQuery, text: str, *, html: bool, markup=None) -> None:
    """Edit the message the button sits on; fall back to a new message."""
    message = callback.message
    mode = "HTML" if html else None
    if isinstance(message, Message):
        try:
            await message.edit_text(text, parse_mode=mode, reply_markup=markup)
            return
        except TelegramBadRequest:
            pass
    assert callback.bot is not None
    await callback.bot.send_message(
        callback.from_user.id, text, parse_mode=mode, reply_markup=markup
    )


async def _after_answer(callback: CallbackQuery, bot: Bot, result: AnswerResult) -> None:
    await _edit(callback, render_result(result), html=True)
    chat_id = callback.message.chat.id if callback.message else callback.from_user.id
    if result.next_prompt is not None:
        await send_prompt(bot, chat_id, result.next_prompt)
    else:
        await bot.send_message(
            chat_id,
            strings.QUIZ_SESSION_DONE.format(good=result.session_good, asked=result.session_asked),
        )


@router.callback_query(F.data.startswith("fc:"))
async def handle_callback(callback: CallbackQuery, flashcards: FlashcardService | None) -> None:
    assert callback.bot is not None
    if flashcards is None or callback.data is None:
        await callback.answer()
        return
    telegram_id = callback.from_user.id
    parts = callback.data.split(":")
    action, args = parts[1] if len(parts) > 1 else "", parts[2:]
    if not args or not all(arg.isdigit() or arg in ("y", "n") for arg in args):
        await callback.answer()
        return

    try:
        if action == "a" and len(args) == 2:
            result = await flashcards.answer_choice(
                telegram_id=telegram_id, item_id=int(args[0]), choice=int(args[1])
            )
            if result is None:
                await callback.answer(strings.QUIZ_ITEM_GONE, show_alert=True)
                return
            await callback.answer(strings.QUIZ_CORRECT if result.correct else "❌")
            await _after_answer(callback, callback.bot, result)

        elif action == "s" and len(args) == 1:
            revealed = await flashcards.reveal(telegram_id=telegram_id, item_id=int(args[0]))
            if revealed is None:
                await callback.answer(strings.QUIZ_ITEM_GONE, show_alert=True)
                return
            item, card = revealed
            if item.answered:
                await callback.answer(strings.QUIZ_ALREADY_ANSWERED, show_alert=True)
                return
            await callback.answer()
            text, markup = render_revealed(card, item.id)
            await _edit(callback, text, html=True, markup=markup)

        elif action == "r" and len(args) == 2:
            result = await flashcards.rate(
                telegram_id=telegram_id, item_id=int(args[0]), grade=int(args[1])
            )
            if result is None:
                await callback.answer(strings.QUIZ_ITEM_GONE, show_alert=True)
                return
            await callback.answer()
            await _after_answer(callback, callback.bot, result)

        elif action == "d" and len(args) == 2 and args[0].isdigit():
            card_id = int(args[0])
            if args[1] == "n":
                await callback.answer()
                await _edit(callback, strings.DELCARD_CANCELLED, html=False)
                return
            card = await flashcards.delete_card(telegram_id=telegram_id, card_id=card_id)
            await callback.answer()
            await _edit(
                callback,
                strings.DELCARD_DONE.format(id=card_id)
                if card is not None
                else strings.DELCARD_NOT_FOUND.format(id=card_id),
                html=False,
            )
        else:
            await callback.answer()
    except FlashcardError as exc:
        await callback.answer(str(exc), show_alert=True)
    except Exception:
        logger.exception("Unexpected failure handling quiz callback %r", callback.data)
        await callback.answer(strings.CARDS_GENERIC_ERROR, show_alert=True)
