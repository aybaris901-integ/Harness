"""Commands that are not part of the tutor dialogue itself."""

from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.states import TutorFlow
from harness import Harness

logger = logging.getLogger(__name__)

router = Router(name="common")

START_TEXT = (
    "Привет! Я твой репетитор.\n\n"
    "Напиши тему, которую хочешь разобрать — например «рекурсия» или "
    "«закон Ома», — и я объясню её на примере и задам вопрос для проверки.\n\n"
    "Или пришли ссылку на статью или видео — сделаю краткий пересказ.\n\n"
    "/help — что я умею"
)

HELP_TEXT = (
    "Что я умею сейчас (Phase 2):\n\n"
    "• Просто напиши тему — начнётся урок.\n"
    "• /tutor <тема> — начать урок явно.\n"
    "• Пришли ссылку на статью — получишь пересказ с ключевыми пунктами.\n"
    "• Пришли ссылку на видео (YouTube, TikTok, Instagram…) — пересказ по "
    "субтитрам, а если их нет — по распознанной речи (это дольше).\n"
    "• /summarize <ссылка> — то же самое явно.\n"
    "• /reset — забыть нашу переписку и начать с чистого листа.\n"
    "• /cancel — выйти из текущего урока.\n\n"
    "Отвечаю на том языке, на котором пишешь ты."
)


@router.message(CommandStart())
async def handle_start(message: Message, state: FSMContext) -> None:
    await state.set_state(TutorFlow.waiting_for_topic)
    await message.answer(START_TEXT)


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("reset"))
async def handle_reset(message: Message, state: FSMContext, harness: Harness) -> None:
    assert message.from_user is not None
    deleted = await harness.reset(telegram_id=message.from_user.id, chat_id=message.chat.id)
    await state.set_state(TutorFlow.waiting_for_topic)
    await message.answer(f"История очищена ({deleted} сообщений). О чём поговорим теперь?")


@router.message(Command("cancel"))
async def handle_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Урок остановлен. Напиши новую тему, когда захочешь продолжить.")
