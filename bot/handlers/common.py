"""Commands that are not part of the tutor dialogue itself."""

from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

import strings
from bot.states import TutorFlow
from harness import Harness

logger = logging.getLogger(__name__)

router = Router(name="common")


@router.message(CommandStart())
async def handle_start(message: Message, state: FSMContext) -> None:
    await state.set_state(TutorFlow.waiting_for_topic)
    await message.answer(strings.START_TEXT)


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.answer(strings.HELP_TEXT)


@router.message(Command("reset"))
async def handle_reset(message: Message, state: FSMContext, harness: Harness) -> None:
    assert message.from_user is not None
    deleted = await harness.reset(telegram_id=message.from_user.id, chat_id=message.chat.id)
    await state.set_state(TutorFlow.waiting_for_topic)
    await message.answer(strings.RESET_DONE.format(count=deleted))


@router.message(Command("cancel"))
async def handle_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(strings.LESSON_CANCELLED)
