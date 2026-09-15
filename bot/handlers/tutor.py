"""The tutor dialogue (CLAUDE.md §7 Phase 1).

State machine:

    /start, /reset            -> waiting_for_topic
    topic text                -> in_lesson   (explanation + check-question sent)
    answer text (in_lesson)   -> in_lesson   (evaluation + next step)
    /cancel                   -> no state

Handlers here do no LLM or storage work of their own — they only translate
Telegram events into `Harness` calls.
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from bot.states import TutorFlow
from bot.utils import split_message
from harness import Harness, HarnessError

logger = logging.getLogger(__name__)

router = Router(name="tutor")

ASK_TOPIC_TEXT = "Какую тему разбираем? Напиши её одним сообщением."
GENERIC_ERROR_TEXT = "Что-то пошло не так. Попробуй ещё раз."


async def _reply(message: Message, text: str) -> None:
    # Model output is sent as plain text: it may contain Markdown that Telegram's
    # strict MarkdownV2 parser would reject and turn into a failed send.
    for chunk in split_message(text):
        await message.answer(chunk)


async def _run_turn(message: Message, state: FSMContext, harness: Harness, *, topic: bool) -> None:
    assert message.from_user is not None and message.text is not None
    text = message.text.strip()
    if not text:
        await message.answer(ASK_TOPIC_TEXT)
        return

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            if topic:
                reply = await harness.start_lesson(
                    telegram_id=message.from_user.id, chat_id=message.chat.id, topic=text
                )
            else:
                reply = await harness.continue_lesson(
                    telegram_id=message.from_user.id, chat_id=message.chat.id, text=text
                )
        except HarnessError as exc:
            await message.answer(str(exc))
            return
        except Exception:
            logger.exception("Unexpected failure in tutor turn")
            await message.answer(GENERIC_ERROR_TEXT)
            return

    await state.set_state(TutorFlow.in_lesson)
    await _reply(message, reply)


@router.message(Command("tutor"))
async def handle_tutor_command(
    message: Message, command: CommandObject, state: FSMContext, harness: Harness
) -> None:
    """`/tutor` asks for a topic; `/tutor <topic>` starts the lesson right away."""
    topic = (command.args or "").strip()
    if not topic:
        await state.set_state(TutorFlow.waiting_for_topic)
        await message.answer(ASK_TOPIC_TEXT)
        return

    assert message.from_user is not None
    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            reply = await harness.start_lesson(
                telegram_id=message.from_user.id, chat_id=message.chat.id, topic=topic
            )
        except HarnessError as exc:
            await message.answer(str(exc))
            return
        except Exception:
            logger.exception("Unexpected failure starting lesson")
            await message.answer(GENERIC_ERROR_TEXT)
            return

    await state.set_state(TutorFlow.in_lesson)
    await _reply(message, reply)


@router.message(TutorFlow.in_lesson, F.text)
async def handle_answer(message: Message, state: FSMContext, harness: Harness) -> None:
    await _run_turn(message, state, harness, topic=False)


@router.message(F.text)
async def handle_topic(message: Message, state: FSMContext, harness: Harness) -> None:
    """Any other text — whether or not `/start` was used — opens a new lesson."""
    await _run_turn(message, state, harness, topic=True)


@router.message()
async def handle_unsupported(message: Message) -> None:
    await message.answer(
        "Пока я понимаю только текст. Ссылки, фото и документы появятся в следующих фазах."
    )
