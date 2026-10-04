"""Telegram bot layer (aiogram 3.x).

It knows nothing about LLMs or storage internals — every request goes through
the `Harness` instance injected into handlers as workflow data.
"""

from __future__ import annotations

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

import strings
from bot.handlers import build_root_router
from bot.middlewares import AccessMiddleware, UserTrackingMiddleware
from config import Settings
from harness import DocumentArchive, FlashcardService, Harness, KnowledgeBase, MediaPipeline
from storage import Storage

BOT_COMMANDS = [
    BotCommand(command="tutor", description=strings.CMD_TUTOR_DESC),
    BotCommand(command="summarize", description=strings.CMD_SUMMARIZE_DESC),
    BotCommand(command="download", description=strings.CMD_DOWNLOAD_DESC),
    BotCommand(command="find", description=strings.CMD_FIND_DESC),
    BotCommand(command="delete", description=strings.CMD_DELETE_DESC),
    BotCommand(command="ask", description=strings.CMD_ASK_DESC),
    BotCommand(command="page", description=strings.CMD_PAGE_DESC),
    BotCommand(command="cards", description=strings.CMD_CARDS_DESC),
    BotCommand(command="quiz", description=strings.CMD_QUIZ_DESC),
    BotCommand(command="quiztime", description=strings.CMD_QUIZTIME_DESC),
    BotCommand(command="delcard", description=strings.CMD_DELCARD_DESC),
    BotCommand(command="reset", description=strings.CMD_RESET_DESC),
    BotCommand(command="cancel", description=strings.CMD_CANCEL_DESC),
    BotCommand(command="help", description=strings.CMD_HELP_DESC),
]


def create_bot(settings: Settings) -> Bot:
    # parse_mode=None by default: static strings and errors go out as plain
    # text. Model/note content opts into HTML per call via bot/formatting.py,
    # which escapes it and falls back to plain text if Telegram rejects it.
    return Bot(token=settings.bot_token, default=DefaultBotProperties(parse_mode=None))


def create_dispatcher(
    *,
    settings: Settings,
    orchestrator: Harness,
    storage: Storage,
    archive: DocumentArchive,
    media: MediaPipeline,
    knowledge: KnowledgeBase | None,
    flashcards: FlashcardService | None,
) -> Dispatcher:
    # MemoryStorage: FSM position resets on restart, which is fine — the actual
    # conversation lives in SQLite and is reloaded on the next turn.
    dp = Dispatcher(
        storage=MemoryStorage(),
        harness=orchestrator,
        settings=settings,
        archive=archive,
        media=media,
        knowledge=knowledge,
        flashcards=flashcards,
    )

    dp.message.outer_middleware(AccessMiddleware(settings))
    dp.message.outer_middleware(UserTrackingMiddleware(storage))
    # Inline buttons (Phase 5b quizzes) arrive as callback queries, which the
    # message middlewares never see — gate them with the same whitelist.
    dp.callback_query.outer_middleware(AccessMiddleware(settings))

    dp.include_router(build_root_router())
    return dp


async def set_bot_commands(bot: Bot) -> None:
    await bot.set_my_commands(BOT_COMMANDS)


__all__ = ["create_bot", "create_dispatcher", "set_bot_commands"]
