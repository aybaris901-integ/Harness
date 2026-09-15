"""Telegram bot layer (aiogram 3.x).

It knows nothing about LLMs or storage internals — every request goes through
the `Harness` instance injected into handlers as workflow data.
"""

from __future__ import annotations

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from bot.handlers import build_root_router
from bot.middlewares import AccessMiddleware, UserTrackingMiddleware
from config import Settings
from harness import Harness
from storage import Storage

BOT_COMMANDS = [
    BotCommand(command="tutor", description="Начать урок по теме"),
    BotCommand(command="reset", description="Очистить историю диалога"),
    BotCommand(command="cancel", description="Выйти из текущего урока"),
    BotCommand(command="help", description="Что умеет бот"),
]


def create_bot(settings: Settings) -> Bot:
    # parse_mode=None: model output is sent verbatim as plain text, so stray
    # Markdown characters can never make a send fail.
    return Bot(token=settings.bot_token, default=DefaultBotProperties(parse_mode=None))


def create_dispatcher(*, settings: Settings, orchestrator: Harness, storage: Storage) -> Dispatcher:
    # MemoryStorage: FSM position resets on restart, which is fine — the actual
    # conversation lives in SQLite and is reloaded on the next turn.
    dp = Dispatcher(storage=MemoryStorage(), harness=orchestrator, settings=settings)

    dp.message.outer_middleware(AccessMiddleware(settings))
    dp.message.outer_middleware(UserTrackingMiddleware(storage))

    dp.include_router(build_root_router())
    return dp


async def set_bot_commands(bot: Bot) -> None:
    await bot.set_my_commands(BOT_COMMANDS)


__all__ = ["create_bot", "create_dispatcher", "set_bot_commands"]
