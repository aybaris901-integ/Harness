"""Middlewares applied to every incoming update."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject, User

import strings
from config import Settings
from storage import Storage, UserProfile

logger = logging.getLogger(__name__)


class AccessMiddleware(BaseMiddleware):
    """Whitelist gate — this is a family bot, not a public one.

    An empty `ALLOWED_USER_IDS` leaves it open (handy while testing).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user: User | None = data.get("event_from_user")
        if user is None:
            return await handler(event, data)

        if not self.settings.is_user_allowed(user.id):
            logger.warning("Blocked user %s (@%s)", user.id, user.username)
            if isinstance(event, Message):
                await event.answer(strings.ACCESS_DENIED.format(user_id=user.id))
            return None

        return await handler(event, data)


class UserTrackingMiddleware(BaseMiddleware):
    """Keep the `users` table current on every interaction."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user: User | None = data.get("event_from_user")
        if user is not None and not user.is_bot:
            await self.storage.upsert_user(
                UserProfile(
                    telegram_id=user.id,
                    username=user.username,
                    first_name=user.first_name,
                    language_code=user.language_code,
                )
            )
        return await handler(event, data)
