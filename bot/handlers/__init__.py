"""Telegram handlers, grouped into routers.

Order matters: `common` claims the commands, `tutor` catches everything else.
"""

from aiogram import Router

from bot.handlers import common, tutor


def build_root_router() -> Router:
    root = Router(name="root")
    root.include_router(common.router)
    root.include_router(tutor.router)
    return root


__all__ = ["build_root_router"]
