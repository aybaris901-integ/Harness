"""Telegram handlers, grouped into routers.

Order matters: `common` claims the commands, `links` claims `/summarize` and
any *non-command* message with a URL (regardless of FSM state), `media` claims
`/download` and photos captioned `/note`, `knowledge` claims `/ask` and photos
captioned `/page` (both must come before `documents` so they can claim those
photos first), `documents` claims every other photo plus `/find` and `/delete`, `tutor`
catches everything else.

`links` sits before `media`/`documents`, so its URL catch-all must not match
commands: `HasUrl` rejects any text/caption starting with "/". Without that,
`/download <url>` was swallowed by the Phase 2 summarizer.
"""

from aiogram import Router

from bot.handlers import common, documents, knowledge, links, media, tutor


def build_root_router() -> Router:
    root = Router(name="root")
    root.include_router(common.router)
    root.include_router(links.router)
    root.include_router(media.router)
    root.include_router(knowledge.router)
    root.include_router(documents.router)
    root.include_router(tutor.router)
    return root


__all__ = ["build_root_router"]
