"""Entry point: wire up storage, the LLM router, the harness and the bot.

Run with:  python main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from bot import create_bot, create_dispatcher, set_bot_commands
from config import ConfigError, Settings, load_settings
from harness import Harness, LinkSummarizer
from llm_router import build_router
from storage import Storage
from tools.transcriber import Transcriber

logger = logging.getLogger("harness")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # aiogram's polling loop is chatty at INFO.
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # yt-dlp logs every step at INFO/DEBUG through its own logger.
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)


async def run(settings: Settings) -> None:
    storage = Storage(settings.db_path)
    await storage.connect()

    llm = build_router(settings)
    transcriber = Transcriber(
        backend=settings.stt_backend,
        groq_api_key=settings.groq.api_key,
        groq_model=settings.groq_whisper_model,
        local_model=settings.whisper_local_model,
        ffmpeg_path=settings.ffmpeg_path,
    )
    logger.info(transcriber.describe())
    links = LinkSummarizer(
        router=llm,
        transcriber=transcriber,
        download_dir=settings.download_dir,
        max_video_minutes=settings.max_video_minutes,
    )
    orchestrator = Harness(
        router=llm, storage=storage, history_limit=settings.history_limit, links=links
    )

    bot = create_bot(settings)
    dp = create_dispatcher(settings=settings, orchestrator=orchestrator, storage=storage)

    try:
        me = await bot.get_me()
        logger.info("Starting @%s (id=%s)", me.username, me.id)
        if settings.allowed_user_ids:
            logger.info("Access limited to user IDs: %s", sorted(settings.allowed_user_ids))
        else:
            logger.warning("ALLOWED_USER_IDS is empty — the bot replies to anyone")

        await set_bot_commands(bot)
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await llm.aclose()
        await storage.close()
        await bot.session.close()
        logger.info("Shut down cleanly")


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    setup_logging(settings.log_level)
    try:
        asyncio.run(run(settings))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Interrupted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
