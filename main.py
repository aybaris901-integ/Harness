"""Entry point: wire up storage, the LLM router, the harness and the bot.

Run with:  python main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from bot import create_bot, create_dispatcher, set_bot_commands
from bot.quiz import QuizScheduler
from config import ConfigError, Settings, load_settings
from harness import (
    DocumentArchive,
    FlashcardService,
    Harness,
    KnowledgeBase,
    LinkSummarizer,
    MediaPipeline,
)
from llm_router import build_embedder, build_router, build_vision_router
from storage import DocumentStore, NoteStore, Storage
from storage.flashcards import FlashcardStore
from storage.knowledge import KnowledgeStore, KnowledgeStoreError
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

    document_store = DocumentStore(
        db_path=settings.documents_db_path,
        scan_dir=settings.documents_scan_dir,
        encryption_key=settings.documents_encryption_key,
    )
    await document_store.connect()
    archive = DocumentArchive(
        store=document_store,
        tesseract_cmd=settings.tesseract_cmd,
        ocr_lang=settings.ocr_lang,
        ocr_tessdata_dir=settings.ocr_tessdata_dir,
        local_llm_enabled=settings.document_local_llm_enabled,
        local_llm_base_url=settings.document_local_llm_base_url,
        local_llm_model=settings.document_local_llm_model,
    )
    logger.info(archive.describe())

    note_store = NoteStore(settings.notes_db_path)
    await note_store.connect()

    llm = build_router(settings)

    # Phase 5a: the knowledge base needs an embedding model; without a Gemini
    # key it is disabled and /ask, /page reply with a static notice.
    embedder = build_embedder(settings)
    vision = build_vision_router(settings)
    knowledge_store: KnowledgeStore | None = None
    knowledge: KnowledgeBase | None = None
    if embedder is None:
        logger.warning("GEMINI_API_KEY not set — knowledge base (/ask, /page) disabled")
    else:
        knowledge_store = KnowledgeStore(
            db_path=settings.knowledge_db_path,
            page_dir=settings.knowledge_page_dir,
            encryption_key=settings.documents_encryption_key,
            embedding_model=embedder.model_id,
            dimensions=settings.embedding_dim,
        )
        try:
            await knowledge_store.connect()
        except KnowledgeStoreError as exc:
            # Refuse to start rather than silently run without the index the
            # user expects — the message says exactly what to fix.
            raise SystemExit(f"Knowledge store error: {exc}") from exc
        knowledge = KnowledgeBase(
            store=knowledge_store,
            embedder=embedder,
            router=llm,
            vision_router=vision,
            note_store=note_store,
            document_archive=archive,
            tesseract_cmd=settings.tesseract_cmd,
            ocr_lang=settings.ocr_lang,
            ocr_tessdata_dir=settings.ocr_tessdata_dir,
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
        )
        logger.info(knowledge.describe())
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
    media = MediaPipeline(
        router=llm,
        download_dir=settings.download_dir,
        note_store=note_store,
        ffmpeg_path=settings.ffmpeg_path,
        tesseract_cmd=settings.tesseract_cmd,
        ocr_lang=settings.ocr_lang,
        ocr_tessdata_dir=settings.ocr_tessdata_dir,
        max_download_mb=settings.media_max_download_mb,
        max_concurrent_downloads=settings.media_max_concurrent,
        document_archive=archive,
        knowledge=knowledge,
    )
    logger.info(media.describe())

    # Phase 5b: flashcards + spaced repetition. Sources come from the knowledge
    # base when it is enabled; PDFs work either way.
    flashcard_store = FlashcardStore(settings.flashcards_db_path)
    await flashcard_store.connect()
    flashcards = FlashcardService(
        store=flashcard_store,
        router=llm,
        knowledge_store=knowledge_store,
        default_quiz_time=settings.quiz_default_time,
        default_utc_offset_minutes=settings.quiz_utc_offset_minutes,
        default_daily_cap=settings.quiz_daily_cap,
        max_sources_per_run=settings.cards_max_sources_per_run,
        pdf_max_mb=settings.pdf_max_mb,
    )
    logger.info(flashcards.describe())

    bot = create_bot(settings)
    dp = create_dispatcher(
        settings=settings,
        orchestrator=orchestrator,
        storage=storage,
        archive=archive,
        media=media,
        knowledge=knowledge,
        flashcards=flashcards,
    )
    # Daily quiz: due-ness is recomputed from the DB every tick, so starting
    # it is all a restart needs.
    quiz_scheduler = QuizScheduler(bot, flashcards)

    migration: asyncio.Task[int] | None = None
    if knowledge is not None:
        # Phase 4 staging notes -> index. Idempotent, so it runs every start;
        # in the background so embedding calls never delay polling.
        migration = asyncio.create_task(knowledge.migrate_staging_notes(), name="kb-migration")

    try:
        me = await bot.get_me()
        logger.info("Starting @%s (id=%s)", me.username, me.id)
        if settings.allowed_user_ids:
            logger.info("Access limited to user IDs: %s", sorted(settings.allowed_user_ids))
        else:
            logger.warning("ALLOWED_USER_IDS is empty — the bot replies to anyone")

        await set_bot_commands(bot)
        quiz_scheduler.start()
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await quiz_scheduler.stop()
        if migration is not None and not migration.done():
            migration.cancel()
        await llm.aclose()
        if vision is not None:
            await vision.aclose()
        if embedder is not None:
            await embedder.aclose()
        if knowledge_store is not None:
            await knowledge_store.close()
        await storage.close()
        await document_store.close()
        await note_store.close()
        await flashcard_store.close()
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
