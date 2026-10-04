"""RAG knowledge base (CLAUDE.md §7 Phase 5a).

- A photo captioned `/page [title]` is a notebook page: local PII gate, then
  handwriting transcription, chunk, embed, index — see harness/knowledge.py.
  This router is included before `documents.router` so it can claim those
  photos; uncaptioned photos still go to the document archive unchanged.
- `/ask <question>` answers only from this user's own notes and pages, with
  the source reference and the original page photo when there is one.

`knowledge` is None when the knowledge base is not configured (no Gemini key
for embeddings); both handlers then reply with a static notice.
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import BaseFilter, Command, CommandObject
from aiogram.types import BufferedInputFile, Message
from aiogram.utils.chat_action import ChatActionSender

import strings
from bot.formatting import answer_html, escape, render_chunks
from bot.handlers.documents import format_document_saved
from harness.knowledge import AnswerStatus, KnowledgeAnswer, KnowledgeBase, KnowledgeError
from storage.knowledge import KnowledgeSource
from tools.language import user_words

logger = logging.getLogger(__name__)

router = Router(name="knowledge")


class HasPageCaption(BaseFilter):
    """A photo captioned `/page` — a notebook page for the knowledge base."""

    async def __call__(self, message: Message) -> bool:
        if not message.photo:
            return False
        caption = (message.caption or "").strip().lower()
        return caption == "/page" or caption.startswith(("/page ", "/page@"))


def _source_line(source: KnowledgeSource) -> str:
    template = strings.KB_SOURCE_PAGE if source.kind == "page" else strings.KB_SOURCE_NOTE
    return "• " + template.format(title=source.title, date=source.created_at[:10])


def _answer_text(result: KnowledgeAnswer) -> str:
    sources = "\n".join(_source_line(source) for source in result.sources)
    return f"{result.answer}\n\n{strings.KB_SOURCES_HEADER}\n{sources}"


@router.message(F.photo, HasPageCaption())
async def handle_page_photo(message: Message, knowledge: KnowledgeBase | None) -> None:
    assert message.from_user is not None and message.bot is not None
    if knowledge is None:
        await message.answer(strings.KB_NOT_CONFIGURED)
        return
    status = await message.answer(strings.KB_PAGE_PROCESSING)

    photo = message.photo[-1]  # largest resolution, best for OCR
    image_bytes = (await message.bot.download(photo)).read()
    # Anything after "/page" is an optional title for the page.
    title_hint = user_words(message.caption)

    try:
        result = await knowledge.ingest_page(
            telegram_id=message.from_user.id, image_bytes=image_bytes, title_hint=title_hint
        )
    except KnowledgeError as exc:
        await status.edit_text(str(exc))
        return
    except Exception:
        logger.exception("Unexpected failure indexing a notebook page")
        await status.edit_text(strings.KB_GENERIC_ERROR)
        return

    if result.redirected_to_documents:
        assert result.document is not None
        await status.edit_text(
            strings.KB_PII_REDIRECTED_AFTER_VISION
            if result.sent_to_vision
            else strings.KB_PII_REDIRECTED
        )
        await message.answer(format_document_saved(result.document))
        return

    assert result.source is not None
    text = strings.KB_PAGE_SAVED.format(title=result.source.title, chunks=result.chunk_count)
    if result.transcribed_by == "tesseract":
        text += "\n" + strings.KB_PAGE_SAVED_TESSERACT_NOTE
    await status.edit_text(text)


@router.message(Command("page"))
async def handle_page_usage(message: Message) -> None:
    # "/page" without a photo arrives as plain text, which HasPageCaption never sees.
    await message.answer(strings.KB_PAGE_USAGE)


@router.message(Command("ask"))
async def handle_ask(
    message: Message, command: CommandObject, knowledge: KnowledgeBase | None
) -> None:
    assert message.from_user is not None
    question = (command.args or "").strip()
    if not question:
        await message.answer(strings.KB_ASK_USAGE)
        return
    if knowledge is None:
        await message.answer(strings.KB_NOT_CONFIGURED)
        return

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            result = await knowledge.ask(
                telegram_id=message.from_user.id,
                question=question,
                # Raw text: tools.language strips "/ask" before picking the language.
                user_message=message.text or "",
            )
        except KnowledgeError as exc:
            await message.answer(str(exc))
            return
        except Exception:
            logger.exception("Unexpected failure answering a knowledge-base question")
            await message.answer(strings.KB_GENERIC_ERROR)
            return

    if result.status is AnswerStatus.NOTHING_RELEVANT:
        await message.answer(strings.KB_NOTHING_RELEVANT)
        return
    if result.status is AnswerStatus.NOT_IN_NOTES:
        await message.answer(strings.KB_NOT_IN_NOTES)
        return

    for chunk in render_chunks(_answer_text(result)):
        await answer_html(message, chunk)
    if result.photo is not None and result.photo_source is not None:
        await message.answer_photo(
            BufferedInputFile(result.photo, filename=f"page-{result.photo_source.id}.jpg"),
            caption=escape(strings.KB_PHOTO_CAPTION.format(title=result.photo_source.title)),
            parse_mode="HTML",
        )
