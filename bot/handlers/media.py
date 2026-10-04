"""Media pipeline extras (CLAUDE.md §7 Phase 4).

- `/download <url> [mp3]` — download any yt-dlp-supported URL, not just the
  hosts `tools.urls` treats as "video" for `/summarize`, and send the file
  back with an LLM-generated description: an mp4 as an inline-playable video
  (falling back to a document if Telegram refuses it), mp3 as audio. Runs as a background task, same
  "processing…" -> edited-status pattern as the Phase 2 video summarizer,
  since a download can take a while.
- A photo captioned `/note` (anywhere in the caption) is a screenshot, not a
  PII document — OCR it locally (reusing `tools.ocr` as-is, same kaz+rus+eng
  config as Phase 3) then format the text into a note via the LLM. This
  router is included *before* `documents.router` in `bot/handlers/__init__.py`
  so it can claim `/note`-captioned photos first; any other photo still falls
  through to `documents.router` unchanged (default photo behavior is
  untouched — Phase 3 still owns every photo without that caption).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import BaseFilter, Command, CommandObject
from aiogram.types import BufferedInputFile, Message

import strings
from bot.formatting import (
    TELEGRAM_CAPTION_LIMIT,
    answer_html,
    edit_html,
    render_chunks,
    to_plain,
)
from harness.media import MediaDownload, MediaError, MediaPipeline
from storage.documents import DocumentRecord
from tools.urls import extract_urls

logger = logging.getLogger(__name__)

router = Router(name="media")

# asyncio only keeps weak references to tasks; hold them so a running download
# job is never garbage-collected mid-flight (same pattern as bot/handlers/links.py).
_background_tasks: set[asyncio.Task[Any]] = set()

# `/download` option words — not part of the user's own language.
_DOWNLOAD_OPTIONS = {"mp3"}


class HasNoteCaption(BaseFilter):
    """A photo captioned `/note` — a screenshot, not a PII document."""

    async def __call__(self, message: Message) -> bool:
        if not message.photo:
            return False
        caption = (message.caption or "").strip().lower()
        return caption == "/note" or caption.startswith("/note ")


def _format_document_saved(record: DocumentRecord) -> str:
    # Mirrors bot/handlers/documents.py's _format_saved — kept as its own
    # small copy rather than importing a private helper across handler
    # modules for what's meant to be a rare redirect path.
    doc_type = strings.DOCUMENT_TYPE_LABELS.get(record.document_type, record.document_type)
    header = strings.DOCUMENT_SAVED.format(type=doc_type, count=len(record.fields))
    lines = [header]
    lines.extend(f"• {name}: {value}" for name, value in record.fields.items())
    return "\n".join(lines)


async def _finish(bot: Bot, status: Message, text: str) -> None:
    try:
        await bot.edit_message_text(text, chat_id=status.chat.id, message_id=status.message_id)
    except TelegramBadRequest:
        await bot.send_message(status.chat.id, text)


# -- /download ------------------------------------------------------------------


async def _download_job(
    message: Message,
    status: Message,
    media: MediaPipeline,
    url: str,
    convert_to_mp3: bool,
    user_message: str,
) -> None:
    assert message.from_user is not None and message.bot is not None
    bot: Bot = message.bot

    async def on_progress(text: str) -> None:
        try:
            await bot.edit_message_text(
                f"⏳ {text}", chat_id=status.chat.id, message_id=status.message_id
            )
        except TelegramBadRequest:
            pass  # "message is not modified" or the user deleted it — cosmetic

    try:
        result = await media.download(
            url,
            convert_to_mp3=convert_to_mp3,
            user_message=user_message,
            on_progress=on_progress,
        )
    except MediaError as exc:
        await _finish(bot, status, f"⚠️ {exc}")
        return
    except Exception:
        logger.exception("Unexpected failure downloading %s", url)
        await _finish(bot, status, f"⚠️ {strings.MEDIA_GENERIC_ERROR}")
        return

    await _finish(bot, status, f"✅ {result.title}")
    await _send_media(message, result)


async def _send_media(message: Message, result: MediaDownload) -> None:
    # The description's first chunk is the caption; anything past Telegram's
    # caption limit follows as plain messages instead of being cut off.
    chunks = render_chunks(result.description, limit=TELEGRAM_CAPTION_LIMIT)
    caption = chunks[0]
    file = BufferedInputFile(result.data, filename=result.filename)

    if result.is_audio:
        try:
            await message.answer_audio(file, title=result.title, caption=caption, parse_mode="HTML")
        except TelegramBadRequest:
            await message.answer_audio(file, title=result.title, caption=to_plain(caption))
    else:
        sent = False
        if result.filename.lower().endswith(".mp4"):
            try:
                await message.answer_video(
                    file,
                    caption=caption,
                    parse_mode="HTML",
                    supports_streaming=True,
                    width=result.width,
                    height=result.height,
                    duration=int(result.duration) if result.duration else None,
                )
                sent = True
            except TelegramBadRequest as exc:
                logger.warning(
                    "send_video rejected %s (%s), sending as a document", result.filename, exc
                )
        if not sent:
            # Plain caption here: this is already the fallback path, so don't
            # risk a second rejection over caption markup.
            await message.answer_document(file, caption=to_plain(caption))

    for chunk in chunks[1:]:
        await answer_html(message, chunk)


@router.message(Command("download"))
async def handle_download(message: Message, command: CommandObject, media: MediaPipeline) -> None:
    args = (command.args or "").strip()
    urls = extract_urls(args)
    if not urls:
        await message.answer(strings.DOWNLOAD_USAGE)
        return
    words = (message.text or "").split()
    convert_to_mp3 = any(word.lower() == "mp3" for word in words[1:])
    # The command word and URL are stripped by tools.language.user_words;
    # option words like "mp3" are /download's own syntax, so they go here.
    user_message = " ".join(w for w in words if w.lower() not in _DOWNLOAD_OPTIONS)

    status = await message.answer(strings.MEDIA_PROCESSING)
    task = asyncio.create_task(
        _download_job(message, status, media, urls[0], convert_to_mp3, user_message),
        name=f"media-download:{message.chat.id}:{message.message_id}",
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# -- screenshot note --------------------------------------------------------------


@router.message(F.photo, HasNoteCaption())
async def handle_note_photo(message: Message, media: MediaPipeline) -> None:
    assert message.from_user is not None and message.bot is not None
    status = await message.answer(strings.NOTE_PROCESSING)

    # Same convention as bot/handlers/documents.py: Telegram sends the same
    # photo at several resolutions, smallest first — take the largest.
    photo = message.photo[-1]
    buffer = await message.bot.download(photo)
    image_bytes = buffer.read()
    # Raw caption: tools.language.user_words strips the "/note" command word
    # before the reply language is picked.
    user_message = (message.caption or "").strip()

    try:
        result = await media.capture_screenshot(
            telegram_id=message.from_user.id, image_bytes=image_bytes, user_message=user_message
        )
    except MediaError as exc:
        await status.edit_text(str(exc))
        return
    except Exception:
        logger.exception("Unexpected failure formatting a screenshot note")
        await status.edit_text(strings.MEDIA_GENERIC_ERROR)
        return

    if result.redirected_to_documents:
        assert result.document is not None
        await status.edit_text(strings.NOTE_PII_REDIRECTED)
        await message.answer(_format_document_saved(result.document))
        return

    assert result.note is not None
    note = result.note
    tags = " ".join("#" + "_".join(tag.split()) for tag in note.tags)
    # verbatim=True: the note body is raw OCR text, not model Markdown — it is
    # only escaped, never reinterpreted (a "*" in code stays a "*").
    chunks = render_chunks(
        f"{tags}\n\n{note.content_md}" if tags else note.content_md,
        header=strings.NOTE_SAVED.format(title=note.title),
        verbatim=True,
    )
    await edit_html(message.bot, status, chunks[0])
    for chunk in chunks[1:]:
        await answer_html(message, chunk)


@router.message(Command("note"))
async def handle_note_usage(message: Message) -> None:
    # `/note` with no photo attached (Telegram sends a caption-only command as
    # plain text, which HasNoteCaption above never sees since there's no photo).
    await message.answer(strings.NOTE_USAGE)
