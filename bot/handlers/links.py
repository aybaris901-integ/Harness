"""Link summarizer (CLAUDE.md §7 Phase 2).

Any non-command message containing a URL — in any FSM state — lands here,
before the tutor router gets a chance. `/summarize <url>` does the same
explicitly. Messages starting with a bot command are never claimed by the
generic URL handler, whatever the router order: `/download <url>`,
`/find <url>`, a photo captioned `/note <url>` belong to their own handlers.

- Article links are handled inline: fetch + summarize takes a few seconds, so
  a "typing" indicator is enough.
- Video links get a "processing…" reply first and then run as a background
  task: subtitle download is quick, but a fallback to speech-to-text can take
  minutes, and the bot must keep answering everyone else meanwhile. The status
  message is edited as the job progresses and replaced by the summary at the end.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import BaseFilter, Command, CommandObject
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

import strings
from bot.formatting import answer_html, edit_html, render_chunks
from harness import Harness, HarnessError, LinkError, LinkSummary
from tools.urls import LinkKind, extract_urls

logger = logging.getLogger(__name__)

router = Router(name="links")

# asyncio only keeps weak references to tasks; hold them so a running video
# job is never garbage-collected mid-flight.
_background_tasks: set[asyncio.Task[Any]] = set()


class HasUrl(BaseFilter):
    """Matches non-command messages with a URL in the text; passes the URLs to
    the handler. Commands are excluded here rather than relying on router
    order — this is the catch-all, it must never shadow `/download <url>`."""

    async def __call__(self, message: Message) -> bool | dict[str, list[str]]:
        if (message.text or message.caption or "").lstrip().startswith("/"):
            return False
        urls = _urls_from_message(message)
        return {"urls": urls} if urls else False


def _urls_from_message(message: Message) -> list[str]:
    text = message.text or message.caption or ""
    urls = extract_urls(text)
    # Hyperlinked text ("click here") carries its URL only in the entity.
    for entity in (message.entities or []) + (message.caption_entities or []):
        if entity.type == "text_link" and entity.url and entity.url not in urls:
            urls.append(entity.url)
    return urls


def _summary_chunks(result: LinkSummary) -> list[str]:
    icon = "🎬" if result.kind is LinkKind.VIDEO else "📄"
    return render_chunks(result.summary, header=f"{icon} {result.title or result.url}")


async def _send_summary(message: Message, result: LinkSummary) -> None:
    for chunk in _summary_chunks(result):
        await answer_html(message, chunk)


async def _handle_article(message: Message, harness: Harness, url: str, user_message: str) -> None:
    assert message.from_user is not None
    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            result = await harness.summarize_link(
                telegram_id=message.from_user.id,
                chat_id=message.chat.id,
                url=url,
                user_message=user_message,
            )
        except (LinkError, HarnessError) as exc:
            await message.answer(str(exc))
            return
        except Exception:
            logger.exception("Unexpected failure summarizing article %s", url)
            await message.answer(strings.LINK_GENERIC_ERROR)
            return
    await _send_summary(message, result)


async def _handle_video(message: Message, harness: Harness, url: str, user_message: str) -> None:
    status = await message.answer(strings.VIDEO_PROCESSING)
    task = asyncio.create_task(
        _video_job(message, status, harness, url, user_message),
        name=f"video-summary:{message.chat.id}:{message.message_id}",
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _video_job(
    message: Message, status: Message, harness: Harness, url: str, user_message: str
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
        result = await harness.summarize_link(
            telegram_id=message.from_user.id,
            chat_id=message.chat.id,
            url=url,
            user_message=user_message,
            on_progress=on_progress,
        )
    except (LinkError, HarnessError) as exc:
        await _finish(bot, status, f"⚠️ {exc}")
        return
    except Exception:
        logger.exception("Unexpected failure summarizing video %s", url)
        await _finish(bot, status, f"⚠️ {strings.LINK_GENERIC_ERROR}")
        return

    chunks = _summary_chunks(result)
    # The first chunk replaces the status message; the rest follow as replies.
    await edit_html(bot, status, chunks[0])
    for chunk in chunks[1:]:
        await answer_html(message, chunk)


async def _finish(bot: Bot, status: Message, text: str) -> None:
    try:
        await bot.edit_message_text(text, chat_id=status.chat.id, message_id=status.message_id)
    except TelegramBadRequest:
        await bot.send_message(status.chat.id, text)


async def _dispatch(message: Message, harness: Harness, urls: list[str], user_message: str) -> None:
    if len(urls) > 1:
        await message.answer(strings.MULTIPLE_URLS_NOTICE)
    url = urls[0]
    if harness.link_kind(url) is LinkKind.VIDEO:
        await _handle_video(message, harness, url, user_message)
    else:
        await _handle_article(message, harness, url, user_message)


@router.message(Command("summarize"))
async def handle_summarize_command(
    message: Message, command: CommandObject, harness: Harness
) -> None:
    args = (command.args or "").strip()
    urls = extract_urls(args)
    if not urls:
        await message.answer(strings.SUMMARIZE_USAGE)
        return
    # Raw text is fine: tools.language.user_words strips "/summarize" (and the
    # URL) before the reply language is picked.
    await _dispatch(message, harness, urls, user_message=(message.text or "").strip())


@router.message(F.text | F.caption, HasUrl())
async def handle_link(message: Message, harness: Harness, urls: list[str]) -> None:
    user_message = (message.text or message.caption or "").strip()
    await _dispatch(message, harness, urls, user_message=user_message)
