"""Link summarizer (CLAUDE.md §7 Phase 2).

Any message containing a URL — in any FSM state — lands here, before the tutor
router gets a chance. `/summarize <url>` does the same explicitly.

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
from bot.utils import split_message
from harness import Harness, HarnessError, LinkError, LinkSummary
from tools.urls import LinkKind, extract_urls

logger = logging.getLogger(__name__)

router = Router(name="links")

# asyncio only keeps weak references to tasks; hold them so a running video
# job is never garbage-collected mid-flight.
_background_tasks: set[asyncio.Task[Any]] = set()


class HasUrl(BaseFilter):
    """Matches messages with a URL in the text; passes the URLs to the handler."""

    async def __call__(self, message: Message) -> bool | dict[str, list[str]]:
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


def _header(result: LinkSummary) -> str:
    icon = "🎬" if result.kind is LinkKind.VIDEO else "📄"
    title = result.title or result.url
    return f"{icon} {title}\n\n"


async def _send_summary(message: Message, result: LinkSummary) -> None:
    chunks = split_message(_header(result) + result.summary)
    for chunk in chunks:
        await message.answer(chunk)


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

    chunks = split_message(_header(result) + result.summary)
    # The first chunk replaces the status message; the rest follow as replies.
    await _finish(bot, status, chunks[0])
    for chunk in chunks[1:]:
        await message.answer(chunk)


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
    # Pass only the arguments: "/summarize" itself must not count as an English
    # word when the reply language is picked from the user's message.
    await _dispatch(message, harness, urls, user_message=args)


@router.message(F.text | F.caption, HasUrl())
async def handle_link(message: Message, harness: Harness, urls: list[str]) -> None:
    user_message = (message.text or message.caption or "").strip()
    await _dispatch(message, harness, urls, user_message=user_message)
