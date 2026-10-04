"""LLM-formatting step for Phase 4 (CLAUDE.md §7 Phase 4).

Two independent jobs, both "take extracted text/metadata and turn it into
user-facing content via the router" — same spirit as `tools/summarizer.py`,
just for Phase 4's two new pipelines instead of Phase 2's:

    describe_media    -> a downloaded file's metadata -> short description/timestamps
    format_screenshot -> OCR'd screenshot text -> title + tags for staging. The
                         OCR text itself is kept verbatim by the caller; the
                         model only labels it. An earlier version let it
                         "clean up" the text, and it rewrote content (e.g.
                         "[See-ee-oh]" -> "[Cee-ее-ой]", mixed scripts).

`format_screenshot` is fine to go through `llm_router.py` (unlike anything in
`tools/document_fields.py`): CLAUDE.md §5 explicitly carves screenshots out as
the non-sensitive case, distinct from the PII documents `harness/documents.py`
handles locally-only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from harness.prompts import MEDIA_DESCRIPTION_SYSTEM_PROMPT, SCREENSHOT_NOTE_SYSTEM_PROMPT
from llm_router import LLMRouter
from tools.downloader import VideoInfo
from tools.language import reply_language
from tools.summarizer import truncate
from tools.transcript import format_timestamp

logger = logging.getLogger(__name__)

DESCRIPTION_MAX_TOKENS = 512
DESCRIPTION_TEMPERATURE = 0.3
NOTE_MAX_TOKENS = 1024
NOTE_TEMPERATURE = 0.2

# Mirrors tools/summarizer.py's own reply-language line: a bare URL or a
# screenshot with no caption gives the model nothing to infer language from,
# so it's decided in code (CLAUDE.md §8) and stated once more at the end.
_REPLY_LANGUAGE_LINE = "\n\n(Reply language: {language}.)"

_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A short, specific title, 3-8 words."},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "0-5 short topic tags.",
        },
    },
    "required": ["title"],
}


class ScreenshotFormatError(RuntimeError):
    """The model's structured-output reply for a screenshot could not be parsed."""


@dataclass(slots=True)
class ScreenshotNote:
    title: str
    tags: list[str]
    raw_json: str  # stored verbatim in the staging table for Phase 5 to consume


async def describe_media(router: LLMRouter, info: VideoInfo, *, user_message: str) -> str:
    header_lines = [f"Title: {info.title}"]
    if info.uploader:
        header_lines.append(f"Uploader: {info.uploader}")
    if info.duration:
        header_lines.append(f"Duration: {format_timestamp(info.duration)}")
    if info.description:
        header_lines.append(f"Description: {info.description}")
    if info.chapters:
        chapters = "\n".join(
            f"[{format_timestamp(start)}] {title}" for start, title in info.chapters
        )
        header_lines.append(f"Chapters:\n{chapters}")
    metadata_text = "\n".join(header_lines)

    system = MEDIA_DESCRIPTION_SYSTEM_PROMPT.format(
        title=info.title.replace('"', "'"), metadata_text=truncate(metadata_text)
    )
    user_turn = user_message + _REPLY_LANGUAGE_LINE.format(language=reply_language(user_message))
    return await router.complete(
        user_turn, system, temperature=DESCRIPTION_TEMPERATURE, max_tokens=DESCRIPTION_MAX_TOKENS
    )


async def format_screenshot(
    router: LLMRouter, ocr_text: str, *, user_message: str
) -> ScreenshotNote:
    system = SCREENSHOT_NOTE_SYSTEM_PROMPT.format(ocr_text=truncate(ocr_text))
    user_turn = (
        "Give a title and tags for the screenshot text above."
        + _REPLY_LANGUAGE_LINE.format(language=reply_language(user_message))
    )
    reply = await router.complete(
        user_turn,
        system,
        json_schema=_NOTE_SCHEMA,
        temperature=NOTE_TEMPERATURE,
        max_tokens=NOTE_MAX_TOKENS,
    )
    try:
        data = json.loads(reply)
        title = str(data["title"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ScreenshotFormatError(f"model returned invalid JSON: {exc}") from exc
    if not title:
        raise ScreenshotFormatError("model returned an empty title")
    tags = [str(tag).strip() for tag in (data.get("tags") or []) if str(tag).strip()]
    return ScreenshotNote(title=title, tags=tags, raw_json=reply)
