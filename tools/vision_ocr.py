"""Handwriting transcription via the vision chain (CLAUDE.md §7 Phase 5a).

    page photo -> llm_router vision chain (Gemini -> Groq -> OpenRouter, each a
                  vision-capable model) -> plain transcription

This module only makes the call. It has no idea whether the image is safe to
send: the PII gate (local Tesseract + document_fields.has_pii_signals) runs in
harness/knowledge.py *before* anything here is reached, and that is the only
caller. Tesseract as the last-resort fallback also lives there, since it is
local and needs no router.
"""

from __future__ import annotations

import re

from harness.prompts import PAGE_TRANSCRIPTION_PROMPT
from llm_router import LLMRouter
from providers import ImagePart

TRANSCRIPTION_MAX_TOKENS = 2048
TRANSCRIPTION_TEMPERATURE = 0.0

_FENCE_RE = re.compile(r"^```[\w-]*\n(.*?)\n```$", re.DOTALL)


def _mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"  # Telegram photos are always JPEG


async def transcribe_page(router: LLMRouter, image_bytes: bytes) -> str:
    """Return the page text (possibly empty). Raises AllProvidersFailedError."""
    text = await router.complete(
        PAGE_TRANSCRIPTION_PROMPT,
        images=[ImagePart(data=image_bytes, mime_type=_mime_type(image_bytes))],
        temperature=TRANSCRIPTION_TEMPERATURE,
        max_tokens=TRANSCRIPTION_MAX_TOKENS,
    )
    text = text.strip()
    # Some models wrap the transcription in a fence despite the instruction.
    if match := _FENCE_RE.match(text):
        text = match.group(1).strip()
    return text
