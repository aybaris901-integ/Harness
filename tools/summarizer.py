"""Summarization step shared by the article and video pipelines.

Builds the §9 prompts from `harness.prompts` and calls the router. Everything
else (fetching, extraction, transcription) happens upstream in the other
`tools.*` modules; this one only turns text into a summary.
"""

from __future__ import annotations

import logging

from harness.prompts import ARTICLE_SUMMARIZER_SYSTEM_PROMPT, VIDEO_SUMMARIZER_SYSTEM_PROMPT
from llm_router import LLMRouter
from tools.article import Article
from tools.downloader import VideoInfo
from tools.language import reply_language
from tools.transcript import format_timestamp

logger = logging.getLogger(__name__)

# Hard cap on source text handed to the model. Gemini would take far more, but
# the free tiers further down the chain (Groq: ~12k tokens/min) would not.
MAX_INPUT_CHARS = 120_000
SUMMARY_MAX_TOKENS = 1024
SUMMARY_TEMPERATURE = 0.3

# LANGUAGE_POLICY alone is not enough here: with an English source and a bare
# URL as the user turn, models drift to English. So the reply language is
# decided in code from the user's own words (CLAUDE.md §8: explicit routing
# over "let the model decide"; see tools/language.py) and stated once more at
# the very end of the prompt, where it carries the most weight.
_REPLY_LANGUAGE_LINE = "\n\n(Reply language: {language}.)"
_TRUNCATION_NOTE = "\n\n[... text truncated here, the source continues ...]"


def user_turn(user_message: str) -> str:
    return user_message + _REPLY_LANGUAGE_LINE.format(language=reply_language(user_message))


def truncate(text: str, limit: int = MAX_INPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    logger.info("truncating source text from %d to %d chars", len(text), limit)
    return text[:limit].rstrip() + _TRUNCATION_NOTE


async def summarize_article(router: LLMRouter, article: Article, *, user_message: str) -> str:
    header_lines = []
    if article.title:
        header_lines.append(f"Title: {article.title}")
    if article.author:
        header_lines.append(f"Author: {article.author}")
    if article.date:
        header_lines.append(f"Date: {article.date}")
    if article.site:
        header_lines.append(f"Source: {article.site}")
    body = "\n".join(header_lines) + ("\n\n" if header_lines else "") + article.text

    system = ARTICLE_SUMMARIZER_SYSTEM_PROMPT.format(article_text=truncate(body))
    return await router.complete(
        user_turn(user_message),
        system,
        temperature=SUMMARY_TEMPERATURE,
        max_tokens=SUMMARY_MAX_TOKENS,
    )


async def summarize_video(
    router: LLMRouter,
    info: VideoInfo,
    transcript_text: str,
    *,
    user_message: str,
    transcript_source: str,
) -> str:
    header_lines = [f"Transcript source: {transcript_source}"]
    if info.uploader:
        header_lines.append(f"Channel: {info.uploader}")
    if info.duration:
        header_lines.append(f"Duration: {format_timestamp(info.duration)}")
    if info.chapters:
        chapters = "\n".join(
            f"[{format_timestamp(start)}] {title}" for start, title in info.chapters
        )
        header_lines.append(f"Chapters:\n{chapters}")
    body = "\n".join(header_lines) + "\n\n" + transcript_text

    system = VIDEO_SUMMARIZER_SYSTEM_PROMPT.format(
        video_title=info.title.replace('"', "'"),
        transcript_text=truncate(body),
    )
    return await router.complete(
        user_turn(user_message),
        system,
        temperature=SUMMARY_TEMPERATURE,
        max_tokens=SUMMARY_MAX_TOKENS,
    )
