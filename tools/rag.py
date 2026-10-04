"""RAG building blocks (CLAUDE.md §7 Phase 5a): chunking and the grounded answer.

Pure text in / text out plus one router call — no storage, no embeddings, no
bot. `harness/knowledge.py` wires these together with the vector store.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from harness.prompts import RAG_ANSWER_SYSTEM_PROMPT
from llm_router import LLMRouter
from tools.language import reply_language

logger = logging.getLogger(__name__)

CHUNK_MAX_CHARS = 800
# Answer text is short, but reasoning models in the chain (Groq gpt-oss) spend
# hidden reasoning tokens out of the same budget before emitting any content.
# Measured on this exact prompt + schema (Phase 5a): default effort 124-221
# reasoning tokens, and at max_tokens=128 Groq returns HTTP 400 "Failed to
# validate JSON" (truncated structured output — the JSON-mode form of the
# empty-content bug); with GROQ_REASONING_EFFORT=low, 23-49 tokens. 2048 keeps
# ~40x headroom over that even if the effort setting is removed.
ANSWER_MAX_TOKENS = 2048
ANSWER_TEMPERATURE = 0.2

_REPLY_LANGUAGE_LINE = "\n\n(Reply language: {language}.)"

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
            "description": "False if the excerpts do not contain the answer.",
        },
        "answer": {"type": "string", "description": "The answer, with [n] citations."},
        "sources": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Numbers of the excerpts the answer is based on.",
        },
    },
    "required": ["answerable", "answer", "sources"],
    "additionalProperties": False,
}


class AnswerFormatError(RuntimeError):
    """The model's structured reply could not be parsed."""


@dataclass(slots=True)
class Excerpt:
    title: str
    text: str


@dataclass(slots=True)
class GroundedAnswer:
    answerable: bool
    answer: str
    # 1-based excerpt numbers the model cited, validated against the excerpt list.
    sources: list[int] = field(default_factory=list)


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split on line boundaries into chunks of at most ~`max_chars`.

    Notebook pages and screenshots are line-structured (one fact per line, a
    formula per line), so lines are the natural unit. The last line of a chunk
    is repeated at the start of the next one, so a fact that straddles the
    boundary is retrievable from either side. A single over-long line is
    hard-split.
    """
    lines = [line.rstrip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line.strip()]
    pieces: list[str] = []
    for line in lines:
        while len(line) > max_chars:
            cut = line.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(line[:cut].rstrip())
            line = line[cut:].lstrip()
        pieces.append(line)

    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len("\n".join([*current, piece])) > max_chars:
            chunks.append("\n".join(current))
            overlap = current[-1]
            current = [overlap] if len(overlap) + len(piece) + 1 <= max_chars else []
        current.append(piece)
    if current:
        chunks.append("\n".join(current))
    return chunks


def embedding_input(title: str, chunk: str) -> str:
    """What actually gets embedded: the title gives a short chunk its topic."""
    return f"{title}\n\n{chunk}" if title else chunk


def build_context(excerpts: list[Excerpt]) -> str:
    return "\n\n".join(
        f"[{number}] ({excerpt.title})\n{excerpt.text}"
        for number, excerpt in enumerate(excerpts, start=1)
    )


async def answer_question(
    router: LLMRouter, question: str, excerpts: list[Excerpt], *, user_message: str
) -> GroundedAnswer:
    system = RAG_ANSWER_SYSTEM_PROMPT.format(context=build_context(excerpts))
    user_turn = question + _REPLY_LANGUAGE_LINE.format(language=reply_language(user_message))
    reply = await router.complete(
        user_turn,
        system,
        json_schema=_ANSWER_SCHEMA,
        temperature=ANSWER_TEMPERATURE,
        max_tokens=ANSWER_MAX_TOKENS,
    )
    try:
        data = json.loads(reply)
        answerable = bool(data["answerable"])
        answer = str(data.get("answer") or "").strip()
        raw_sources = data.get("sources") or []
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AnswerFormatError(f"model returned invalid JSON: {exc}") from exc

    sources: list[int] = []
    for value in raw_sources:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= len(excerpts) and number not in sources:
            sources.append(number)
    if not answer:
        answerable = False
    return GroundedAnswer(answerable=answerable, answer=answer, sources=sources)
