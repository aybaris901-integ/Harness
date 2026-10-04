"""Flashcard extraction (CLAUDE.md §9 "Flashcard extraction", §7 Phase 5b).

    source text -> segments (~6000 chars, line boundaries)
                -> one structured-output call per segment (§9 prompt + JSON schema)
                -> validated, de-duplicated (front, back) drafts

Pure text in / drafts out. The PII gate (document_fields.has_pii_signals) is
NOT here: harness/flashcards.py runs it on every segment before calling
`generate_cards`, so this module never decides what may leave the server.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from harness.prompts import FLASHCARD_SYSTEM_PROMPT
from llm_router import LLMRouter
from tools.language import reply_language
from tools.quiz_mode import ANSWER_MAX_CHARS, OPTIONS_NEEDED, pick_options
from tools.rag import chunk_text

logger = logging.getLogger(__name__)

SEGMENT_MAX_CHARS = 6000
MAX_CARDS_PER_SEGMENT = 15
FRONT_MAX_CHARS = 300
BACK_MAX_CHARS = 500
EXPLANATION_MAX_CHARS = 500
# Reasoning models in the chain (Groq gpt-oss) spend hidden reasoning tokens
# out of the same budget before any JSON appears. Measured with the full
# schema (answer + explanation + 3 wrong options, Kazakh, 8-13 cards): at
# reasoning_effort=low, 34-50 reasoning / <=989 completion tokens; at default
# effort, 1460-1758 reasoning / up to 2601 completion tokens, and at 2048 Groq
# failed with json_validate_failed (truncated). 8192 keeps ~3x headroom at the
# default effort. A reply still cut at max_tokens is rejected by the provider
# (finish_reason=length), never stored as a silently shorter deck.
FLASHCARD_MAX_TOKENS = 8192
FLASHCARD_TEMPERATURE = 0.3

_REPLY_LANGUAGE_LINE = "\n\n(Reply language: {language}.)"
_WHITESPACE_RE = re.compile(r"\s+")

_CARDS_SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "front": {"type": "string", "description": "Question or fill-in-the-blank."},
                    "back": {
                        "type": "string",
                        "description": "Short answer only, at most 60 characters.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Optional extra context, or an empty string.",
                    },
                    "wrong_options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exactly 3 plausible wrong answers, like 'back' in shape.",
                    },
                },
                # Strict structured output (Groq/OpenAI) needs every property
                # listed; "explanation" may be "" and options are validated in code.
                "required": ["front", "back", "explanation", "wrong_options"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cards"],
    "additionalProperties": False,
}


class FlashcardFormatError(RuntimeError):
    """The model's structured reply could not be parsed."""


class FlashcardLanguageError(FlashcardFormatError):
    """The cards kept coming back in the wrong language (e.g. English/Russian
    when Kazakh was required) — never stored."""


# Letters Kazakh has and Russian/English don't.
_KAZAKH_LETTER_RE = re.compile(r"[әғқңөұүһіӘҒҚҢӨҰҮҺІ]")
_LANGUAGE_ATTEMPTS = 2
_LANGUAGE_INSTRUCTIONS = {
    "Kazakh": (
        'Write every "front", "back", "explanation" and "wrong_options" item in Kazakh '
        "(қазақ тілінде), even though the source text may be in Russian or English. "
        "Never write a card in Russian."
    ),
    "English": 'Write every "front", "back", "explanation" and "wrong_options" item in English.',
}


@dataclass(frozen=True, slots=True)
class CardDraft:
    front: str
    back: str
    explanation: str = ""
    # Generated wrong options that passed tools.quiz_mode.option_ok — never
    # equal to the answer, no duplicates, similar length and shape.
    options: tuple[str, ...] = ()


@dataclass(slots=True)
class ParseStats:
    """What validation dropped, for the generation log line."""

    long_answers: int = 0  # "back" over 60 chars despite the rule
    options_offered: int = 0
    options_rejected: int = 0


def segments(text: str) -> list[str]:
    return chunk_text(text, max_chars=SEGMENT_MAX_CHARS)


def normalize_front(front: str) -> str:
    """Key used to de-duplicate cards (case/whitespace-insensitive)."""
    return _WHITESPACE_RE.sub(" ", front).strip().casefold()


def parse_cards(reply: str, stats: ParseStats | None = None) -> list[CardDraft]:
    try:
        data = json.loads(reply)
        raw_cards = data["cards"]
        if not isinstance(raw_cards, list):
            raise TypeError("'cards' is not a list")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FlashcardFormatError(f"model returned invalid JSON: {exc}") from exc

    drafts: list[CardDraft] = []
    seen: set[str] = set()
    for item in raw_cards:
        if not isinstance(item, dict):
            continue
        front = _WHITESPACE_RE.sub(" ", str(item.get("front") or "")).strip()
        back = str(item.get("back") or "").strip()
        explanation = str(item.get("explanation") or "").strip()
        if not front or not back or front.casefold() == back.casefold():
            continue
        key = normalize_front(front)
        if key in seen:
            continue
        seen.add(key)
        raw_options = item.get("wrong_options")
        offered = [str(o) for o in raw_options] if isinstance(raw_options, list) else []
        if len(back) > ANSWER_MAX_CHARS:
            # The model broke the short-answer rule. Keep the card (it is still
            # a valid show-answer card) but no options: they would be judged
            # against an answer that can't be a button anyway.
            options: list[str] = []
            if stats is not None:
                stats.long_answers += 1
        else:
            # Spares beyond 3: the harness PII gate may still drop some.
            options = pick_options(back, offered, limit=2 * OPTIONS_NEEDED)
        if stats is not None:
            stats.options_offered += len(offered)
            stats.options_rejected += len(offered) - len(options)
        drafts.append(
            CardDraft(
                front=front[:FRONT_MAX_CHARS],
                back=back[:BACK_MAX_CHARS],
                explanation=explanation[:EXPLANATION_MAX_CHARS],
                options=tuple(options),
            )
        )
        if len(drafts) >= MAX_CARDS_PER_SEGMENT:
            break
    return drafts


def cards_in_language(drafts: list[CardDraft], language: str) -> bool:
    """Code-side check of LANGUAGE_POLICY on the generated cards (§8).

    Kazakh: the cards must contain Kazakh-specific letters somewhere — this
    rejects both English and Russian output (Russian has none of them), while
    tolerating individual terms like "митохондрия" or "ATP". English: mostly
    Latin letters. Measured over all cards together, not per card, so one
    formula-only card can't fail a good deck.
    """
    text = " ".join(f"{d.front} {d.back} {d.explanation}" for d in drafts)
    if language == "Kazakh":
        return bool(_KAZAKH_LETTER_RE.search(text))
    letters = [ch for ch in text if ch.isalpha()]
    latin = sum(1 for ch in letters if ch.isascii())
    return not letters or latin / len(letters) >= 0.6


async def generate_cards(router: LLMRouter, text: str, *, user_message: str) -> list[CardDraft]:
    """Cards for ONE segment of already-PII-checked text, verified to be in the
    reply language; one corrective retry, then FlashcardLanguageError."""
    system = FLASHCARD_SYSTEM_PROMPT.replace("{source_text}", text)
    language = reply_language(user_message)
    # Explicit and native-script: the §9 prompt is English, and a bare
    # "(Reply language: Kazakh.)" line was ignored by Groq gpt-oss in testing
    # (15/15 cards came back in English).
    instruction = _LANGUAGE_INSTRUCTIONS[language]
    user_turn = (
        "Make the flashcards now. " + instruction + _REPLY_LANGUAGE_LINE.format(language=language)
    )
    for attempt in range(1, _LANGUAGE_ATTEMPTS + 1):
        reply = await router.complete(
            user_turn,
            system,
            json_schema=_CARDS_SCHEMA,
            temperature=FLASHCARD_TEMPERATURE,
            max_tokens=FLASHCARD_MAX_TOKENS,
        )
        stats = ParseStats()
        drafts = parse_cards(reply, stats)
        if not drafts or cards_in_language(drafts, language):
            logger.info(
                "flashcards parsed: %d card(s), %d with 3 own options, %d long answer(s); "
                "options %d offered / %d rejected by validation",
                len(drafts),
                sum(len(d.options) == 3 for d in drafts),
                stats.long_answers,
                stats.options_offered,
                stats.options_rejected,
            )
            return drafts
        logger.warning(
            "flashcards came back in the wrong language (wanted %s), attempt %d/%d",
            language,
            attempt,
            _LANGUAGE_ATTEMPTS,
        )
        user_turn = (
            f"Your previous cards were NOT in {language}. Rewrite ALL of them. "
            + instruction
            + _REPLY_LANGUAGE_LINE.format(language=language)
        )
    raise FlashcardLanguageError(f"cards not in {language} after {_LANGUAGE_ATTEMPTS} attempts")
