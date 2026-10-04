"""Quiz presentation mode per card (CLAUDE.md §7 Phase 5b). Pure, no I/O.

    multiple_choice   the answer is short and 3 valid wrong options exist
    show_answer       fallback: reveal + self-rating; `reason` says why:
                      answer_too_long | too_few_distractors | other

Wrong options are taken in order of preference: the card's own stored
options (generated with it), then answers of cards from the same source, then
the user's other cards. Every candidate passes `option_ok` against the answer
and against the options already picked, whatever its origin.

One function decides for both the quiz and the /cards "supports multiple
choice" count, so the two can never disagree.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

ANSWER_MAX_CHARS = 60  # longer answers don't fit a button
OPTIONS_NEEDED = 3

_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT = " .,;:!?\"'«»()[]"
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip(_EDGE_PUNCT).casefold()


def _script(text: str) -> str:
    if _CYRILLIC_RE.search(text):
        return "cyrillic"
    if _LATIN_RE.search(text):
        return "latin"
    return "none"


def option_ok(answer: str, option: str) -> bool:
    """Is `option` a plausible wrong answer for `answer`? Similar length and
    shape: same script, digits only if the answer has digits (and vice
    versa), not equal to the answer, fits a button."""
    answer, option = answer.strip(), option.strip()
    if not option or len(option) > ANSWER_MAX_CHARS:
        return False
    if normalize(option) == normalize(answer) or not normalize(option):
        return False
    a, o = len(answer), len(option)
    if o > max(3 * a, a + 15) or o < min(a / 3, a - 15):
        return False
    if any(c.isdigit() for c in answer) != any(c.isdigit() for c in option):
        return False
    answer_script = _script(answer)
    return answer_script == "none" or _script(option) == answer_script


def pick_options(
    answer: str, *candidate_groups: Iterable[str], limit: int = OPTIONS_NEEDED
) -> list[str]:
    """Up to `limit` valid, mutually distinct options, groups in order."""
    picked: list[str] = []
    seen = {normalize(answer)}
    for group in candidate_groups:
        for candidate in group:
            candidate = candidate.strip()
            key = normalize(candidate)
            if key in seen or not option_ok(answer, candidate):
                continue
            seen.add(key)
            picked.append(candidate)
            if len(picked) == limit:
                return picked
    return picked


@dataclass(slots=True)
class ModeDecision:
    mode: str  # "multiple_choice" | "show_answer"
    reason: str  # "-" | "answer_too_long" | "too_few_distractors" | "other"
    answer_len: int
    stored_options: int  # the card's own stored options that passed validation
    candidates: int  # all valid distractor candidates, from every group
    options: list[str] = field(default_factory=list)  # the 3 chosen (multiple_choice only)


def choose_mode(
    answer: str,
    stored_options: Iterable[str],
    same_source_answers: Iterable[str],
    other_answers: Iterable[str],
) -> ModeDecision:
    answer = answer.strip()
    stored = list(stored_options)
    same = list(same_source_answers)
    other = list(other_answers)
    if not answer:
        return ModeDecision("show_answer", "other", 0, 0, 0)
    if len(answer) > ANSWER_MAX_CHARS:
        return ModeDecision("show_answer", "answer_too_long", len(answer), 0, 0)
    valid_stored = pick_options(answer, stored)
    # Count every valid candidate (not just the first 3) for the log line.
    seen = {normalize(answer)}
    candidates = 0
    for candidate in [*stored, *same, *other]:
        key = normalize(candidate)
        if key not in seen and option_ok(answer, candidate):
            seen.add(key)
            candidates += 1
    options = pick_options(answer, stored, same, other)
    if len(options) < OPTIONS_NEEDED:
        return ModeDecision(
            "show_answer", "too_few_distractors", len(answer), len(valid_stored), candidates
        )
    return ModeDecision("multiple_choice", "-", len(answer), len(valid_stored), candidates, options)
