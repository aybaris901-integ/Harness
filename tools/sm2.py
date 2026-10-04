"""SM-2 spaced repetition (CLAUDE.md §3, §7 Phase 5b), implemented directly.

Per card: repetitions (successful reviews in a row), interval (days), ease
factor (EF, starts at 2.5). After each review with grade q in 0..5:

    q >= 3 (recalled):  interval = 1, then 6, then round(previous * EF)
                        repetitions += 1
    q <  3 (lapse):     repetitions = 0, interval = 1 day, lapses += 1
    always:             EF += 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02), floor 1.3

EF is updated on lapses too (the common reading of SuperMemo's description,
used by e.g. the Wikipedia pseudo-code): a card that keeps failing gets
shorter intervals once it is relearned, which is the point. The interval
uses the EF from *before* this review, as in the original.

Pure functions, no I/O: storage persists the state, the harness decides when
a card is due.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

INITIAL_EASE = 2.5
MIN_EASE = 1.3
FIRST_INTERVAL_DAYS = 1
SECOND_INTERVAL_DAYS = 6
LAPSE_INTERVAL_DAYS = 1
PASSING_GRADE = 3


@dataclass(frozen=True, slots=True)
class CardState:
    ease: float = INITIAL_EASE
    interval_days: int = 0
    repetitions: int = 0
    lapses: int = 0


def next_ease(ease: float, grade: int) -> float:
    miss = 5 - grade
    return max(MIN_EASE, ease + 0.1 - miss * (0.08 + miss * 0.02))


def review(state: CardState, grade: int) -> CardState:
    """New state after a review graded 0..5."""
    if not 0 <= grade <= 5:
        raise ValueError(f"SM-2 grade must be 0..5, got {grade}")
    if grade >= PASSING_GRADE:
        if state.repetitions == 0:
            interval = FIRST_INTERVAL_DAYS
        elif state.repetitions == 1:
            interval = SECOND_INTERVAL_DAYS
        else:
            interval = max(1, round(state.interval_days * state.ease))
        new = replace(state, interval_days=interval, repetitions=state.repetitions + 1)
    else:
        new = replace(
            state,
            interval_days=LAPSE_INTERVAL_DAYS,
            repetitions=0,
            lapses=state.lapses + 1,
        )
    return replace(new, ease=next_ease(state.ease, grade))


def due_after(reviewed_at: datetime, state: CardState) -> datetime:
    return reviewed_at + timedelta(days=state.interval_days)
