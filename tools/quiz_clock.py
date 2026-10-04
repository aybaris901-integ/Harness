"""Daily-quiz clock rules (CLAUDE.md §7 Phase 5b). Pure functions, no I/O.

Local time is a fixed UTC offset per user, not a tz database name: the
default (Almaty) has been a single UTC+5 zone with no DST since 2024, and a
fixed offset needs no tzdata on the host (Termux-friendly).

The whole "is today's quiz due?" decision is `daily_due(...)` over values
stored in the DB — no in-memory job state, so restarts change nothing.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_OFFSET_RE = re.compile(r"^(?:utc|gmt)?([+-])(\d{1,2})(?::?([0-5]\d))?$", re.IGNORECASE)
MAX_OFFSET_MINUTES = 14 * 60


def parse_quiz_time(value: str) -> str | None:
    """Parse "20:00" / "7:05" into zero-padded "HH:MM"; None if not a valid time."""
    match = _TIME_RE.match(value.strip())
    return f"{int(match.group(1)):02d}:{match.group(2)}" if match else None


def parse_utc_offset(value: str) -> int | None:
    """Parse "+5" / "UTC+5" / "+05:30" / "-3" into minutes east of UTC; None if invalid."""
    match = _OFFSET_RE.match(value.strip())
    if not match:
        return None
    minutes = int(match.group(2)) * 60 + int(match.group(3) or 0)
    if minutes > MAX_OFFSET_MINUTES:
        return None
    return minutes if match.group(1) == "+" else -minutes


def format_utc_offset(minutes: int) -> str:
    sign = "+" if minutes >= 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours}" + (f":{mins:02d}" if mins else "")


def to_local(now_utc: datetime, utc_offset_minutes: int) -> datetime:
    return now_utc + timedelta(minutes=utc_offset_minutes)


def daily_due(
    *,
    now_utc: datetime,
    quiz_time: str,
    utc_offset_minutes: int,
    enabled: bool,
    last_daily_date: str | None,
) -> bool:
    """True once per local day, at or after `quiz_time`.

    Catch-up is deliberately gentle: if the bot was down at quiz time, the
    quiz still goes out when it comes back *the same local day*; missed
    previous days are not replayed (one capped session, never a backlog of
    sessions), and coming back after midnight but before quiz time waits for
    quiz time instead of firing at 02:00.
    """
    if not enabled:
        return False
    local = to_local(now_utc, utc_offset_minutes)
    return local.strftime("%H:%M") >= quiz_time and last_daily_date != local.date().isoformat()
