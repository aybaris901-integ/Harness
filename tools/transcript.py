"""Timestamped transcript model shared by the subtitle and speech-to-text paths.

Both produce a list of `Segment`s; `format_transcript` turns them into the
`[mm:ss] text` form the video-summarizer prompt expects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TAG_RE = re.compile(r"<[^>]+>")
_TIMESTAMP_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_CUE_LINE_RE = re.compile(r"-->")


@dataclass(slots=True)
class Segment:
    start: float  # seconds
    end: float
    text: str


def _parse_timestamp(raw: str) -> float | None:
    match = _TIMESTAMP_RE.search(raw)
    if not match:
        return None
    hours, minutes, seconds, fraction = match.groups()
    return (
        int(hours or 0) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(fraction.ljust(3, "0")) / 1000
    )


def parse_vtt(content: str) -> list[Segment]:
    """Parse WebVTT (also tolerates SRT) into segments.

    YouTube auto-captions are "rolling": each cue repeats the previous cue's
    line and adds inline `<00:00:01.000><c>word</c>` timing. Tags are stripped
    and a line identical to the previously emitted one is dropped, which turns
    the rolling form back into plain consecutive lines.
    """
    segments: list[Segment] = []
    last_line = ""
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        # Skip the header, NOTE/STYLE/REGION blocks.
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")) and not any(
            _CUE_LINE_RE.search(line) for line in lines
        ):
            continue
        cue_index = next((i for i, line in enumerate(lines) if _CUE_LINE_RE.search(line)), None)
        if cue_index is None:
            continue
        start_raw, _, end_raw = lines[cue_index].partition("-->")
        start = _parse_timestamp(start_raw)
        end = _parse_timestamp(end_raw)
        if start is None:
            continue
        if end is None:
            end = start

        text_lines: list[str] = []
        for raw_line in lines[cue_index + 1 :]:
            line = _TAG_RE.sub("", raw_line).replace("&nbsp;", " ").strip()
            line = re.sub(r"\s+", " ", line)
            if not line or line == last_line:
                continue
            text_lines.append(line)
            last_line = line
        if text_lines:
            segments.append(Segment(start=start, end=end, text=" ".join(text_lines)))
    return segments


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_transcript(segments: list[Segment], *, window_seconds: float = 30.0) -> str:
    """Render segments as `[mm:ss] ...` lines, one per ~`window_seconds` of speech.

    Subtitle cues are tiny (a few words each); merging them into windows keeps
    the prompt short and still gives the model a usable timestamp per point.
    """
    if not segments:
        return ""
    lines: list[str] = []
    window_start = segments[0].start
    buffer: list[str] = []
    for segment in segments:
        if buffer and segment.start - window_start >= window_seconds:
            lines.append(f"[{format_timestamp(window_start)}] {' '.join(buffer)}")
            buffer = []
            window_start = segment.start
        buffer.append(segment.text)
    if buffer:
        lines.append(f"[{format_timestamp(window_start)}] {' '.join(buffer)}")
    return "\n".join(lines)


def plain_text(segments: list[Segment]) -> str:
    return " ".join(segment.text for segment in segments)
