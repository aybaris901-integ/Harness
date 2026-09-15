"""Video pipeline, step 1: yt-dlp.

Two entry points, both blocking-in-a-thread wrappers around yt-dlp:

- `probe(url)`            -> `VideoInfo` (title, duration, which subtitles exist)
- `fetch_subtitles(...)`  -> transcript segments from existing subs, if any
- `fetch_audio(...)`      -> path to an audio file for speech-to-text

The subtitle path is preferred (CLAUDE.md §3): free, instant, and usually more
accurate than STT. Audio is only downloaded when there are no usable subs.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError

from tools.transcript import Segment, parse_vtt

logger = logging.getLogger(__name__)

# Languages we'd rather read subtitles in, in order. Any other language still
# works (the summarizer answers in Kazakh regardless) — this is just tie-breaking.
PREFERRED_SUB_LANGS = ("en", "kk", "ru")
# Subtitle tracks yt-dlp lists that are not speech.
_IGNORED_SUB_LANGS = {"live_chat", "rechat"}


class DownloadFailed(RuntimeError):
    """User-presentable failure of the yt-dlp step."""


class NotAVideo(DownloadFailed):
    """yt-dlp could not find media at the URL (e.g. a text-only tweet)."""


@dataclass(slots=True)
class SubtitleTrack:
    lang: str
    automatic: bool


@dataclass(slots=True)
class VideoInfo:
    url: str
    id: str
    title: str
    duration: float | None
    uploader: str | None
    is_live: bool
    extractor: str
    manual_subs: list[str] = field(default_factory=list)
    auto_subs: list[str] = field(default_factory=list)
    chapters: list[tuple[float, str]] = field(default_factory=list)
    description: str | None = None

    @property
    def duration_minutes(self) -> float | None:
        return self.duration / 60 if self.duration else None

    def best_subtitle(self) -> SubtitleTrack | None:
        """Manual subs beat auto-captions; the video's own language beats a translation."""
        original = self._original_lang()
        for langs, automatic in ((self.manual_subs, False), (self.auto_subs, True)):
            if not langs:
                continue
            for wanted in self._lang_priority(original, automatic):
                match = next((lang for lang in langs if _lang_matches(lang, wanted)), None)
                if match:
                    return SubtitleTrack(lang=match, automatic=automatic)
            return SubtitleTrack(lang=langs[0], automatic=automatic)
        return None

    def _original_lang(self) -> str | None:
        # YouTube marks the untranslated auto track as "<lang>-orig".
        for lang in self.auto_subs:
            if lang.endswith("-orig"):
                return lang[: -len("-orig")]
        return None

    @staticmethod
    def _lang_priority(original: str | None, automatic: bool) -> list[str]:
        priority: list[str] = []
        if original:
            if automatic:
                priority.append(f"{original}-orig")
            priority.append(original)
        priority.extend(PREFERRED_SUB_LANGS)
        return priority


def _lang_matches(available: str, wanted: str) -> bool:
    available = available.lower()
    wanted = wanted.lower()
    return available == wanted or available.startswith(wanted + "-")


def _base_opts(ffmpeg_path: str | None) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "logger": logging.getLogger("yt_dlp"),
        "socket_timeout": 30,
        "retries": 3,
    }
    if ffmpeg_path:
        opts["ffmpeg_location"] = ffmpeg_path
    return opts


def _translate_error(exc: DownloadError, url: str) -> DownloadFailed:
    message = str(exc)
    lowered = message.lower()
    if "unsupported url" in lowered or "no video" in lowered or "no media" in lowered:
        return NotAVideo(f"no video found at {url}")
    if "private video" in lowered or "sign in" in lowered or "login" in lowered:
        return DownloadFailed("the video is private or requires a login")
    if "video unavailable" in lowered or "removed" in lowered:
        return DownloadFailed("the video is unavailable")
    if "age" in lowered and "confirm" in lowered:
        return DownloadFailed("the video is age-restricted and cannot be fetched anonymously")
    return DownloadFailed(f"yt-dlp failed: {message.splitlines()[-1][:200]}")


def _probe_sync(url: str, ffmpeg_path: str | None) -> VideoInfo:
    opts = _base_opts(ffmpeg_path) | {"skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise _translate_error(exc, url) from exc
    if info is None:
        raise NotAVideo(f"no video found at {url}")

    if info.get("_type") == "playlist":
        entries = [entry for entry in info.get("entries") or [] if entry]
        if not entries:
            raise NotAVideo("that link is an empty playlist")
        if len(entries) > 1:
            raise DownloadFailed("that link is a playlist — send a link to a single video")
        info = entries[0]

    def lang_list(key: str) -> list[str]:
        tracks = info.get(key) or {}
        return [lang for lang in tracks if lang not in _IGNORED_SUB_LANGS]

    chapters = [
        (float(ch.get("start_time") or 0), str(ch.get("title") or ""))
        for ch in (info.get("chapters") or [])
        if ch.get("title")
    ]
    return VideoInfo(
        url=url,
        id=str(info.get("id") or ""),
        title=str(info.get("title") or "Untitled video"),
        duration=float(info["duration"]) if info.get("duration") else None,
        uploader=info.get("uploader") or info.get("channel"),
        is_live=bool(info.get("is_live")),
        extractor=str(info.get("extractor_key") or info.get("extractor") or "?"),
        manual_subs=lang_list("subtitles"),
        auto_subs=lang_list("automatic_captions"),
        chapters=chapters,
        description=info.get("description") or None,
    )


async def probe(url: str, *, ffmpeg_path: str | None = None) -> VideoInfo:
    info = await asyncio.to_thread(_probe_sync, url, ffmpeg_path)
    logger.info(
        "video %s (%s): %r, %s, subs=%s auto=%s",
        info.id,
        info.extractor,
        info.title,
        f"{info.duration_minutes:.1f} min" if info.duration_minutes else "unknown length",
        info.manual_subs[:5],
        len(info.auto_subs),
    )
    return info


def _fetch_subtitles_sync(
    url: str, track: SubtitleTrack, work_dir: Path, ffmpeg_path: str | None
) -> list[Segment]:
    work_dir.mkdir(parents=True, exist_ok=True)
    opts = _base_opts(ffmpeg_path) | {
        "skip_download": True,
        "writesubtitles": not track.automatic,
        "writeautomaticsub": track.automatic,
        "subtitleslangs": [track.lang],
        "subtitlesformat": "vtt/best",
        "outtmpl": str(work_dir / "subs.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except DownloadError as exc:
        raise _translate_error(exc, url) from exc

    files = sorted(work_dir.glob("subs.*"))
    if not files:
        raise DownloadFailed("subtitle download produced no file")
    content = files[0].read_text(encoding="utf-8", errors="replace")
    segments = parse_vtt(content)
    if not segments:
        raise DownloadFailed("subtitle file was empty")
    return segments


async def fetch_subtitles(
    url: str, track: SubtitleTrack, work_dir: Path, *, ffmpeg_path: str | None = None
) -> list[Segment]:
    segments = await asyncio.to_thread(_fetch_subtitles_sync, url, track, work_dir, ffmpeg_path)
    logger.info(
        "subtitles for %s: lang=%s auto=%s, %d cues",
        url,
        track.lang,
        track.automatic,
        len(segments),
    )
    return segments


def _fetch_audio_sync(
    url: str, work_dir: Path, ffmpeg_path: str | None, max_bytes: int | None
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    opts = _base_opts(ffmpeg_path) | {
        # Smallest audio-only stream first; STT does not benefit from bitrate.
        "format": "worstaudio[abr>=48]/worstaudio/bestaudio/best",
        "outtmpl": str(work_dir / "audio.%(ext)s"),
    }
    if max_bytes:
        opts["max_filesize"] = max_bytes
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except DownloadError as exc:
        raise _translate_error(exc, url) from exc

    files = [p for p in work_dir.glob("audio.*") if p.suffix != ".part"]
    if not files:
        raise DownloadFailed(
            "audio download produced no file (over the size limit, or no audio stream?)"
        )
    return files[0]


async def fetch_audio(
    url: str, work_dir: Path, *, ffmpeg_path: str | None = None, max_bytes: int | None = None
) -> Path:
    path = await asyncio.to_thread(_fetch_audio_sync, url, work_dir, ffmpeg_path, max_bytes)
    logger.info("audio for %s: %s (%.1f MB)", url, path.name, path.stat().st_size / 1e6)
    return path


def cleanup(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
