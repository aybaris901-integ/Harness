"""Video pipeline, step 1: yt-dlp.

Blocking-in-a-thread wrappers around yt-dlp:

- `probe(url)`            -> `VideoInfo` (title, duration, which subtitles exist)
- `fetch_subtitles(...)`  -> transcript segments from existing subs, if any
- `fetch_audio(...)`      -> path to a small audio file for speech-to-text (Phase 2)
- `fetch_media(...)`      -> a Telegram-playable mp4 (largest resolution under
                              the size cap) or an mp3, to hand back to the user (Phase 4)

The subtitle path is preferred (CLAUDE.md §3): free, instant, and usually more
accurate than STT. Audio is only downloaded when there are no usable subs.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError

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


# Telegram plays a video inline only as mp4 with H.264 video + AAC audio; a
# VP9/AV1 webm (what "bestvideo+bestaudio" picks on YouTube) arrives as a file
# to download instead. So: H.264 + m4a merged into mp4 first, then a single-
# file mp4, and only then whatever is best (still remuxed into an mp4
# container via merge_output_format — playable on most clients, and the bot
# layer falls back to sending a document if Telegram rejects it as a video).
# Bare "best" alone no longer matches anything on YouTube, which serves
# separate DASH video/audio streams only.
_VIDEO_FORMAT = "bv*[vcodec^=avc1]{h}+ba[ext=m4a]/b[ext=mp4]{h}/bv*{h}+ba/b{h}"
# Tried in order until the estimated size fits the cap — a lower resolution
# beats failing with "file too large".
_HEIGHT_LADDER: tuple[int | None, ...] = (None, 1080, 720, 480, 360, 240, 144)


@dataclass(slots=True)
class FetchedMedia:
    path: Path
    width: int | None = None
    height: int | None = None
    duration: float | None = None


def video_format_selector(max_height: int | None) -> str:
    return _VIDEO_FORMAT.format(h=f"[height<={max_height}]" if max_height else "")


def estimated_size(selected: dict[str, Any]) -> int | None:
    """Bytes the selected format(s) will take, or None if any part is unknown."""
    total = 0
    for part in selected.get("requested_formats") or [selected]:
        size = part.get("filesize") or part.get("filesize_approx")
        if not size:
            return None
        total += int(size)
    return total


def select_video_format(
    raw_info: dict[str, Any], opts: dict[str, Any], max_bytes: int | None
) -> tuple[str, dict[str, Any]]:
    """Walk `_HEIGHT_LADDER` and return the first selector (plus yt-dlp's
    resolved selection) whose estimated size fits `max_bytes`. Pure format
    selection on an already-extracted info dict — no network."""
    last: tuple[str, dict[str, Any]] | None = None
    for max_height in _HEIGHT_LADDER:
        selector = video_format_selector(max_height)
        try:
            with yt_dlp.YoutubeDL(opts | {"format": selector}) as ydl:
                selected = ydl.process_ie_result(copy.deepcopy(raw_info), download=False)
        except (DownloadError, ExtractorError):
            continue  # nothing at this height (or at all) — try the next rung
        if selected is None:
            continue
        last = (selector, selected)
        size = estimated_size(selected)
        # Unknown size: can't do better than trying it; max_filesize and the
        # harness' post-download check still guard the cap.
        if not max_bytes or size is None or size <= max_bytes:
            return selector, selected
        logger.info(
            "format %s ~%.1f MB exceeds the %.1f MB cap, trying a lower resolution",
            selected.get("format_id"),
            size / 1e6,
            max_bytes / 1e6,
        )
    if last is None:
        raise NotAVideo("no downloadable video format found")
    raise DownloadFailed(
        f"even the lowest resolution is larger than {max_bytes / 1e6:.0f} MB"
        if max_bytes
        else "no downloadable video format found"
    )


def _fetch_media_sync(
    url: str,
    work_dir: Path,
    ffmpeg_path: str | None,
    convert_to_mp3: bool,
    max_bytes: int | None,
) -> FetchedMedia:
    work_dir.mkdir(parents=True, exist_ok=True)
    opts = _base_opts(ffmpeg_path) | {"outtmpl": str(work_dir / "media.%(ext)s")}
    if max_bytes:
        # Best-effort pre-check per stream; select_video_format below is what
        # actually keeps the merged total under the cap.
        opts["max_filesize"] = max_bytes

    selected: dict[str, Any] = {}
    try:
        if convert_to_mp3:
            # An audio-only stream exists standalone, so "bestaudio" matches
            # directly. Requires ffmpeg; a missing one surfaces as a
            # DownloadError through _translate_error's generic case.
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
            ]
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        else:
            # Extract once, pick a format that fits offline, download that one.
            with yt_dlp.YoutubeDL(opts) as ydl:
                raw_info = ydl.extract_info(url, download=False, process=False)
            if raw_info is None:
                raise NotAVideo(f"no video found at {url}")
            opts["merge_output_format"] = "mp4"
            selector, selected = select_video_format(raw_info, opts, max_bytes)
            logger.info(
                "media for %s: format %s (%sp, ~%s MB)",
                url,
                selected.get("format_id"),
                selected.get("height") or "?",
                f"{size / 1e6:.1f}" if (size := estimated_size(selected)) else "?",
            )
            with yt_dlp.YoutubeDL(opts | {"format": selector}) as ydl:
                ydl.process_ie_result(raw_info, download=True)
    except DownloadError as exc:
        raise _translate_error(exc, url) from exc
    except ExtractorError as exc:
        raise DownloadFailed(f"yt-dlp failed: {str(exc).splitlines()[-1][:200]}") from exc

    files = [p for p in work_dir.glob("media.*") if p.suffix not in {".part", ".ytdl"}]
    if not files:
        raise DownloadFailed("download produced no file (over the size limit, or no media stream?)")
    # Prefer the merged mp4 if a stray intermediate stream file is left over.
    files.sort(key=lambda p: p.suffix != ".mp4")
    duration = selected.get("duration")
    return FetchedMedia(
        path=files[0],
        width=selected.get("width"),
        height=selected.get("height"),
        duration=float(duration) if duration else None,
    )


async def fetch_media(
    url: str,
    work_dir: Path,
    *,
    ffmpeg_path: str | None = None,
    convert_to_mp3: bool = False,
    max_bytes: int | None = None,
) -> FetchedMedia:
    """Download any yt-dlp-supported URL (CLAUDE.md §7 Phase 4) — unlike
    `fetch_audio`, this is for "save the file", not speech-to-text input: video
    comes back as a Telegram-playable mp4 at the highest resolution that fits
    `max_bytes`, or as mp3 when `convert_to_mp3` is set."""
    media = await asyncio.to_thread(
        _fetch_media_sync, url, work_dir, ffmpeg_path, convert_to_mp3, max_bytes
    )
    logger.info("media for %s: %s (%.1f MB)", url, media.path.name, media.path.stat().st_size / 1e6)
    return media


def cleanup(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
