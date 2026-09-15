"""Video pipeline, step 2 (fallback): speech-to-text when a video has no subtitles.

Backends, tried in order under `STT_BACKEND=auto`:

1. Whisper via Groq's API (`whisper-large-v3-turbo`, free tier, ~25 MB per file)
2. `faster-whisper` locally (optional dependency; slow on a small VPS, but private)

If ffmpeg is available the audio is first re-encoded to 16 kHz mono at 32 kbps
and cut into chunks, so even a two-hour video fits under Groq's file limit.
Without ffmpeg the raw download is sent as-is and long videos are refused.

This deliberately does NOT go through `llm_router` — STT is not a chat
completion and has its own providers/limits.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from tools.transcript import Segment

logger = logging.getLogger(__name__)

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
# Groq free tier rejects files over 25 MB; keep headroom for container overhead.
GROQ_MAX_BYTES = 24 * 1024 * 1024
# 20-minute chunks at 32 kbps ≈ 4.8 MB each.
CHUNK_SECONDS = 20 * 60
_FFMPEG_AUDIO_ARGS = ["-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k"]

ProgressCallback = Callable[[str], Any]


class TranscriptionError(RuntimeError):
    """User-presentable failure of the speech-to-text step."""


def resolve_ffmpeg(configured: str | None) -> str | None:
    """Path to the ffmpeg executable, from config or PATH; None if absent."""
    if configured:
        candidate = Path(configured)
        if candidate.is_dir():
            for name in ("ffmpeg.exe", "ffmpeg"):
                if (candidate / name).exists():
                    return str(candidate / name)
        elif candidate.exists():
            return str(candidate)
        logger.warning("FFMPEG_PATH=%r does not point at ffmpeg; falling back to PATH", configured)
    return shutil.which("ffmpeg")


class Transcriber:
    def __init__(
        self,
        *,
        backend: str = "auto",
        groq_api_key: str | None = None,
        groq_model: str = "whisper-large-v3-turbo",
        local_model: str = "base",
        ffmpeg_path: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        if backend not in ("auto", "groq", "local"):
            raise ValueError(f"unknown STT backend {backend!r}")
        self.backend = backend
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.local_model_name = local_model
        self.ffmpeg = resolve_ffmpeg(ffmpeg_path)
        self.timeout = timeout
        self._local_model: Any = None

    # -- capability checks -------------------------------------------------

    @property
    def groq_enabled(self) -> bool:
        return self.backend in ("auto", "groq") and bool(self.groq_api_key)

    @property
    def local_enabled(self) -> bool:
        if self.backend not in ("auto", "local"):
            return False
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def available(self) -> bool:
        return self.groq_enabled or self.local_enabled

    def describe(self) -> str:
        parts = []
        if self.groq_enabled:
            parts.append(f"groq({self.groq_model})")
        if self.local_enabled:
            parts.append(f"faster-whisper({self.local_model_name})")
        backends = " -> ".join(parts) or "(none)"
        return f"STT: {backends}; ffmpeg: {self.ffmpeg or 'not found'}"

    # -- public API ---------------------------------------------------------

    async def transcribe(
        self,
        audio_path: Path,
        work_dir: Path,
        *,
        language: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[Segment]:
        if not self.available:
            raise TranscriptionError(
                "no speech-to-text backend is configured: set GROQ_API_KEY or "
                "pip install faster-whisper"
            )
        chunks = await asyncio.to_thread(self._prepare_chunks, audio_path, work_dir)
        segments: list[Segment] = []
        for index, (offset, chunk) in enumerate(chunks, start=1):
            if on_progress and len(chunks) > 1:
                await _maybe_await(on_progress(f"{index}/{len(chunks)}"))
            chunk_segments = await self._transcribe_chunk(chunk, language=language)
            for segment in chunk_segments:
                segment.start += offset
                segment.end += offset
            segments.extend(chunk_segments)
        logger.info(
            "transcribed %s: %d chunk(s), %d segments", audio_path.name, len(chunks), len(segments)
        )
        return segments

    # -- audio preparation ----------------------------------------------------

    def _prepare_chunks(self, audio_path: Path, work_dir: Path) -> list[tuple[float, Path]]:
        """Re-encode + split with ffmpeg when possible, else use the file as-is."""
        if self.ffmpeg:
            chunk_dir = work_dir / "chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            pattern = chunk_dir / "chunk_%03d.mp3"
            command = [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                *_FFMPEG_AUDIO_ARGS,
                "-f",
                "segment",
                "-segment_time",
                str(CHUNK_SECONDS),
                "-reset_timestamps",
                "1",
                str(pattern),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise TranscriptionError(
                    f"ffmpeg failed to convert the audio: {result.stderr.strip()[:300]}"
                )
            chunks = sorted(chunk_dir.glob("chunk_*.mp3"))
            if not chunks:
                raise TranscriptionError("ffmpeg produced no audio")
            return [(index * CHUNK_SECONDS, path) for index, path in enumerate(chunks)]

        size = audio_path.stat().st_size
        if self.groq_enabled and not self.local_enabled and size > GROQ_MAX_BYTES:
            raise TranscriptionError(
                f"the audio is {size / 1e6:.0f} MB, over Groq's 25 MB limit — install "
                "ffmpeg so it can be compressed and split, or pick a shorter video"
            )
        return [(0.0, audio_path)]

    # -- backends -----------------------------------------------------------

    async def _transcribe_chunk(self, path: Path, *, language: str | None) -> list[Segment]:
        errors: list[str] = []
        if self.groq_enabled and path.stat().st_size <= GROQ_MAX_BYTES:
            try:
                return await self._transcribe_groq(path, language=language)
            except TranscriptionError as exc:
                logger.warning("groq STT failed, %s", exc)
                errors.append(str(exc))
        elif self.groq_enabled:
            errors.append("chunk too large for Groq")
        if self.local_enabled:
            try:
                return await asyncio.to_thread(self._transcribe_local, path, language)
            except TranscriptionError as exc:
                errors.append(str(exc))
        raise TranscriptionError("; ".join(errors) or "no usable STT backend")

    async def _transcribe_groq(self, path: Path, *, language: str | None) -> list[Segment]:
        data: dict[str, str] = {
            "model": self.groq_model,
            "response_format": "verbose_json",
            "temperature": "0",
        }
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {self.groq_api_key}"}
        try:
            with path.open("rb") as handle:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        GROQ_TRANSCRIPTIONS_URL,
                        headers=headers,
                        data=data,
                        files={"file": (path.name, handle, "audio/mpeg")},
                    )
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"groq request failed: {exc}") from exc
        if response.status_code == 429:
            raise TranscriptionError("groq rate limit reached, try again in a minute")
        if not response.is_success:
            raise TranscriptionError(f"groq HTTP {response.status_code}: {response.text[:200]}")

        payload = response.json()
        segments = [
            Segment(
                start=float(item.get("start") or 0),
                end=float(item.get("end") or 0),
                text=str(item.get("text") or "").strip(),
            )
            for item in payload.get("segments") or []
        ]
        segments = [segment for segment in segments if segment.text]
        if not segments:
            text = str(payload.get("text") or "").strip()
            if not text:
                raise TranscriptionError("groq returned an empty transcript")
            segments = [Segment(start=0.0, end=0.0, text=text)]
        return segments

    def _transcribe_local(self, path: Path, language: str | None) -> list[Segment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - guarded by local_enabled
            raise TranscriptionError("faster-whisper is not installed") from exc
        if self._local_model is None:
            logger.info("loading faster-whisper model %r (first use)", self.local_model_name)
            self._local_model = WhisperModel(
                self.local_model_name, device="cpu", compute_type="int8"
            )
        try:
            raw_segments, _info = self._local_model.transcribe(
                str(path), language=language, vad_filter=True
            )
            segments = [
                Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
                for s in raw_segments
                if s.text.strip()
            ]
        except Exception as exc:  # faster-whisper raises plain RuntimeErrors
            raise TranscriptionError(f"local whisper failed: {exc}") from exc
        if not segments:
            raise TranscriptionError("local whisper returned an empty transcript")
        return segments


async def _maybe_await(result: Any) -> None:
    if asyncio.iscoroutine(result):
        await result
