"""Link summarizer route (CLAUDE.md §7 Phase 2).

Orchestrates the article and video pipelines in `tools/`:

    article URL  -> tools.article  -> tools.summarizer
    video URL    -> tools.downloader (subtitles)              -> tools.summarizer
                 -> tools.downloader (audio) -> tools.transcriber -> tools.summarizer

No pipeline logic lives here — only the decision of which module to call next,
progress reporting, and translating tool errors into user-facing messages.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from llm_router import AllProvidersFailedError, LLMRouter
from tools import article as article_tool
from tools import downloader, summarizer, transcript
from tools.article import ArticleError
from tools.downloader import DownloadFailed, NotAVideo
from tools.transcriber import Transcriber, TranscriptionError
from tools.urls import LinkKind, classify_url

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]


class LinkError(RuntimeError):
    """A user-presentable failure of the link pipeline."""


@dataclass(slots=True)
class LinkSummary:
    url: str
    kind: LinkKind
    title: str | None
    summary: str
    # "article", "subtitles (en)", "auto-captions (en)", "speech-to-text (groq)"
    source: str


class LinkSummarizer:
    def __init__(
        self,
        *,
        router: LLMRouter,
        transcriber: Transcriber,
        download_dir: Path,
        max_video_minutes: float = 180.0,
        max_concurrent_videos: int = 2,
    ) -> None:
        self.router = router
        self.transcriber = transcriber
        self.download_dir = download_dir
        self.max_video_minutes = max_video_minutes
        # yt-dlp + ffmpeg + STT on a small VPS: serialize the heavy jobs.
        self._video_slots = asyncio.Semaphore(max_concurrent_videos)

    @staticmethod
    def classify(url: str) -> LinkKind:
        return classify_url(url)

    async def summarize(
        self, url: str, *, user_message: str, on_progress: ProgressCallback | None = None
    ) -> LinkSummary:
        kind = classify_url(url)
        if kind is LinkKind.VIDEO:
            try:
                return await self._summarize_video(
                    url, user_message=user_message, on_progress=on_progress
                )
            except NotAVideo:
                # A tweet / Instagram post with no media: read it as a page instead.
                logger.info("no media at %s, trying the article path", url)
        return await self._summarize_article(url, user_message=user_message)

    # -- article ---------------------------------------------------------------

    async def _summarize_article(self, url: str, *, user_message: str) -> LinkSummary:
        try:
            article = await article_tool.load_article(url)
        except ArticleError as exc:
            raise LinkError(f"Не удалось прочитать страницу: {exc}") from exc
        summary = await self._complete(
            summarizer.summarize_article(self.router, article, user_message=user_message)
        )
        return LinkSummary(
            url=url, kind=LinkKind.ARTICLE, title=article.title, summary=summary, source="article"
        )

    # -- video -----------------------------------------------------------------

    async def _summarize_video(
        self, url: str, *, user_message: str, on_progress: ProgressCallback | None
    ) -> LinkSummary:
        async with self._video_slots:
            work_dir = self.download_dir / uuid.uuid4().hex
            try:
                return await self._video_job(
                    url, work_dir, user_message=user_message, on_progress=on_progress
                )
            finally:
                downloader.cleanup(work_dir)

    async def _video_job(
        self,
        url: str,
        work_dir: Path,
        *,
        user_message: str,
        on_progress: ProgressCallback | None,
    ) -> LinkSummary:
        ffmpeg = self.transcriber.ffmpeg
        try:
            info = await downloader.probe(url, ffmpeg_path=ffmpeg)
        except NotAVideo:
            raise
        except DownloadFailed as exc:
            raise LinkError(f"Не удалось получить видео: {exc}") from exc

        if info.is_live:
            raise LinkError("Это прямая трансляция — её нельзя пересказать, пока она идёт.")

        segments: list[transcript.Segment] = []
        source = ""
        track = info.best_subtitle()
        if track is not None:
            source = f"{'auto-captions' if track.automatic else 'subtitles'} ({track.lang})"
            await _report(on_progress, f"Нашёл субтитры ({track.lang}), читаю…")
            try:
                segments = await downloader.fetch_subtitles(
                    url, track, work_dir, ffmpeg_path=ffmpeg
                )
            except DownloadFailed as exc:
                # Listed but not downloadable happens on YouTube; fall back to STT.
                logger.warning("subtitles for %s failed (%s), falling back to STT", url, exc)

        if not segments:
            segments, source = await self._transcribe(info, work_dir, on_progress=on_progress)

        await _report(on_progress, "Составляю пересказ…")
        transcript_text = transcript.format_transcript(segments)
        summary = await self._complete(
            summarizer.summarize_video(
                self.router,
                info,
                transcript_text,
                user_message=user_message,
                transcript_source=source,
            )
        )
        return LinkSummary(
            url=url, kind=LinkKind.VIDEO, title=info.title, summary=summary, source=source
        )

    async def _transcribe(
        self,
        info: downloader.VideoInfo,
        work_dir: Path,
        *,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[transcript.Segment], str]:
        if not self.transcriber.available:
            raise LinkError(
                "У видео нет субтитров, а распознавание речи не настроено "
                "(нужен GROQ_API_KEY или faster-whisper)."
            )
        minutes = info.duration_minutes
        if minutes is not None and minutes > self.max_video_minutes:
            raise LinkError(
                f"У видео нет субтитров, а оно слишком длинное для распознавания "
                f"({minutes:.0f} мин, лимит {self.max_video_minutes:.0f})."
            )

        length = f" ({minutes:.0f} мин)" if minutes else ""
        await _report(on_progress, f"Субтитров нет. Скачиваю аудио{length}…")
        try:
            audio_path = await downloader.fetch_audio(
                info.url, work_dir, ffmpeg_path=self.transcriber.ffmpeg
            )
        except DownloadFailed as exc:
            raise LinkError(f"Не удалось скачать аудио: {exc}") from exc

        await _report(on_progress, "Распознаю речь…")

        async def chunk_progress(done: str) -> None:
            await _report(on_progress, f"Распознаю речь… часть {done}")

        try:
            segments = await self.transcriber.transcribe(
                audio_path, work_dir, on_progress=chunk_progress
            )
        except TranscriptionError as exc:
            raise LinkError(f"Не удалось распознать речь: {exc}") from exc
        backend = "groq" if self.transcriber.groq_enabled else "faster-whisper"
        return segments, f"speech-to-text ({backend})"

    # -- shared ----------------------------------------------------------------

    @staticmethod
    async def _complete(call: Awaitable[str]) -> str:
        try:
            return await call
        except AllProvidersFailedError as exc:
            logger.error("summary failed: %s", exc)
            raise LinkError(
                "Все LLM-провайдеры сейчас недоступны. Попробуй ещё раз через минуту."
            ) from exc


async def _report(on_progress: ProgressCallback | None, text: str) -> None:
    if on_progress is None:
        return
    try:
        await on_progress(text)
    except Exception:  # progress is cosmetic; never let it kill the job
        logger.debug("progress callback failed", exc_info=True)
