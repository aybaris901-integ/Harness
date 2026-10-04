"""Media pipeline extras (CLAUDE.md §7 Phase 4).

    any URL           -> tools.downloader.fetch_media (download, optional mp3)
                       -> tools.media_notes.describe_media (LLM description/timestamps)

    screenshot photo   -> tools.ocr (local Tesseract, same kaz+rus+eng config as
                          Phase 3 — CLAUDE.md §5 explicitly carves screenshots
                          out as the non-sensitive case, unlike
                          harness/documents.py's PII-only-local path)
                       -> tools.document_fields.has_pii_signals (safety net: a
                          passport/vehicle-registration photo sent with a
                          /note caption by mistake is redirected into the
                          encrypted document archive instead, never the
                          unencrypted notes table below)
                       -> tools.media_notes.format_screenshot (LLM -> title + tags
                          only; the OCR text is stored verbatim, never rewritten)
                       -> storage.notes (staging table; Phase 5's vector store
                          migrates from this later)

No pipeline logic lives here — only the decision of which module to call next,
progress reporting, and translating tool/storage errors into user-facing
messages, mirroring `harness/links.py` and `harness/documents.py`. Not wired
through `Harness` (unlike `links`): like `DocumentArchive`, it's injected into
the bot as its own workflow-data object, since neither download nor screenshot
capture needs the tutor's chat-history threading.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import strings
from harness.documents import DocumentArchive, DocumentError
from llm_router import AllProvidersFailedError, LLMRouter
from storage.documents import DocumentRecord
from storage.notes import NoteRecord, NoteStore
from tools import document_fields, downloader, media_notes, ocr
from tools.downloader import DownloadFailed
from tools.media_notes import ScreenshotFormatError
from tools.ocr import OcrError
from tools.transcriber import resolve_ffmpeg

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]

_T = TypeVar("_T")


class MediaError(RuntimeError):
    """A user-presentable failure of the Phase 4 media pipeline."""


@dataclass(slots=True)
class MediaDownload:
    filename: str
    data: bytes
    title: str
    description: str
    is_audio: bool
    # Passed to send_video so Telegram renders the right aspect ratio.
    width: int | None = None
    height: int | None = None
    duration: float | None = None


@dataclass(slots=True)
class ScreenshotCapture:
    """Result of `MediaPipeline.capture_screenshot`. Exactly one of `note` /
    `document` is set — see `redirected_to_documents`."""

    redirected_to_documents: bool
    note: NoteRecord | None = None
    document: DocumentRecord | None = None


class MediaPipeline:
    def __init__(
        self,
        *,
        router: LLMRouter,
        download_dir: Path,
        note_store: NoteStore,
        ffmpeg_path: str | None = None,
        tesseract_cmd: str | None = None,
        ocr_lang: str = "eng",
        ocr_tessdata_dir: str | None = None,
        max_download_mb: float = 45.0,
        max_concurrent_downloads: int = 2,
        document_archive: DocumentArchive | None = None,
    ) -> None:
        self.router = router
        self.download_dir = download_dir
        self.note_store = note_store
        # PII safety net for /note (see capture_screenshot): when set, a
        # screenshot that trips document_fields.has_pii_signals is routed
        # here instead of the unencrypted notes table. None only in contexts
        # that don't wire one up (e.g. an ad-hoc script) — capture_screenshot
        # still refuses to save unencrypted in that case, it just can't
        # redirect automatically.
        self.document_archive = document_archive
        self.ocr_lang = ocr_lang
        self.ocr_tessdata_dir = ocr_tessdata_dir
        self.max_download_mb = max_download_mb
        self._ffmpeg_path = resolve_ffmpeg(ffmpeg_path)
        self._tesseract_path = ocr.resolve_tesseract(tesseract_cmd)
        # yt-dlp + ffmpeg on a small VPS: serialize the heavy jobs, same
        # reasoning as LinkSummarizer's video semaphore.
        self._slots = asyncio.Semaphore(max_concurrent_downloads)

    def describe(self) -> str:
        tesseract = self._tesseract_path or "NOT FOUND"
        ffmpeg = self._ffmpeg_path or "NOT FOUND (mp3 conversion will fail)"
        return (
            f"Media pipeline: downloads -> {self.download_dir} "
            f"(max {self.max_download_mb:.0f} MB, ffmpeg {ffmpeg}); "
            f"screenshot OCR: tesseract {tesseract} (lang={self.ocr_lang})"
        )

    # -- generalized downloader (Phase 4) --------------------------------------

    async def download(
        self,
        url: str,
        *,
        convert_to_mp3: bool,
        user_message: str,
        on_progress: ProgressCallback | None = None,
    ) -> MediaDownload:
        async with self._slots:
            work_dir = self.download_dir / uuid.uuid4().hex
            try:
                return await self._download_job(
                    url,
                    work_dir,
                    convert_to_mp3=convert_to_mp3,
                    user_message=user_message,
                    on_progress=on_progress,
                )
            finally:
                downloader.cleanup(work_dir)

    async def _download_job(
        self,
        url: str,
        work_dir: Path,
        *,
        convert_to_mp3: bool,
        user_message: str,
        on_progress: ProgressCallback | None,
    ) -> MediaDownload:
        try:
            info = await downloader.probe(url, ffmpeg_path=self._ffmpeg_path)
        except DownloadFailed as exc:
            raise MediaError(strings.MEDIA_PROBE_FAILED.format(error=exc)) from exc
        if info.is_live:
            raise MediaError(strings.LIVESTREAM_NOT_SUPPORTED)

        await _report(on_progress, strings.MEDIA_DOWNLOADING)
        max_bytes = int(self.max_download_mb * 1024 * 1024)
        try:
            fetched = await downloader.fetch_media(
                url,
                work_dir,
                ffmpeg_path=self._ffmpeg_path,
                convert_to_mp3=convert_to_mp3,
                max_bytes=max_bytes,
            )
        except DownloadFailed as exc:
            raise MediaError(strings.MEDIA_DOWNLOAD_FAILED.format(error=exc)) from exc

        path = fetched.path
        size_mb = path.stat().st_size / 1e6
        if size_mb > self.max_download_mb:
            # yt-dlp's max_filesize is a best-effort pre-check, not a hard
            # guarantee (e.g. it can't know the size of some streams up
            # front) — a Telegram upload of an oversized file just fails, so
            # this is checked explicitly before spending an LLM call on it.
            raise MediaError(
                strings.MEDIA_TOO_LARGE.format(size=size_mb, limit=self.max_download_mb)
            )

        await _report(on_progress, strings.MEDIA_DESCRIBING)
        description = await self._complete(
            media_notes.describe_media(self.router, info, user_message=user_message)
        )

        data = await asyncio.to_thread(path.read_bytes)
        return MediaDownload(
            filename=path.name,
            data=data,
            title=info.title,
            description=description,
            is_audio=convert_to_mp3,
            width=fetched.width,
            height=fetched.height,
            duration=fetched.duration or info.duration,
        )

    # -- screenshot OCR + note formatting (Phase 4) ----------------------------

    async def capture_screenshot(
        self, *, telegram_id: int, image_bytes: bytes, user_message: str
    ) -> ScreenshotCapture:
        try:
            raw_text = await ocr.extract_text(
                image_bytes,
                tesseract_cmd=self._tesseract_path,
                lang=self.ocr_lang,
                tessdata_dir=self.ocr_tessdata_dir,
            )
        except OcrError as exc:
            raise MediaError(strings.NOTE_OCR_FAILED.format(error=exc)) from exc
        if not raw_text.strip():
            raise MediaError(strings.NOTE_NO_TEXT_FOUND)

        # Safety net (CLAUDE.md §5): /note is meant for non-sensitive
        # screenshots only. If the OCR'd text shows the same signals
        # tools.document_fields already uses to recognize a passport/vehicle
        # document, don't let it reach the unencrypted staging table below —
        # redirect it into the encrypted document archive instead, the same
        # pipeline a plain (uncaptioned) photo would have gone through.
        if document_fields.has_pii_signals(raw_text):
            logger.warning(
                "user %s sent /note on what looks like a personal document — "
                "redirecting to the encrypted document archive instead of notes",
                telegram_id,
            )
            if self.document_archive is None:
                raise MediaError(strings.NOTE_PII_NO_ARCHIVE)
            try:
                document = await self.document_archive.ingest(
                    telegram_id=telegram_id, image_bytes=image_bytes
                )
            except DocumentError as exc:
                raise MediaError(strings.NOTE_PII_ARCHIVE_FAILED.format(error=exc)) from exc
            return ScreenshotCapture(redirected_to_documents=True, document=document)

        try:
            note = await self._complete(
                media_notes.format_screenshot(self.router, raw_text, user_message=user_message)
            )
        except ScreenshotFormatError as exc:
            raise MediaError(strings.NOTE_FORMAT_FAILED.format(error=exc)) from exc

        record = await self.note_store.save(
            telegram_id=telegram_id,
            kind="screenshot",
            title=note.title,
            # Verbatim OCR output (outer whitespace aside) — not model output.
            content_md=raw_text.strip(),
            content_json=note.raw_json,
            tags=note.tags,
        )
        logger.info(
            "saved screenshot note %d for user %s (%d tags)",
            record.id,
            telegram_id,
            len(note.tags),
        )
        return ScreenshotCapture(redirected_to_documents=False, note=record)

    # -- shared -----------------------------------------------------------------

    @staticmethod
    async def _complete(call: Awaitable[_T]) -> _T:
        try:
            return await call
        except AllProvidersFailedError as exc:
            logger.error("media pipeline LLM call failed: %s", exc)
            raise MediaError(strings.ALL_PROVIDERS_FAILED) from exc


async def _report(on_progress: ProgressCallback | None, text: str) -> None:
    if on_progress is None:
        return
    try:
        await on_progress(text)
    except Exception:  # progress is cosmetic; never let it kill the job
        logger.debug("progress callback failed", exc_info=True)
