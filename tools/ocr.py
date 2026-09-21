"""Local OCR for the document archive (CLAUDE.md §5, Phase 3).

Runs Tesseract entirely on this machine — a document photo is decoded and
OCR'd in a worker thread and the raw pixels never leave the process, let alone
the server. This module must never call `llm_router.py` or any cloud vision
API; that routing decision belongs in code, not in a model's judgement.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from io import BytesIO
from pathlib import Path

import pytesseract
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


class OcrError(RuntimeError):
    """User-presentable failure of the local OCR step."""


def resolve_tesseract(configured: str | None) -> str | None:
    """Path to the tesseract executable, from config or PATH; None if absent."""
    if configured:
        candidate = Path(configured)
        if candidate.is_dir():
            for name in ("tesseract.exe", "tesseract"):
                if (candidate / name).exists():
                    return str(candidate / name)
        elif candidate.exists():
            return str(candidate)
        logger.warning(
            "TESSERACT_CMD=%r does not point at tesseract; falling back to PATH", configured
        )
    return shutil.which("tesseract") or shutil.which("tesseract.exe")


def _run_ocr(
    image_bytes: bytes, *, tesseract_cmd: str | None, lang: str, tessdata_dir: str | None
) -> str:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise OcrError(f"could not decode the image: {exc}") from exc

    # --tessdata-dir replaces Tesseract's search path outright (it doesn't
    # append), so it's only passed when a directory was actually configured —
    # e.g. for a user-writable language-pack location on a machine where the
    # system tessdata dir (under Program Files) isn't writable without admin.
    # pytesseract tokenizes `config` with shlex.split(..., posix=False) on
    # Windows, which does NOT strip quote characters the way posix mode does
    # — a quoted path here would reach Tesseract with literal `"` in it. Only
    # unquoted paths work portably, so this assumes tessdata_dir has no spaces.
    config = f"--tessdata-dir {tessdata_dir}" if tessdata_dir else ""
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrError(
            "Tesseract OCR is not installed or not found — set TESSERACT_CMD "
            "or install it and add it to PATH"
        ) from exc
    except pytesseract.TesseractError as exc:
        raise OcrError(f"OCR failed: {exc}") from exc
    return text.strip()


async def extract_text(
    image_bytes: bytes,
    *,
    tesseract_cmd: str | None = None,
    lang: str = "eng",
    tessdata_dir: str | None = None,
) -> str:
    """OCR one image, off the event loop — Tesseract shells out and blocks."""
    text = await asyncio.to_thread(
        _run_ocr, image_bytes, tesseract_cmd=tesseract_cmd, lang=lang, tessdata_dir=tessdata_dir
    )
    logger.info("OCR extracted %d chars (lang=%s)", len(text), lang)
    return text
