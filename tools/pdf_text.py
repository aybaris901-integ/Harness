"""Text extraction from PDFs (CLAUDE.md §7 Phase 5b flashcard sources).

`pypdf` only — a pure-Python wheel with no native dependencies, so it installs
anywhere Python does (Termux included). It reads the PDF's text layer; it
cannot OCR. A scanned PDF has no (or almost no) text layer, which is detected
here and reported, so the caller can point the user at /page (local OCR +
vision) instead of making cards from nothing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

MAX_PAGES = 60
# Below this many non-whitespace characters per page on average, the PDF is
# treated as scanned (images of pages, no text layer). A real text page has
# hundreds to thousands; scans typically yield 0 or a stray page number.
MIN_CHARS_PER_PAGE = 40


class PdfError(RuntimeError):
    """The file is not a readable PDF (corrupt, or encrypted with a password)."""


@dataclass(slots=True)
class PdfText:
    text: str
    pages_read: int
    total_pages: int
    scanned: bool

    @property
    def truncated(self) -> bool:
        return self.pages_read < self.total_pages


def _extract_sync(data: bytes, max_pages: int) -> PdfText:
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfError("the PDF is password-protected")
        total = len(reader.pages)
        pages: list[str] = []
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")
    except PdfError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise PdfError(f"could not read the PDF: {exc}") from exc

    text = "\n\n".join(page.strip() for page in pages if page.strip())
    visible = sum(1 for ch in text if not ch.isspace())
    scanned = visible < MIN_CHARS_PER_PAGE * max(1, len(pages))
    return PdfText(text=text, pages_read=len(pages), total_pages=total, scanned=scanned)


async def extract_text(data: bytes, *, max_pages: int = MAX_PAGES) -> PdfText:
    result = await asyncio.to_thread(_extract_sync, data, max_pages)
    logger.info(
        "pdf: %d/%d page(s) read, %d chars, scanned=%s",
        result.pages_read,
        result.total_pages,
        len(result.text),
        result.scanned,
    )
    return result
