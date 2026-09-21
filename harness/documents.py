"""Document archive route (CLAUDE.md §7 Phase 3).

    photo -> tools.ocr (local Tesseract)
          -> tools.document_fields (regex/heuristics; local-LLM fallback, also local-only)
          -> storage.documents (encrypted, keyed per Telegram user)

    query -> storage.documents.search (keyword/synonym match, never an LLM)
          -> matched fields + the original encrypted scan, decrypted for this reply only

No pipeline logic lives here — only the decision of which module to call next
and translating tool/storage errors into user-facing messages, mirroring
`harness/links.py`. Nothing in this file, or anything it calls, may go through
`llm_router.py` — that chain is cloud-only, and document images/text are PII
(CLAUDE.md §5).
"""

from __future__ import annotations

import logging

import strings
from storage.documents import DocumentMatch, DocumentRecord, DocumentStorageError, DocumentStore
from tools import document_fields, ocr
from tools.ocr import OcrError

logger = logging.getLogger(__name__)


class DocumentError(RuntimeError):
    """A user-presentable failure of the document pipeline."""


class DocumentArchive:
    def __init__(
        self,
        *,
        store: DocumentStore,
        tesseract_cmd: str | None,
        ocr_lang: str,
        ocr_tessdata_dir: str | None = None,
        local_llm_enabled: bool = False,
        local_llm_base_url: str = "",
        local_llm_model: str = "",
    ) -> None:
        self.store = store
        self.ocr_lang = ocr_lang
        self.ocr_tessdata_dir = ocr_tessdata_dir
        self.local_llm_enabled = local_llm_enabled
        self.local_llm_base_url = local_llm_base_url
        self.local_llm_model = local_llm_model
        self._tesseract_path = ocr.resolve_tesseract(tesseract_cmd)

    def describe(self) -> str:
        tesseract = self._tesseract_path or "NOT FOUND"
        fallback = (
            f"enabled ({self.local_llm_model} @ {self.local_llm_base_url})"
            if self.local_llm_enabled
            else "disabled"
        )
        tessdata = f", tessdata_dir={self.ocr_tessdata_dir}" if self.ocr_tessdata_dir else ""
        return (
            f"OCR: tesseract {tesseract} (lang={self.ocr_lang}{tessdata}); "
            f"local LLM fallback: {fallback}"
        )

    async def ingest(self, *, telegram_id: int, image_bytes: bytes) -> DocumentRecord:
        try:
            raw_text = await ocr.extract_text(
                image_bytes,
                tesseract_cmd=self._tesseract_path,
                lang=self.ocr_lang,
                tessdata_dir=self.ocr_tessdata_dir,
            )
        except OcrError as exc:
            raise DocumentError(strings.DOCUMENT_OCR_FAILED.format(error=exc)) from exc
        if not raw_text.strip():
            raise DocumentError(strings.DOCUMENT_NO_TEXT_FOUND)

        extracted = document_fields.extract_fields(raw_text)
        if self.local_llm_enabled and document_fields.needs_local_llm_fallback(extracted):
            logger.info("regex extraction found nothing, trying the local LLM fallback")
            llm_result = await document_fields.extract_with_local_llm(
                raw_text, base_url=self.local_llm_base_url, model=self.local_llm_model
            )
            if llm_result is not None:
                extracted = llm_result

        record = await self.store.save(
            telegram_id=telegram_id,
            document_type=extracted.document_type,
            fields=extracted.fields,
            raw_text=extracted.raw_text,
            image_bytes=image_bytes,
        )
        logger.info(
            "stored document %d for user %s: type=%s, %d field(s)",
            record.id,
            telegram_id,
            record.document_type,
            len(record.fields),
        )
        return record

    async def search(self, *, telegram_id: int, query: str) -> DocumentMatch | None:
        try:
            return await self.store.search(telegram_id=telegram_id, query=query)
        except DocumentStorageError as exc:
            raise DocumentError(strings.DOCUMENT_STORAGE_ERROR.format(error=exc)) from exc

    async def load_scan(self, record: DocumentRecord) -> bytes:
        try:
            return await self.store.load_scan(record)
        except DocumentStorageError as exc:
            raise DocumentError(strings.DOCUMENT_STORAGE_ERROR.format(error=exc)) from exc
