"""RAG knowledge base (CLAUDE.md §7 Phase 5a).

    notebook page photo
        -> tools.ocr (LOCAL Tesseract)  ─┐ PII gate, before any cloud call:
        -> document_fields.has_pii_signals ─┘ fires -> encrypted document archive,
                                              exactly like /note (harness/media.py)
        -> tools.vision_ocr (vision chain: Gemini -> Groq -> OpenRouter)
           (all fail -> the Tesseract text from the gate, printed text only)
        -> has_pii_signals again on the transcription (handwriting Tesseract
           could not read) -> archive instead of the plain index
        -> tools.rag.chunk_text -> Embedder -> storage.knowledge
           (+ the original photo, Fernet-encrypted, for answers to send back)

    /note staging rows (storage.notes) -> same chunk/embed/store path, once each

    question -> Embedder.embed_query -> storage.knowledge.search (this user's
                partition only) -> drop hits under RAG_MIN_SCORE -> none left:
                "nothing in your notes" (static Kazakh string, no LLM call)
             -> tools.rag.answer_question (grounded, structured) -> not
                answerable: same static string -> else answer + sources + photo

The PII gate fails closed: if local OCR is unavailable or errors, the page is
refused rather than sent to a cloud vision API unchecked (CLAUDE.md §5). No
pipeline logic lives here, only the routing between modules, mirroring
harness/media.py and harness/documents.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import strings
from harness.documents import DocumentArchive, DocumentError
from llm_router import AllProvidersFailedError, Embedder, EmbeddingError, LLMRouter
from storage.documents import DocumentRecord
from storage.encrypted_files import EncryptedFileError
from storage.knowledge import ChunkHit, KnowledgeSource, KnowledgeStore
from storage.notes import NoteRecord, NoteStore
from tools import document_fields, ocr, rag, vision_ocr
from tools.ocr import OcrError

logger = logging.getLogger(__name__)

_TITLE_MAX_CHARS = 60


class KnowledgeError(RuntimeError):
    """A user-presentable failure of the knowledge-base pipeline."""


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NOTHING_RELEVANT = "nothing_relevant"  # no chunk passed the similarity threshold
    NOT_IN_NOTES = "not_in_notes"  # chunks retrieved, but the model found no answer in them


@dataclass(slots=True)
class PageCapture:
    """Result of `ingest_page`: exactly one of `source` / `document` is set."""

    redirected_to_documents: bool
    # True when only the vision transcription revealed PII, i.e. the image
    # did go to the vision chain before being redirected.
    sent_to_vision: bool = False
    source: KnowledgeSource | None = None
    document: DocumentRecord | None = None
    transcription: str = ""
    transcribed_by: str = ""  # "vision" | "tesseract"
    chunk_count: int = 0


@dataclass(slots=True)
class KnowledgeAnswer:
    status: AnswerStatus
    answer: str = ""
    sources: list[KnowledgeSource] = field(default_factory=list)
    photo: bytes | None = None
    photo_source: KnowledgeSource | None = None
    # Every retrieved chunk with its score, threshold or not — for logs/selfcheck.
    hits: list[ChunkHit] = field(default_factory=list)


class KnowledgeBase:
    def __init__(
        self,
        *,
        store: KnowledgeStore,
        embedder: Embedder,
        router: LLMRouter,
        vision_router: LLMRouter | None,
        note_store: NoteStore | None = None,
        document_archive: DocumentArchive | None = None,
        tesseract_cmd: str | None = None,
        ocr_lang: str = "eng",
        ocr_tessdata_dir: str | None = None,
        top_k: int = 5,
        min_score: float = 0.65,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.router = router
        self.vision_router = vision_router
        self.note_store = note_store
        self.document_archive = document_archive
        self.ocr_lang = ocr_lang
        self.ocr_tessdata_dir = ocr_tessdata_dir
        self.top_k = top_k
        self.min_score = min_score
        self._tesseract_path = ocr.resolve_tesseract(tesseract_cmd)

    def describe(self) -> str:
        vision = (
            " -> ".join(f"{p.name}({p.model})" for p in self.vision_router.providers)
            if self.vision_router
            else "none (Tesseract only)"
        )
        gate = self._tesseract_path or "NOT FOUND — notebook pages will be refused"
        return (
            f"Knowledge base: {self.embedder.model_id}, top_k={self.top_k}, "
            f"min_score={self.min_score}; vision: {vision}; PII gate OCR: {gate}"
        )

    # -- ingestion: notebook pages ---------------------------------------------

    async def ingest_page(
        self, *, telegram_id: int, image_bytes: bytes, title_hint: str = ""
    ) -> PageCapture:
        gate_text = await self._local_ocr_gate(image_bytes)
        if fired := document_fields.pii_signals(gate_text):
            logger.warning(
                "user %s sent a notebook page that looks like a personal document (%s) — "
                "not sending it to a cloud vision API; redirecting to the document archive",
                telegram_id,
                document_fields.describe_pii_signals(fired),
            )
            return await self._redirect_to_archive(telegram_id, image_bytes)

        text, transcribed_by = await self._transcribe(image_bytes, gate_text)
        if not text.strip():
            raise KnowledgeError(strings.KB_PAGE_NO_TEXT)

        if transcribed_by == "vision" and (fired := document_fields.pii_signals(text)):
            # Handwritten PII that Tesseract could not read. The image has
            # already been through the vision chain, but it must still not
            # land in the plain index — store it encrypted instead.
            logger.warning(
                "vision transcription for user %s shows PII signals the local gate missed "
                "(%s) — storing in the document archive instead of the knowledge index",
                telegram_id,
                document_fields.describe_pii_signals(fired),
            )
            return await self._redirect_to_archive(telegram_id, image_bytes, sent_to_vision=True)

        title = title_hint.strip() or _title_from_text(text)
        source, chunk_count = await self._index(
            telegram_id=telegram_id, kind="page", title=title, text=text, photo=image_bytes
        )
        logger.info(
            "indexed page %d for user %s (%s, %d chunks)",
            source.id,
            telegram_id,
            transcribed_by,
            chunk_count,
        )
        return PageCapture(
            redirected_to_documents=False,
            source=source,
            transcription=text,
            transcribed_by=transcribed_by,
            chunk_count=chunk_count,
        )

    async def _local_ocr_gate(self, image_bytes: bytes) -> str:
        if self._tesseract_path is None:
            raise KnowledgeError(strings.KB_PII_GATE_UNAVAILABLE)
        try:
            return await ocr.extract_text(
                image_bytes,
                tesseract_cmd=self._tesseract_path,
                lang=self.ocr_lang,
                tessdata_dir=self.ocr_tessdata_dir,
            )
        except OcrError as exc:
            logger.error("local OCR (PII gate) failed, refusing the page: %s", exc)
            raise KnowledgeError(strings.KB_PII_GATE_UNAVAILABLE) from exc

    async def _transcribe(self, image_bytes: bytes, gate_text: str) -> tuple[str, str]:
        if self.vision_router is not None:
            try:
                text = await vision_ocr.transcribe_page(self.vision_router, image_bytes)
                if text.strip():
                    return text, "vision"
                logger.info("vision chain returned no text; falling back to Tesseract")
            except AllProvidersFailedError as exc:
                logger.warning("vision chain failed, falling back to Tesseract: %s", exc)
        return gate_text.strip(), "tesseract"

    async def _redirect_to_archive(
        self, telegram_id: int, image_bytes: bytes, *, sent_to_vision: bool = False
    ) -> PageCapture:
        if self.document_archive is None:
            raise KnowledgeError(strings.KB_PII_NO_ARCHIVE)
        try:
            document = await self.document_archive.ingest(
                telegram_id=telegram_id, image_bytes=image_bytes
            )
        except DocumentError as exc:
            raise KnowledgeError(strings.KB_PII_ARCHIVE_FAILED.format(error=exc)) from exc
        return PageCapture(
            redirected_to_documents=True, sent_to_vision=sent_to_vision, document=document
        )

    # -- ingestion: /note staging rows ---------------------------------------------

    async def index_note(self, note: NoteRecord) -> KnowledgeSource | None:
        """Index one staging note; no-op (None) if it is already indexed."""
        origin = f"note:{note.id}"
        if await self.store.has_origin(origin):
            return None
        source, _ = await self._index(
            telegram_id=note.telegram_id,
            kind="note",
            title=note.title,
            text=note.content_md,
            origin=origin,
        )
        return source

    async def migrate_staging_notes(self) -> int:
        """Index every staging note not yet in the index. Idempotent; safe at
        every startup. Returns how many notes were newly indexed."""
        if self.note_store is None:
            return 0
        migrated = 0
        after_id = 0
        while batch := await self.note_store.iter_notes(after_id=after_id):
            for note in batch:
                after_id = note.id
                try:
                    if await self.index_note(note) is not None:
                        migrated += 1
                except KnowledgeError as exc:
                    # Leave it for the next startup rather than aborting the rest.
                    logger.warning("could not index staging note %d: %s", note.id, exc)
        if migrated:
            logger.info("migrated %d staging note(s) into the knowledge index", migrated)
        return migrated

    async def _index(
        self,
        *,
        telegram_id: int,
        kind: str,
        title: str,
        text: str,
        origin: str | None = None,
        photo: bytes | None = None,
    ) -> tuple[KnowledgeSource, int]:
        chunks = rag.chunk_text(text)
        if not chunks:
            raise KnowledgeError(strings.KB_PAGE_NO_TEXT)
        try:
            vectors = await self.embedder.embed_documents(
                [rag.embedding_input(title, chunk) for chunk in chunks]
            )
        except EmbeddingError as exc:
            logger.error("embedding failed: %s", exc)
            raise KnowledgeError(strings.KB_EMBEDDING_FAILED) from exc
        source = await self.store.add_source(
            telegram_id=telegram_id,
            kind=kind,
            title=title,
            chunks=chunks,
            embeddings=vectors,
            origin=origin,
            photo=photo,
        )
        return source, len(chunks)

    # -- query ------------------------------------------------------------------

    async def ask(self, *, telegram_id: int, question: str, user_message: str) -> KnowledgeAnswer:
        try:
            query_vector = await self.embedder.embed_query(question)
        except EmbeddingError as exc:
            logger.error("query embedding failed: %s", exc)
            raise KnowledgeError(strings.KB_EMBEDDING_FAILED) from exc

        hits = await self.store.search(
            telegram_id=telegram_id, embedding=query_vector, k=self.top_k
        )
        relevant = [hit for hit in hits if hit.score >= self.min_score]
        best = hits[0].score if hits else None
        if not relevant:
            self._log_outcome(telegram_id, "refused:threshold", best, len(relevant), len(hits))
            return KnowledgeAnswer(status=AnswerStatus.NOTHING_RELEVANT, hits=hits)

        excerpts = [rag.Excerpt(title=hit.source.title, text=hit.text) for hit in relevant]
        try:
            grounded = await rag.answer_question(
                self.router, question, excerpts, user_message=user_message
            )
        except AllProvidersFailedError as exc:
            logger.error("kb answer failed: %s", exc)
            raise KnowledgeError(strings.ALL_PROVIDERS_FAILED) from exc
        except rag.AnswerFormatError as exc:
            logger.error("kb answer unparseable: %s", exc)
            raise KnowledgeError(strings.KB_ANSWER_FAILED) from exc

        if not grounded.answerable:
            self._log_outcome(telegram_id, "refused:model", best, len(relevant), len(hits))
            return KnowledgeAnswer(status=AnswerStatus.NOT_IN_NOTES, hits=hits)
        self._log_outcome(
            telegram_id, "answered", best, len(relevant), len(hits), cited=grounded.sources
        )

        # Cited excerpts, in citation order; the best hit if the model cited none.
        cited_hits = [relevant[number - 1] for number in grounded.sources] or relevant[:1]
        sources: list[KnowledgeSource] = []
        for hit in cited_hits:
            if all(hit.source.id != seen.id for seen in sources):
                sources.append(hit.source)

        photo_source = next((s for s in sources if s.photo_filename), None)
        photo = None
        if photo_source is not None:
            try:
                photo = await self.store.load_photo(
                    telegram_id=telegram_id, source_id=photo_source.id
                )
            except EncryptedFileError as exc:
                logger.error("could not load page photo %d: %s", photo_source.id, exc)
                photo_source = None
        return KnowledgeAnswer(
            status=AnswerStatus.ANSWERED,
            answer=grounded.answer,
            sources=sources,
            photo=photo,
            photo_source=photo_source,
            hits=hits,
        )

    def _log_outcome(
        self,
        telegram_id: int,
        outcome: str,
        best: float | None,
        passed: int,
        retrieved: int,
        *,
        cited: list[int] | None = None,
    ) -> None:
        """One line per /ask, for tuning RAG_MIN_SCORE from real use:
        outcome is "answered", "refused:threshold" (no chunk passed, the LLM was
        never called) or "refused:model" (chunks passed, but the model marked
        the question unanswerable from them). The question text is not logged."""
        logger.info(
            "kb /ask user=%s outcome=%s best=%s threshold=%.2f passed=%d/%d%s",
            telegram_id,
            outcome,
            f"{best:.3f}" if best is not None else "none",
            self.min_score,
            passed,
            retrieved,
            f" cited={cited}" if cited is not None else "",
        )


def _title_from_text(text: str) -> str:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if len(first) > _TITLE_MAX_CHARS:
        first = first[: _TITLE_MAX_CHARS - 1].rstrip() + "…"
    return first or strings.KB_UNTITLED
