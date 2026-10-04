"""Application settings, loaded from the environment / `.env`.

Secrets never get hardcoded — see CLAUDE.md §5. This module is the only place
that reads `os.environ`; everything else takes a `Settings` instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet
from dotenv import load_dotenv

from tools.quiz_clock import parse_quiz_time, parse_utc_offset

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


class ConfigError(RuntimeError):
    """Raised when the environment is missing something the bot cannot run without."""


def _get(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_optional_int(name: str) -> int | None:
    raw = _get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _get_csv(name: str, default: str) -> tuple[str, ...]:
    raw = _get(name, default) or ""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _get_id_set(name: str) -> frozenset[int]:
    parts = _get_csv(name, "")
    try:
        return frozenset(int(part) for part in parts)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a comma-separated list of numeric IDs") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Everything `llm_router` needs to construct one provider."""

    name: str
    api_key: str | None
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True, slots=True)
class Settings:
    bot_token: str
    gemini: ProviderConfig
    groq: ProviderConfig
    openrouter: ProviderConfig
    # Same key as `openrouter`, but a paid model. Last tier in the default chain.
    openrouter_paid: ProviderConfig
    provider_chain: tuple[str, ...]
    gemini_thinking_budget: int | None
    db_path: Path
    history_limit: int
    request_timeout: float
    allowed_user_ids: frozenset[int]
    log_level: str
    # --- Phase 2: link/video summarizer ---
    download_dir: Path
    # auto = Groq if GROQ_API_KEY is set, else local faster-whisper if installed.
    stt_backend: str
    groq_whisper_model: str
    whisper_local_model: str
    ffmpeg_path: str | None
    max_video_minutes: float
    # --- Phase 3: document archive ---
    documents_db_path: Path
    documents_scan_dir: Path
    documents_encryption_key: bytes
    tesseract_cmd: str | None
    ocr_lang: str
    ocr_tessdata_dir: str | None
    document_local_llm_enabled: bool
    document_local_llm_base_url: str
    document_local_llm_model: str
    # --- Phase 4: media pipeline extras ---
    notes_db_path: Path
    media_max_download_mb: float
    media_max_concurrent: int
    # --- Phase 5a: RAG knowledge base ---
    groq_reasoning_effort: str | None
    knowledge_db_path: Path
    knowledge_page_dir: Path
    embedding_model: str
    embedding_dim: int
    rag_top_k: int
    rag_min_score: float
    vision_provider_chain: tuple[str, ...]
    gemini_vision_model: str
    groq_vision_model: str
    openrouter_vision_model: str
    # --- Phase 5b: flashcards + spaced repetition ---
    flashcards_db_path: Path
    quiz_default_time: str
    quiz_utc_offset_minutes: int
    quiz_daily_cap: int
    cards_max_sources_per_run: int
    pdf_max_mb: float

    @classmethod
    def from_env(cls) -> Settings:
        bot_token = _get("BOT_TOKEN")
        if not bot_token:
            raise ConfigError(
                "BOT_TOKEN is not set. Copy .env.example to .env and fill it in "
                "(get a token from @BotFather)."
            )

        db_path = Path(_get("DB_PATH", "data/harness.db") or "data/harness.db")
        if not db_path.is_absolute():
            db_path = BASE_DIR / db_path
        download_dir = Path(_get("DOWNLOAD_DIR", "data/downloads") or "data/downloads")
        if not download_dir.is_absolute():
            download_dir = BASE_DIR / download_dir

        stt_backend = (_get("STT_BACKEND", "auto") or "auto").lower()
        if stt_backend not in ("auto", "groq", "local"):
            raise ConfigError(f"STT_BACKEND must be auto, groq or local, got {stt_backend!r}")

        documents_db_path = Path(
            _get("DOCUMENTS_DB_PATH", "data/documents.db") or "data/documents.db"
        )
        if not documents_db_path.is_absolute():
            documents_db_path = BASE_DIR / documents_db_path
        documents_scan_dir = Path(
            _get("DOCUMENTS_SCAN_DIR", "data/document_scans") or "data/document_scans"
        )
        if not documents_scan_dir.is_absolute():
            documents_scan_dir = BASE_DIR / documents_scan_dir

        notes_db_path = Path(_get("NOTES_DB_PATH", "data/notes.db") or "data/notes.db")
        if not notes_db_path.is_absolute():
            notes_db_path = BASE_DIR / notes_db_path
        knowledge_db_path = Path(
            _get("KNOWLEDGE_DB_PATH", "data/knowledge.db") or "data/knowledge.db"
        )
        if not knowledge_db_path.is_absolute():
            knowledge_db_path = BASE_DIR / knowledge_db_path
        knowledge_page_dir = Path(
            _get("KNOWLEDGE_PAGE_DIR", "data/knowledge_pages") or "data/knowledge_pages"
        )
        if not knowledge_page_dir.is_absolute():
            knowledge_page_dir = BASE_DIR / knowledge_page_dir

        vision_chain = _get_csv("VISION_PROVIDER_CHAIN", "gemini,groq,openrouter")
        unknown_vision = set(vision_chain) - {"gemini", "groq", "openrouter"}
        if unknown_vision:
            raise ConfigError(
                f"VISION_PROVIDER_CHAIN contains unknown providers: "
                f"{', '.join(sorted(unknown_vision))}. Known: gemini, groq, openrouter."
            )
        flashcards_db_path = Path(
            _get("FLASHCARDS_DB_PATH", "data/flashcards.db") or "data/flashcards.db"
        )
        if not flashcards_db_path.is_absolute():
            flashcards_db_path = BASE_DIR / flashcards_db_path

        raw_quiz_time = _get("QUIZ_DEFAULT_TIME", "20:00") or "20:00"
        quiz_default_time = parse_quiz_time(raw_quiz_time)
        if quiz_default_time is None:
            raise ConfigError(f"QUIZ_DEFAULT_TIME must be HH:MM, got {raw_quiz_time!r}")
        # Default Almaty: Kazakhstan has been on a single UTC+5 zone, no DST,
        # since 2024 — a fixed offset needs no tz database (Termux-friendly).
        raw_offset = _get("QUIZ_UTC_OFFSET", "+05:00") or "+05:00"
        quiz_utc_offset_minutes = parse_utc_offset(raw_offset)
        if quiz_utc_offset_minutes is None:
            raise ConfigError(f"QUIZ_UTC_OFFSET must look like +05:00, got {raw_offset!r}")
        quiz_daily_cap = _get_int("QUIZ_DAILY_CAP", 10)
        if not 1 <= quiz_daily_cap <= 50:
            raise ConfigError(f"QUIZ_DAILY_CAP must be 1..50, got {quiz_daily_cap}")

        rag_min_score = _get_float("RAG_MIN_SCORE", 0.65)
        if not 0.0 < rag_min_score < 1.0:
            raise ConfigError(f"RAG_MIN_SCORE must be between 0 and 1, got {rag_min_score}")

        # CLAUDE.md §5: this key encrypts document scans + PII at rest. It must
        # come from .env, never be generated silently, and never be logged.
        raw_key = _get("DOCUMENTS_ENCRYPTION_KEY")
        if not raw_key:
            raise ConfigError(
                "DOCUMENTS_ENCRYPTION_KEY is not set. Generate one with:\n"
                '  python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"\n'
                "and put it in .env. It encrypts document scans and OCR text at rest — "
                "back it up somewhere other than the VPS, since losing it makes every "
                "stored document unrecoverable."
            )
        try:
            documents_encryption_key = raw_key.encode("ascii")
            Fernet(documents_encryption_key)
        except (ValueError, UnicodeEncodeError) as exc:
            raise ConfigError(f"DOCUMENTS_ENCRYPTION_KEY is not a valid Fernet key: {exc}") from exc

        openrouter_api_key = _get("OPENROUTER_API_KEY")
        settings = cls(
            bot_token=bot_token,
            gemini=ProviderConfig(
                name="gemini",
                api_key=_get("GEMINI_API_KEY"),
                model=_get("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
            ),
            groq=ProviderConfig(
                name="groq",
                api_key=_get("GROQ_API_KEY"),
                # llama-3.3-70b-versatile was deprecated by Groq on 2026-08-16
                # (HTTP 404 "does not exist"); verified openai/gpt-oss-120b is
                # live via GET https://api.groq.com/openai/v1/models.
                model=_get("GROQ_MODEL", "openai/gpt-oss-120b") or "openai/gpt-oss-120b",
            ),
            openrouter=ProviderConfig(
                name="openrouter",
                api_key=openrouter_api_key,
                model=(
                    _get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
                    or "meta-llama/llama-3.3-70b-instruct:free"
                ),
            ),
            openrouter_paid=ProviderConfig(
                name="openrouter-paid",
                api_key=openrouter_api_key,
                model=(
                    _get("OPENROUTER_PAID_MODEL", "google/gemini-3.8-flash")
                    or "google/gemini-3.8-flash"
                ),
            ),
            provider_chain=_get_csv("LLM_PROVIDER_CHAIN", "gemini,groq,openrouter,openrouter-paid"),
            gemini_thinking_budget=_get_optional_int("GEMINI_THINKING_BUDGET"),
            db_path=db_path,
            history_limit=_get_int("HISTORY_LIMIT", 20),
            request_timeout=_get_float("LLM_REQUEST_TIMEOUT", 60.0),
            allowed_user_ids=_get_id_set("ALLOWED_USER_IDS"),
            log_level=(_get("LOG_LEVEL", "INFO") or "INFO").upper(),
            download_dir=download_dir,
            stt_backend=stt_backend,
            groq_whisper_model=(
                _get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo") or "whisper-large-v3-turbo"
            ),
            whisper_local_model=_get("WHISPER_LOCAL_MODEL", "base") or "base",
            ffmpeg_path=_get("FFMPEG_PATH"),
            max_video_minutes=_get_float("MAX_VIDEO_MINUTES", 180.0),
            documents_db_path=documents_db_path,
            documents_scan_dir=documents_scan_dir,
            documents_encryption_key=documents_encryption_key,
            tesseract_cmd=_get("TESSERACT_CMD"),
            ocr_lang=_get("OCR_LANG", "eng") or "eng",
            ocr_tessdata_dir=_get("OCR_TESSDATA_DIR"),
            document_local_llm_enabled=_get_bool("DOCUMENT_LOCAL_LLM_ENABLED", False),
            document_local_llm_base_url=(
                _get("DOCUMENT_LOCAL_LLM_BASE_URL", "http://localhost:11434")
                or "http://localhost:11434"
            ),
            document_local_llm_model=_get("DOCUMENT_LOCAL_LLM_MODEL", "llama3.2") or "llama3.2",
            notes_db_path=notes_db_path,
            media_max_download_mb=_get_float("MEDIA_MAX_DOWNLOAD_MB", 45.0),
            media_max_concurrent=_get_int("MEDIA_MAX_CONCURRENT", 2),
            # "low" by default: gpt-oss reasoning tokens count against
            # max_tokens; empty value disables the field for non-reasoning models.
            groq_reasoning_effort=_get("GROQ_REASONING_EFFORT", "low"),
            knowledge_db_path=knowledge_db_path,
            knowledge_page_dir=knowledge_page_dir,
            # Chosen by the Phase 5a retrieval test (kk->kk and kk->ru/en
            # cross-lingual, 7/7 top-1). Changing model or dim needs a reindex:
            # the knowledge store refuses to mix vectors from different models.
            embedding_model=_get("EMBEDDING_MODEL", "gemini-embedding-2") or "gemini-embedding-2",
            embedding_dim=_get_int("EMBEDDING_DIM", 768),
            rag_top_k=_get_int("RAG_TOP_K", 5),
            rag_min_score=rag_min_score,
            vision_provider_chain=vision_chain,
            gemini_vision_model=_get("GEMINI_VISION_MODEL")
            or (_get("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"),
            groq_vision_model=_get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b") or "qwen/qwen3.8-27b",
            openrouter_vision_model=(
                _get("OPENROUTER_VISION_MODEL", "qwen/qwen3.8-27b:free") or "qwen/qwen3.8-27b:free"
            ),
            flashcards_db_path=flashcards_db_path,
            quiz_default_time=quiz_default_time,
            quiz_utc_offset_minutes=quiz_utc_offset_minutes,
            quiz_daily_cap=quiz_daily_cap,
            cards_max_sources_per_run=_get_int("CARDS_MAX_SOURCES_PER_RUN", 5),
            # Telegram bots can download files up to 20 MB.
            pdf_max_mb=_get_float("PDF_MAX_MB", 20.0),
        )

        known = set(settings._providers_by_name())
        unknown = set(settings.provider_chain) - known
        if unknown:
            raise ConfigError(
                f"LLM_PROVIDER_CHAIN contains unknown providers: {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(known))}."
            )
        if not settings.enabled_providers():
            raise ConfigError(
                "No LLM provider is usable: set at least one of GEMINI_API_KEY, "
                "GROQ_API_KEY, OPENROUTER_API_KEY (and keep it in LLM_PROVIDER_CHAIN)."
            )
        return settings

    def _providers_by_name(self) -> dict[str, ProviderConfig]:
        return {
            "gemini": self.gemini,
            "groq": self.groq,
            "openrouter": self.openrouter,
            "openrouter-paid": self.openrouter_paid,
        }

    def enabled_providers(self) -> list[ProviderConfig]:
        """Configured providers in fallback order, skipping any without an API key."""
        by_name = self._providers_by_name()
        return [by_name[name] for name in self.provider_chain if by_name[name].enabled]

    def is_user_allowed(self, telegram_id: int) -> bool:
        """Empty ALLOWED_USER_IDS means "open to everyone"."""
        return not self.allowed_user_ids or telegram_id in self.allowed_user_ids


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings.from_env()
