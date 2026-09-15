"""Application settings, loaded from the environment / `.env`.

Secrets never get hardcoded — see CLAUDE.md §5. This module is the only place
that reads `os.environ`; everything else takes a `Settings` instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

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
                model=_get("GROQ_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile",
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
            provider_chain=_get_csv(
                "LLM_PROVIDER_CHAIN", "gemini,groq,openrouter,openrouter-paid"
            ),
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
