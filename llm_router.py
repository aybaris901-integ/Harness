"""The single entry point for every LLM call in the project (CLAUDE.md §8).

Feature code calls `router.complete(...)` and never touches a provider SDK.
Free-tier limits tighten without warning, so the router walks a fallback chain:
rate limits and transient failures move on to the next provider.

The last link in the default chain is a paid tier (`openrouter-paid`). It is
only reached once every free provider has failed, and every call to it is
logged at WARNING with a `[PAID]` marker so spend can be tracked from the logs.

Two more entry points live here for the same reason — feature code never talks
to a provider directly:

- `build_vision_router`: a separate fallback chain of vision-capable models for
  handwriting transcription (Phase 5a). It is an ordinary `LLMRouter`; only the
  provider list differs. Callers must run the PII gate before using it.
- `Embedder`: text embeddings for the RAG index. No cross-provider fallback on
  purpose: vectors from different models live in incompatible spaces, so the
  index is pinned to one model (see storage/knowledge.py) and transient
  failures are retried against that same model instead.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from config import ProviderConfig, Settings
from providers import (
    ChatMessage,
    GeminiProvider,
    GroqProvider,
    ImagePart,
    InvalidRequest,
    LLMProvider,
    OpenRouterPaidProvider,
    OpenRouterProvider,
    ProviderError,
    ProviderUnavailable,
    RateLimitError,
    UnsupportedFeature,
)

logger = logging.getLogger(__name__)

# Transient failures are worth one retry against the same provider before
# falling through; rate limits are not (the next request would just 429 too).
_RETRIES_PER_PROVIDER = 2
_RETRY_BACKOFF_SECONDS = 1.0


class AllProvidersFailedError(RuntimeError):
    """Every provider in the chain failed. Carries the per-provider errors."""

    def __init__(self, errors: dict[str, Exception]) -> None:
        detail = "; ".join(f"{name}: {err}" for name, err in errors.items()) or "no providers"
        super().__init__(f"all LLM providers failed ({detail})")
        self.errors = errors


class LLMRouter:
    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        if not providers:
            raise ValueError("LLMRouter needs at least one provider")
        self.providers = list(providers)
        # Number of requests that reached a paid provider since startup —
        # the in-process counterpart of the `[PAID]` log lines.
        self.paid_calls = 0

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        *,
        history: Sequence[ChatMessage] | None = None,
        images: Sequence[ImagePart] | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return model text from the first provider in the chain that succeeds."""
        errors: dict[str, Exception] = {}

        for provider in self.providers:
            if provider.paid:
                # This is the only place the bot spends credit. Chain order
                # guarantees every free provider has already failed by now;
                # log at WARNING so a grep for "[PAID]" shows how often.
                self.paid_calls += 1
                logger.warning(
                    "[PAID] %s(%s) firing (paid call #%d this process) — free tiers failed: %s",
                    provider.name,
                    provider.model,
                    self.paid_calls,
                    "; ".join(f"{name}: {err}" for name, err in errors.items()) or "none tried",
                )
            for attempt in range(1, _RETRIES_PER_PROVIDER + 1):
                try:
                    text = await provider.complete(
                        prompt=prompt,
                        system=system,
                        history=history,
                        images=images,
                        json_schema=json_schema,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except RateLimitError as exc:
                    logger.warning("%s rate limited, falling through: %s", provider.name, exc)
                    errors[provider.name] = exc
                    break
                except ProviderUnavailable as exc:
                    errors[provider.name] = exc
                    if attempt < _RETRIES_PER_PROVIDER:
                        logger.warning(
                            "%s unavailable (attempt %d/%d), retrying: %s",
                            provider.name,
                            attempt,
                            _RETRIES_PER_PROVIDER,
                            exc,
                        )
                        await asyncio.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                        continue
                    logger.warning("%s unavailable, falling through: %s", provider.name, exc)
                    break
                except (UnsupportedFeature, InvalidRequest) as exc:
                    logger.warning("%s cannot serve this request: %s", provider.name, exc)
                    errors[provider.name] = exc
                    break
                except ProviderError as exc:
                    logger.warning("%s failed: %s", provider.name, exc)
                    errors[provider.name] = exc
                    break
                else:
                    if errors:
                        logger.info(
                            "%s succeeded after %d failed provider(s)", provider.name, len(errors)
                        )
                    return text

        raise AllProvidersFailedError(errors)

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()


def _build_provider(config: ProviderConfig, settings: Settings) -> LLMProvider:
    assert config.api_key is not None  # guaranteed by Settings.enabled_providers()
    common = {
        "api_key": config.api_key,
        "model": config.model,
        "timeout": settings.request_timeout,
    }
    if config.name == "gemini":
        return GeminiProvider(**common, thinking_budget=settings.gemini_thinking_budget)
    if config.name == "groq":
        return GroqProvider(**common, reasoning_effort=settings.groq_reasoning_effort)
    if config.name == "openrouter":
        return OpenRouterProvider(**common)
    if config.name == "openrouter-paid":
        return OpenRouterPaidProvider(**common)
    raise ValueError(f"unknown provider {config.name!r}")


def build_router(settings: Settings) -> LLMRouter:
    """Construct the router from settings, in `LLM_PROVIDER_CHAIN` order."""
    configs = settings.enabled_providers()
    providers = [_build_provider(config, settings) for config in configs]
    logger.info(
        "LLM chain: %s",
        " -> ".join(f"{p.name}({p.model}){' [PAID]' if p.paid else ''}" for p in providers)
        or "(none)",
    )
    # A paid tier is meant to be rare. If LLM_PROVIDER_CHAIN puts it ahead of a
    # free one, every request would spend credit before trying the free tier.
    first_paid = next((i for i, p in enumerate(providers) if p.paid), None)
    if first_paid is not None and any(not p.paid for p in providers[first_paid:]):
        logger.warning(
            "LLM_PROVIDER_CHAIN places a paid provider before a free one — "
            "paid calls will not be a last resort. Check the chain order."
        )
    return LLMRouter(providers)


def build_vision_router(settings: Settings) -> LLMRouter | None:
    """Vision-capable chain for notebook pages, in `VISION_PROVIDER_CHAIN` order.

    None when no provider in that chain has an API key — callers then fall back
    to local Tesseract only.
    """
    timeout = max(settings.request_timeout, 90.0)  # images upload slower than text
    providers: list[LLMProvider] = []
    for name in settings.vision_provider_chain:
        if name == "gemini" and settings.gemini.api_key:
            providers.append(
                GeminiProvider(
                    api_key=settings.gemini.api_key,
                    model=settings.gemini_vision_model,
                    timeout=timeout,
                    thinking_budget=settings.gemini_thinking_budget,
                )
            )
        elif name == "groq" and settings.groq.api_key:
            providers.append(
                GroqProvider(
                    api_key=settings.groq.api_key,
                    model=settings.groq_vision_model,
                    timeout=timeout,
                    supports_images=True,
                )
            )
        elif name == "openrouter" and settings.openrouter.api_key:
            providers.append(
                OpenRouterProvider(
                    api_key=settings.openrouter.api_key,
                    model=settings.openrouter_vision_model,
                    timeout=timeout,
                    supports_images=True,
                )
            )
    if not providers:
        return None
    logger.info("Vision chain: %s", " -> ".join(f"{p.name}({p.model})" for p in providers))
    return LLMRouter(providers)


class EmbeddingError(RuntimeError):
    """The embedding model could not be reached after retries."""


class Embedder:
    """Document/query embeddings from one pinned model (see module docstring)."""

    def __init__(self, provider: GeminiProvider, *, dimensions: int, retries: int = 3) -> None:
        self.provider = provider
        self.dimensions = dimensions
        self.retries = retries

    @property
    def model_id(self) -> str:
        """Stamped into the index; a different value means a reindex is needed."""
        return f"{self.provider.model}@{self.dimensions}"

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]

    async def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return await self.provider.embed(
                    texts, task_type=task_type, output_dimensionality=self.dimensions
                )
            except (RateLimitError, ProviderUnavailable) as exc:
                last = exc
                logger.warning("embedding attempt %d/%d failed: %s", attempt, self.retries, exc)
                if attempt < self.retries:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS * 2**attempt)
            except ProviderError as exc:
                raise EmbeddingError(str(exc)) from exc
        raise EmbeddingError(str(last))

    async def aclose(self) -> None:
        await self.provider.aclose()


def build_embedder(settings: Settings) -> Embedder | None:
    """None without GEMINI_API_KEY — the knowledge base is then disabled."""
    if not settings.gemini.api_key:
        return None
    provider = GeminiProvider(
        api_key=settings.gemini.api_key,
        model=settings.embedding_model,
        timeout=settings.request_timeout,
    )
    return Embedder(provider, dimensions=settings.embedding_dim)
