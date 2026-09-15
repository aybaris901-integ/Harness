"""Shared types and the provider interface used by `llm_router`.

A provider is a thin adapter over one vendor's HTTP API. It knows nothing about
the bot, the harness or storage — it takes a normalized request and returns text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

Role = Literal["user", "assistant"]


@dataclass(slots=True)
class ChatMessage:
    """One turn of conversation, provider-agnostic."""

    role: Role
    content: str


@dataclass(slots=True)
class ImagePart:
    """An image attached to the current user turn (multimodal providers only)."""

    data: bytes
    mime_type: str = "image/jpeg"


class ProviderError(RuntimeError):
    """Base class for provider failures."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class RateLimitError(ProviderError):
    """429 / quota exhausted — try the next provider in the chain."""


class ProviderUnavailable(ProviderError):
    """Network error or 5xx — retryable, then fall through to the next provider."""


class UnsupportedFeature(ProviderError):
    """The request needs something this provider cannot do (e.g. images)."""


class InvalidRequest(ProviderError):
    """4xx that retrying will not fix (bad model name, malformed body, bad key)."""


class LLMProvider(ABC):
    """Uniform interface every provider implements."""

    name: str
    # True for tiers that spend real credit. The router only reaches a paid
    # provider after every free one has failed, and logs each time it does.
    paid: bool = False

    def __init__(self, *, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    @abstractmethod
    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        images: Sequence[ImagePart] | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text response, or raise a `ProviderError` subclass."""

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Translate an HTTP error into the right `ProviderError` subclass."""
        if response.is_success:
            return
        body = response.text[:500]
        if response.status_code == 429:
            raise RateLimitError(self.name, f"rate limited / quota exhausted: {body}")
        if response.status_code >= 500:
            raise ProviderUnavailable(self.name, f"HTTP {response.status_code}: {body}")
        raise InvalidRequest(self.name, f"HTTP {response.status_code}: {body}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r}>"
