"""Providers that speak the OpenAI chat-completions dialect: Groq and OpenRouter.

Both are fallbacks in the chain (CLAUDE.md §4), and both accept the same request
shape, so they share one implementation. OpenRouter appears twice: once for the
`:free` pool and once as the paid last resort — same key, same endpoint,
different model.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import httpx

from providers.base import (
    ChatMessage,
    ImagePart,
    LLMProvider,
    ProviderError,
    ProviderUnavailable,
)


class OpenAICompatibleProvider(LLMProvider):
    """Base for any `/chat/completions` endpoint."""

    base_url: str
    supports_images: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(api_key=api_key, model=model, timeout=timeout)
        self.extra_headers = extra_headers or {}

    def _build_messages(
        self,
        *,
        prompt: str,
        system: str | None,
        history: Sequence[ChatMessage] | None,
        images: Sequence[ImagePart] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for message in history or ():
            messages.append({"role": message.role, "content": message.content})

        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image in images:
                encoded = base64.b64encode(image.data).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        return messages

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(
                prompt=prompt, system=system, history=history, images=images
            ),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(self.name, f"request failed: {exc}") from exc

        self._raise_for_status(response)
        return self._extract_text(response.json())

    def _extract_text(self, data: dict[str, Any]) -> str:
        # OpenRouter reports upstream failures as a 200 with an `error` object.
        if isinstance(data.get("error"), dict):
            raise ProviderError(self.name, str(data["error"].get("message", data["error"])))
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(self.name, "no choices returned")
        text = (choices[0].get("message") or {}).get("content") or ""
        text = text.strip()
        if not text:
            raise ProviderError(
                self.name, f"empty response (finish_reason={choices[0].get('finish_reason')})"
            )
        return text


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(self, *, api_key: str, model: str, timeout: float = 60.0) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            # OpenRouter uses these for attribution on free models.
            extra_headers={
                "HTTP-Referer": "https://github.com/local/harness-bot",
                "X-Title": "Harness Telegram Bot",
            },
        )


class OpenRouterPaidProvider(OpenRouterProvider):
    """Last resort in the chain: a paid OpenRouter model, billed against credit.

    Only reached when Gemini, Groq and the OpenRouter free pool have all failed.
    Keep the model cheap — this is the one place the bot spends money.
    """

    name = "openrouter-paid"
    paid = True
