"""Google AI Studio (Gemini) provider — the primary in the fallback chain.

Free tier, 1M token context, multimodal. Uses the REST API directly so the only
dependency is httpx.
"""

from __future__ import annotations

import base64
import logging
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

logger = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Keys that JSON Schema allows but Gemini's responseSchema rejects.
_SCHEMA_DROP_KEYS = frozenset(
    {"$schema", "$id", "additionalProperties", "definitions", "$defs", "default", "examples"}
)


def _sanitize_schema(schema: Any) -> Any:
    """Strip JSON Schema keywords Gemini's structured-output mode does not accept."""
    if isinstance(schema, dict):
        return {
            key: _sanitize_schema(value)
            for key, value in schema.items()
            if key not in _SCHEMA_DROP_KEYS
        }
    if isinstance(schema, list):
        return [_sanitize_schema(item) for item in schema]
    return schema


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        thinking_budget: int | None = None,
    ) -> None:
        super().__init__(api_key=api_key, model=model, timeout=timeout)
        # 0 disables "thinking" on 2.5-series models (faster, cheaper on free tier).
        # Leave as None for models that do not support the field at all.
        self.thinking_budget = thinking_budget

    def _build_payload(
        self,
        *,
        prompt: str,
        system: str | None,
        history: Sequence[ChatMessage] | None,
        images: Sequence[ImagePart] | None,
        json_schema: dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        for message in history or ():
            # Gemini calls the assistant role "model".
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for image in images or ():
            parts.append(
                {
                    "inline_data": {
                        "mime_type": image.mime_type,
                        "data": base64.b64encode(image.data).decode("ascii"),
                    }
                }
            )
        contents.append({"role": "user", "parts": parts})

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if json_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = _sanitize_schema(json_schema)
        if self.thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload

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
        payload = self._build_payload(
            prompt=prompt,
            system=system,
            history=history,
            images=images,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        url = f"{API_BASE}/models/{self.model}:generateContent"
        try:
            response = await self.client.post(
                url,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(self.name, f"request failed: {exc}") from exc

        self._raise_for_status(response)
        return self._extract_text(response.json())

    def _extract_text(self, data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise ProviderError(self.name, f"no candidates returned (blockReason={reason})")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            finish_reason = candidate.get("finishReason")
            if finish_reason == "MAX_TOKENS":
                raise ProviderError(
                    self.name, "response truncated before any text (raise max_tokens)"
                )
            raise ProviderError(self.name, f"empty response (finishReason={finish_reason})")
        return text
