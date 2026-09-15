"""The harness: the only component that talks to both the bot layer and the
tools/LLM layer (CLAUDE.md §2).

Routes so far:

- tutor (Phase 1): pure prompting, FSM-driven dialogue
- links (Phase 2): article / video summarizer, delegated to `harness.links`

Handlers call only this class; pipeline logic lives in `tools/`.
"""

from __future__ import annotations

import logging

from harness.links import LinkSummarizer, LinkSummary, ProgressCallback
from harness.prompts import TUTOR_SYSTEM_PROMPT, TUTOR_TOPIC_INSTRUCTION
from llm_router import AllProvidersFailedError, LLMRouter
from storage import Storage
from tools.urls import LinkKind

logger = logging.getLogger(__name__)

# Free-tier models ramble; the tutor prompt asks for ~150 words, this is the hard stop.
TUTOR_MAX_TOKENS = 1024
TUTOR_TEMPERATURE = 0.7


class HarnessError(RuntimeError):
    """A user-presentable failure from the harness."""


class Harness:
    def __init__(
        self,
        *,
        router: LLMRouter,
        storage: Storage,
        history_limit: int = 20,
        links: LinkSummarizer | None = None,
    ) -> None:
        self.router = router
        self.storage = storage
        self.history_limit = history_limit
        self.links = links

    # -- links (Phase 2) ----------------------------------------------------

    def link_kind(self, url: str) -> LinkKind:
        return LinkSummarizer.classify(url)

    async def summarize_link(
        self,
        *,
        telegram_id: int,
        chat_id: int,
        url: str,
        user_message: str,
        on_progress: ProgressCallback | None = None,
    ) -> LinkSummary:
        """Summarize one URL. `LinkError` carries a user-presentable message."""
        if self.links is None:
            raise HarnessError("Пересказ ссылок не настроен.")
        result = await self.links.summarize(url, user_message=user_message, on_progress=on_progress)
        # Keep the exchange in history so the tutor can answer follow-up
        # questions ("explain point 2") about what was just summarized.
        await self.storage.add_message(
            telegram_id=telegram_id, chat_id=chat_id, role="user", content=user_message
        )
        await self.storage.add_message(
            telegram_id=telegram_id, chat_id=chat_id, role="assistant", content=result.summary
        )
        return result

    # -- tutor (Phase 1) ----------------------------------------------------

    async def start_lesson(self, *, telegram_id: int, chat_id: int, topic: str) -> str:
        """Open a tutor session on `topic`. Clears any previous conversation."""
        await self.storage.clear_history(telegram_id=telegram_id, chat_id=chat_id)
        return await self._tutor_turn(
            telegram_id=telegram_id,
            chat_id=chat_id,
            stored_text=topic,
            prompt=TUTOR_TOPIC_INSTRUCTION.format(topic=topic),
        )

    async def continue_lesson(self, *, telegram_id: int, chat_id: int, text: str) -> str:
        """Feed the student's answer / follow-up back into the running lesson."""
        return await self._tutor_turn(
            telegram_id=telegram_id, chat_id=chat_id, stored_text=text, prompt=text
        )

    async def _tutor_turn(
        self, *, telegram_id: int, chat_id: int, stored_text: str, prompt: str
    ) -> str:
        history = await self.storage.recent_messages(
            telegram_id=telegram_id, chat_id=chat_id, limit=self.history_limit
        )
        try:
            reply = await self.router.complete(
                prompt,
                TUTOR_SYSTEM_PROMPT,
                history=history,
                temperature=TUTOR_TEMPERATURE,
                max_tokens=TUTOR_MAX_TOKENS,
            )
        except AllProvidersFailedError as exc:
            logger.error("tutor turn failed for user %s: %s", telegram_id, exc)
            raise HarnessError(
                "Все LLM-провайдеры сейчас недоступны. Попробуй ещё раз через минуту."
            ) from exc

        # Store what the student actually typed, not the wrapped prompt, so the
        # history stays a faithful transcript.
        await self.storage.add_message(
            telegram_id=telegram_id, chat_id=chat_id, role="user", content=stored_text
        )
        await self.storage.add_message(
            telegram_id=telegram_id, chat_id=chat_id, role="assistant", content=reply
        )
        return reply

    async def reset(self, *, telegram_id: int, chat_id: int) -> int:
        return await self.storage.clear_history(telegram_id=telegram_id, chat_id=chat_id)
