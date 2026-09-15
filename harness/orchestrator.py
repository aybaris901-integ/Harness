"""The harness: the only component that talks to both the bot layer and the
tools/LLM layer (CLAUDE.md §2).

Phase 1 has exactly one route — the tutor — and no tools. Later phases add
intent routing and tool pipelines here; handlers keep calling only this class.
"""

from __future__ import annotations

import logging

from harness.prompts import TUTOR_SYSTEM_PROMPT, TUTOR_TOPIC_INSTRUCTION
from llm_router import AllProvidersFailedError, LLMRouter
from storage import Storage

logger = logging.getLogger(__name__)

# Free-tier models ramble; the tutor prompt asks for ~150 words, this is the hard stop.
TUTOR_MAX_TOKENS = 1024
TUTOR_TEMPERATURE = 0.7


class HarnessError(RuntimeError):
    """A user-presentable failure from the harness."""


class Harness:
    def __init__(self, *, router: LLMRouter, storage: Storage, history_limit: int = 20) -> None:
        self.router = router
        self.storage = storage
        self.history_limit = history_limit

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
