"""Local sanity check — no Telegram, no bot token needed.

python scripts/selfcheck.py          # offline: storage + router fallback logic
python scripts/selfcheck.py --live   # also sends one real prompt via .env keys
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import Harness  # noqa: E402
from harness.prompts import TUTOR_SYSTEM_PROMPT  # noqa: E402
from llm_router import AllProvidersFailedError, LLMRouter  # noqa: E402
from providers.base import ChatMessage, LLMProvider, RateLimitError  # noqa: E402
from storage import Storage, UserProfile  # noqa: E402


class StubProvider(LLMProvider):
    """Provider that either always fails or always answers, for wiring tests."""

    def __init__(self, name: str, *, reply: str | None = None) -> None:
        super().__init__(api_key="stub", model="stub")
        self.name = name
        self.reply = reply
        self.calls = 0

    async def complete(self, *, prompt: str, system=None, history=None, **_kwargs) -> str:
        self.calls += 1
        if self.reply is None:
            raise RateLimitError(self.name, "stub quota exhausted")
        return f"{self.reply} (history={len(history or [])} turns)"


def check(label: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        raise SystemExit(1)


async def offline_checks() -> None:
    print("Storage:")
    # ignore_cleanup_errors: SQLite may still hold the WAL files on Windows.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        storage = Storage(Path(tmp) / "test.db")
        await storage.connect()
        try:
            # messages.telegram_id is a foreign key into users — in the bot this
            # row is created by UserTrackingMiddleware before any handler runs.
            await storage.upsert_user(
                UserProfile(telegram_id=1, username="tester", first_name="Test", language_code="ru")
            )

            failing = StubProvider("always-429")
            working = StubProvider("backup", reply="Ответ репетитора")
            router = LLMRouter([failing, working])
            harness = Harness(router=router, storage=storage, history_limit=10)

            first = await harness.start_lesson(telegram_id=1, chat_id=1, topic="рекурсия")
            check("lesson starts with an empty history", "history=0" in first)
            check("router fell through the rate-limited provider", working.calls == 1)

            second = await harness.continue_lesson(telegram_id=1, chat_id=1, text="мой ответ")
            check("second turn replays the stored history", "history=2" in second)
            check(
                "both turns persisted",
                await storage.message_count(telegram_id=1, chat_id=1) == 4,
            )

            history = await storage.recent_messages(telegram_id=1, chat_id=1, limit=10)
            check("history is oldest-first", history[0].content == "рекурсия")
            check("topic stored verbatim, not the wrapped prompt", history[0].role == "user")

            deleted = await harness.reset(telegram_id=1, chat_id=1)
            check("reset clears the conversation", deleted == 4)

            print("\nRouter:")
            dead_router = LLMRouter([StubProvider("a"), StubProvider("b")])
            try:
                await dead_router.complete("hi")
            except AllProvidersFailedError as exc:
                check("all-providers-failed names each provider", set(exc.errors) == {"a", "b"})
            else:
                check("all-providers-failed raised", False)
        finally:
            # aiosqlite runs a non-daemon worker thread; without this the
            # process hangs on exit when a check fails.
            await storage.close()


async def live_check() -> None:
    from config import load_settings
    from llm_router import build_router

    print("\nLive LLM call:")
    settings = load_settings()
    router = build_router(settings)
    try:
        reply = await router.complete(
            "Explain what a hash table is.",
            TUTOR_SYSTEM_PROMPT,
            history=[ChatMessage(role="user", content="Hi")],
            max_tokens=512,
        )
        check("provider returned text", bool(reply.strip()))
        print("\n--- model reply ---")
        print(reply)
        print("-------------------")
    finally:
        await router.aclose()


async def main() -> None:
    await offline_checks()
    if "--live" in sys.argv:
        await live_check()
    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
