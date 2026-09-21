"""Local sanity check — no Telegram, no bot token needed.

python scripts/selfcheck.py                 # offline: storage, router, URL/VTT, documents
python scripts/selfcheck.py --live          # also sends one real prompt via .env keys
python scripts/selfcheck.py --link <url>    # run the Phase 2 pipeline on one article/video URL

Document checks (Phase 3) cover regex field extraction, search scoring and the
encrypted store's save/search/decrypt roundtrip — all offline, no Tesseract
binary or LLM required, since none of that pipeline calls out over the network.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

# Document checks print Kazakh/Russian sample text; Windows terminals default
# to a legacy codepage (cp1252) that can't encode it and would crash the run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptography.fernet import Fernet  # noqa: E402

from harness import Harness  # noqa: E402
from harness.prompts import TUTOR_SYSTEM_PROMPT  # noqa: E402
from llm_router import AllProvidersFailedError, LLMRouter  # noqa: E402
from providers.base import ChatMessage, LLMProvider, RateLimitError  # noqa: E402
from storage import DocumentStore, Storage, UserProfile  # noqa: E402
from tools import document_fields  # noqa: E402
from tools.transcript import Segment, format_transcript, parse_vtt  # noqa: E402
from tools.urls import LinkKind, classify_url, extract_urls  # noqa: E402


class StubProvider(LLMProvider):
    """Provider that either always fails or always answers, for wiring tests."""

    def __init__(self, name: str, *, reply: str | None = None, paid: bool = False) -> None:
        super().__init__(api_key="stub", model="stub")
        self.name = name
        self.reply = reply
        self.paid = paid
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

            print("\nPaid tier:")
            free_ok = StubProvider("free", reply="free answer")
            paid = StubProvider("paid", reply="paid answer", paid=True)
            router = LLMRouter([free_ok, paid])
            await router.complete("hi")
            check("paid tier not called while a free one works", paid.calls == 0)
            check("paid counter stays at zero", router.paid_calls == 0)

            paid = StubProvider("paid", reply="paid answer", paid=True)
            router = LLMRouter([StubProvider("free-a"), StubProvider("free-b"), paid])
            reply = await router.complete("hi")
            check("paid tier reached only after all free tiers fail", reply.startswith("paid"))
            check("paid calls are counted", router.paid_calls == 1)
        finally:
            # aiosqlite runs a non-daemon worker thread; without this the
            # process hangs on exit when a check fails.
            await storage.close()


def phase2_offline_checks() -> None:
    print("\nURL detection:")
    urls = extract_urls("see https://example.com/a?x=1, and www.youtu.be/abc). ok")
    check("extracts both URLs", urls == ["https://example.com/a?x=1", "https://www.youtu.be/abc"])
    check("trailing punctuation trimmed", not urls[0].endswith(","))
    check("no URL -> empty list", extract_urls("just text") == [])
    check("youtube is video", classify_url("https://m.youtube.com/watch?v=x") is LinkKind.VIDEO)
    check("youtu.be is video", classify_url("https://youtu.be/x") is LinkKind.VIDEO)
    check("tiktok is video", classify_url("https://vm.tiktok.com/ZM/") is LinkKind.VIDEO)
    check("blog is article", classify_url("https://blog.example.org/post") is LinkKind.ARTICLE)

    print("\nSubtitle parsing:")
    vtt = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000 align:start position:0%
hello<00:00:01.000><c> world</c>

00:00:02.000 --> 00:00:04.000
hello world
second line

00:01:05.500 --> 00:01:07.000
much later
"""
    segments = parse_vtt(vtt)
    check(
        "rolling duplicate lines collapsed",
        [s.text for s in segments] == ["hello world", "second line", "much later"],
    )
    check("timestamps parsed", segments[2].start == 65.5)
    rendered = format_transcript(segments, window_seconds=30)
    check("windows rendered with [mm:ss]", rendered.startswith("[00:00] hello world second line"))
    check("new window after 30s", "[01:05] much later" in rendered)
    check("hours rendered", format_transcript([Segment(3661, 3662, "x")]).startswith("[1:01:01]"))


SAMPLE_VEHICLE_OCR = """
VEHICLE REGISTRATION CERTIFICATE
VIN: 1HGCM82633A004352
Plate: 123 ABC 45
Owner: Aigerim Bekova
Issue date: 05.03.2021
"""

SAMPLE_PASSPORT_OCR = """
PASSPORT / ПАСПОРТ
IIN: 900101300123
Surname: Bekov
Date of birth: 01.01.1990
"""

# A synthetic (fake data, no real PII) reproduction of the bug report: an
# OCR pass so badly garbled under the wrong Tesseract language pack that
# every Cyrillic/Kazakh label is unreadable noise, while the fixed-width TD3
# MRZ block (bottom two lines) survives intact — exactly like the real
# passport that got extracted fields but document_type="unknown".
SAMPLE_GARBLED_PASSPORT_OCR = """
�SITbI / HAUMOHAIbHOCTb
sie SS  ~~
| eee awe DK
�TYPI/ TYPE MEMMEKET KOJIbI
Bah AE a9 a KAZ ail
MACTIOPT MEPSIMI/ DATE OF EXPIRY
25.06.2029

P<KAZTESTOV<<TESTBEK<<<<<<<<<<<<<<<<<<<<<<<<<
N000000015KAZ9001010F3001010990101300123<<00
"""


def phase3_offline_checks() -> None:
    """Regex/heuristic extraction — no Tesseract, no LLM (CLAUDE.md §5/§9)."""
    print("\nDocument field extraction:")
    vehicle = document_fields.extract_fields(SAMPLE_VEHICLE_OCR)
    check("vehicle registration type detected", vehicle.document_type == "vehicle_registration")
    check("VIN extracted", vehicle.fields.get("vin") == "1HGCM82633A004352")
    check("plate extracted", vehicle.fields.get("plate_number") == "123ABC45")
    check(
        "issue date canonicalized regardless of label wording",
        vehicle.fields.get("date_of_issue") == "05.03.2021",
    )

    passport = document_fields.extract_fields(SAMPLE_PASSPORT_OCR)
    check("passport type detected", passport.document_type == "passport")
    check("IIN extracted", passport.fields.get("iin") == "900101300123")
    check("surname canonicalized", passport.fields.get("surname") == "Bekov")
    check("date of birth canonicalized", passport.fields.get("date_of_birth") == "01.01.1990")

    print("\nMRZ extraction (regression: type must agree with extracted fields):")
    garbled = document_fields.extract_fields(SAMPLE_GARBLED_PASSPORT_OCR)
    check(
        "type is 'passport', not 'unknown', once MRZ fields are extracted",
        garbled.document_type == "passport",
    )
    check("MRZ document number extracted", garbled.fields.get("document_number") == "N00000001")
    check("MRZ nationality extracted", garbled.fields.get("nationality") == "KAZ")
    check("MRZ date of birth extracted", garbled.fields.get("date_of_birth") == "01.01.1990")
    check("MRZ sex extracted", garbled.fields.get("sex") == "F")
    check("MRZ date of expiry extracted", garbled.fields.get("date_of_expiry") == "01.01.2030")
    check("MRZ personal number stored as iin", garbled.fields.get("iin") == "990101300123")
    check(
        "needs_local_llm_fallback is False once MRZ classified the document",
        not document_fields.needs_local_llm_fallback(garbled),
    )

    print("\nDocument search scoring:")
    vehicle_fields = ["vin", "plate_number"]
    score, matched = document_fields.score_query(
        "көлік тіркеу нөмірін жібер",
        document_type="vehicle_registration",
        field_names=vehicle_fields,
    )
    check("vehicle query scores positive on type keyword", score > 0)
    score2, matched2 = document_fields.score_query(
        "vin көрсет", document_type="vehicle_registration", field_names=vehicle_fields
    )
    check("vin field name matched directly", "vin" in matched2)
    unrelated_score, _ = document_fields.score_query(
        "паспорт нөмірі", document_type="vehicle_registration", field_names=["vin"]
    )
    check("passport query does not match vehicle doc", unrelated_score == 0)

    print("\nMultilingual document-type search (кк/ru/en all resolve the same way):")
    passport_fields = ["iin", "document_number"]
    for query in ("паспорт", "passport", "жеке куәлік"):
        score, _ = document_fields.score_query(
            query, document_type="passport", field_names=passport_fields
        )
        check(f"{query!r} matches document_type=passport", score > 0)

    print("\nSearch is not gated on document_type when fields exist:")
    score, matched = document_fields.score_query(
        "иин", document_type="unknown", field_names=["iin", "date"]
    )
    check("an 'unknown'-type document is still findable by its field data", score > 0)
    check("the matched field is reported", "iin" in matched)


async def document_store_offline_check() -> None:
    """Encrypted save -> search -> load_scan roundtrip (CLAUDE.md §7 Phase 3)."""
    print("\nDocument store (encrypted):")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = DocumentStore(
            db_path=Path(tmp) / "documents.db",
            scan_dir=Path(tmp) / "scans",
            encryption_key=Fernet.generate_key(),
        )
        await store.connect()
        try:
            fake_scan = b"not a real jpeg, just bytes for the roundtrip test"
            record = await store.save(
                telegram_id=1,
                document_type="vehicle_registration",
                fields={"vin": "1HGCM82633A004352", "plate_number": "123ABC45"},
                raw_text=SAMPLE_VEHICLE_OCR,
                image_bytes=fake_scan,
            )
            check("save assigns an id", record.id > 0)

            on_disk = record.scan_path.read_bytes()
            check("scan is encrypted at rest", on_disk != fake_scan)

            match = await store.search(telegram_id=1, query="көлік vin нөмірі")
            found_right_doc = match is not None and match.record.id == record.id
            check("search finds the stored document", found_right_doc)
            check("search decrypts the fields", match.record.fields["vin"] == "1HGCM82633A004352")

            no_match = await store.search(telegram_id=1, query="паспорт")
            check("unrelated query finds nothing", no_match is None)

            other_user = await store.search(telegram_id=2, query="vin")
            check("documents are isolated per user", other_user is None)

            scan_bytes = await store.load_scan(record)
            check("decrypted scan matches the original bytes", scan_bytes == fake_scan)
        finally:
            await store.close()


async def link_check(url: str) -> None:
    """Run the full Phase 2 pipeline (no Telegram) and print the summary."""
    from config import load_settings
    from harness import LinkError, LinkSummarizer
    from llm_router import build_router
    from tools.transcriber import Transcriber

    print(f"\nLink pipeline: {url}")
    settings = load_settings()
    router = build_router(settings)
    transcriber = Transcriber(
        backend=settings.stt_backend,
        groq_api_key=settings.groq.api_key,
        groq_model=settings.groq_whisper_model,
        local_model=settings.whisper_local_model,
        ffmpeg_path=settings.ffmpeg_path,
    )
    print("  " + transcriber.describe())
    links = LinkSummarizer(
        router=router,
        transcriber=transcriber,
        download_dir=settings.download_dir,
        max_video_minutes=settings.max_video_minutes,
    )

    async def progress(text: str) -> None:
        print(f"  ... {text}")

    try:
        result = await links.summarize(url, user_message=url, on_progress=progress)
    except LinkError as exc:
        check(f"pipeline failed: {exc}", False)
    finally:
        await router.aclose()
    check("summary returned", bool(result.summary.strip()))
    print(f"\n--- {result.kind.value}: {result.title!r} (source: {result.source}) ---")
    print(result.summary)
    print("-------------------")


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
    if "--link" in sys.argv:
        index = sys.argv.index("--link")
        if index + 1 >= len(sys.argv):
            raise SystemExit("usage: selfcheck.py --link <url>")
        await link_check(sys.argv[index + 1])
        print("\nLink check passed.")
        return
    await offline_checks()
    phase2_offline_checks()
    phase3_offline_checks()
    await document_store_offline_check()
    if "--live" in sys.argv:
        await live_check()
    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
