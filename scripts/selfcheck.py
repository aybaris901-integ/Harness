"""Local sanity check — no Telegram, no bot token needed.

python scripts/selfcheck.py                    # offline: storage, router, URL/VTT, documents, notes
python scripts/selfcheck.py --live             # also sends one real prompt via .env keys
python scripts/selfcheck.py --link <url>       # run the Phase 2 pipeline on one article/video URL
python scripts/selfcheck.py --download <url> [mp3]  # run the Phase 4 downloader on one URL
python scripts/selfcheck.py --screenshot <path>     # run the Phase 4 screenshot pipeline

Document checks (Phase 3) cover regex field extraction, search scoring and the
encrypted store's save/search/decrypt roundtrip — all offline, no Tesseract
binary or LLM required, since none of that pipeline calls out over the network.

Phase 4's offline checks cover the note staging store's roundtrip and the
screenshot formatter's JSON parsing (via a stub provider) — also no network.
`--download`/`--screenshot` are the live, real-network/real-OCR equivalents of
`--link`, since the generalized downloader and screenshot OCR both need a real
yt-dlp fetch / real Tesseract install to meaningfully exercise.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

# Document checks print Kazakh/Russian sample text; Windows terminals default
# to a legacy codepage (cp1252) that can't encode it and would crash the run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptography.fernet import Fernet  # noqa: E402

import strings  # noqa: E402
from harness import Harness  # noqa: E402
from harness.prompts import TUTOR_SYSTEM_PROMPT  # noqa: E402
from llm_router import AllProvidersFailedError, LLMRouter  # noqa: E402
from providers.base import ChatMessage, LLMProvider, RateLimitError  # noqa: E402
from storage import DocumentStore, NoteStore, Storage, UserProfile  # noqa: E402
from tools import document_fields, media_notes  # noqa: E402
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


class JsonStubProvider(LLMProvider):
    """Returns text verbatim, no history-turn suffix — for testing callers
    that parse a structured-output (JSON) reply, like `tools.media_notes`."""

    def __init__(self, reply: str) -> None:
        super().__init__(api_key="stub", model="stub")
        self.name = "json-stub"
        self.reply = reply

    async def complete(self, **_kwargs) -> str:
        return self.reply


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


# SYNTHETIC study-material samples (written for these checks, not copied from
# any textbook) with the shape of the /page false positive: problem labels
# (Берілгені / Табу керек / Шешуі / Жауабы), formulas, units, long numbers, and
# everyday words whose stems used to collide with gate keywords ("көлемі" =
# volume matched the vehicle keyword "көлік").
SYNTHETIC_TEXTBOOK_PAGES = {
    "kk chemistry worksheet": """
§12. Газдардың молярлық көлемі
7-есеп. Қалыпты жағдайда 4,4 г CO2 қандай көлем алады?
Берілгені: m(CO2) = 4,4 г
M(CO2) = 44 г/моль
Табу керек: V(CO2) - ?
Шешуі: n = m / M = 4,4 / 44 = 0,1 моль
V = n · Vm = 0,1 · 22,4 = 2,24 л
Жауабы: 2,24 л
8-есеп. C6H12O6 + 6O2 -> 6CO2 + 6H2O реакциясындағы оттектің көлемін есепте.
Жауабы: C,H,O; 134,4 л
""",
    "kk geography + physics": """
Балқаш көлі — Қазақстандағы ірі көлдердің бірі, ауданы 16 400 км².
Көлдің көлемі шамамен 112 км³.
Көліктің жылдамдығы v = 72 км/сағ = 20 м/с. Табу керек: t - ?
Жауабы: t = s / v = 1200 / 20 = 60 с
""",
    "ru physics worksheet": """
Задача 3. Определите объём тела массой 2,7 кг, если плотность 2700 кг/м³.
Дано: m = 2,7 кг; ρ = 2700 кг/м³
Найти: V - ?
Решение: V = m / ρ = 0,001 м³
Ответ: 0,001 м³ = 1 дм³
Постоянная Авогадро: 602214076000000000000000 (≈ 6,02·10²³ моль⁻¹)
""",
    "en chemistry + vocabulary": """
Vinegar is a 5% solution of acetic acid, CH3COOH. Vinyl chloride: C2H3Cl.
Vocabulary: nationality, citizen, identity, register (verb).
Exercise 4: 123456789012 + 987654321098 = 1111111110110
Answer: 12345678901234567 is a 17-digit number.
""",
}

# SYNTHETIC Kaspi-style transfer receipt (fake names/numbers) — classification
# must stay "receipt" after the gate/extraction changes.
SAMPLE_RECEIPT_OCR = """
Перевод успешно совершен
5 000 ₸
№ квитанции
Отправитель
Получатель
4012345678901
Тестов А.
Примеров Б.
"""


def pii_gate_regression_checks() -> None:
    """Phase 5a regression: the /page false positive on a printed textbook page."""
    print("\nPII gate: study material is NOT flagged (synthetic samples):")
    for label, text in SYNTHETIC_TEXTBOOK_PAGES.items():
        fired = document_fields.pii_signals(text)
        check(
            f"{label}: no signal ({document_fields.describe_pii_signals(fired) or 'clear'})",
            not fired,
        )

    print("\nPII gate: real-document samples still flagged:")
    for label, text in (
        ("passport (title keyword + IIN)", SAMPLE_PASSPORT_OCR),
        ("garbled passport (MRZ only)", SAMPLE_GARBLED_PASSPORT_OCR),
        ("vehicle registration (VIN)", SAMPLE_VEHICLE_OCR),
        ("Russian inflection: «серия паспорта»", "Серия паспорта: N0000001"),
        ("Kazakh inflection: «жеке куәлігі»", "Менің жеке куәлігім жоғалды"),
        ("vehicle cert heading only", "ТЕХПАСПОРТ\nМарка: Test"),
        ("Kazakh state plate label", "Мемлекеттік нөмірі: 123 ABC 02"),
        ("handwritten IIN line", "ЖСН: 900101300123"),
        ("bare VIN line", "VIN: 1HGCM82633A004352"),
    ):
        fired = document_fields.pii_signals(text)
        check(f"{label}: {document_fields.describe_pii_signals(fired)}", bool(fired))
    check(
        "logged summary never contains the IIN/VIN value itself",
        "900101300123"
        not in document_fields.describe_pii_signals(
            document_fields.pii_signals(SAMPLE_PASSPORT_OCR)
        ),
    )

    print("\nDocument type of samples unchanged:")
    check(
        "receipt sample still classified as receipt",
        document_fields.extract_fields(SAMPLE_RECEIPT_OCR).document_type == "receipt",
    )
    check(
        "receipt fields unchanged (receipt number + sender)",
        # Same output as before this change (verified against HEAD). "amount"
        # is not extracted from "5 000 ₸" — a pre-existing _AMOUNT_RE issue,
        # out of scope here.
        {
            k: v
            for k, v in document_fields.extract_fields(SAMPLE_RECEIPT_OCR).fields.items()
            if k in ("receipt_number", "sender")
        }
        == {"receipt_number": "4012345678901", "sender": "Тестов А."},
    )

    print("\nStudy material is not classified as a document type:")
    for label, text in SYNTHETIC_TEXTBOOK_PAGES.items():
        doc_type = document_fields.extract_fields(text).document_type
        check(f"{label}: {doc_type}", doc_type == "unknown")

    print("\nUnknown documents keep no guessed fields:")
    worksheet = document_fields.extract_fields(SYNTHETIC_TEXTBOOK_PAGES["kk chemistry worksheet"])
    check("worksheet classified as unknown", worksheet.document_type == "unknown")
    check("... with no 'label: value' junk fields", worksheet.fields == {})
    check("... raw OCR text kept for /find", "Берілгені" in worksheet.raw_text)


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


async def phase4_offline_checks() -> None:
    """PII safety net + screenshot note formatting — no network, no OCR
    (CLAUDE.md §7 Phase 4)."""
    print("\n/note PII safety net (document_fields.has_pii_signals):")
    check(
        "passport sample (title-line keyword) flagged",
        document_fields.has_pii_signals(SAMPLE_PASSPORT_OCR),
    )
    check(
        "garbled passport sample (MRZ only, no readable keyword) flagged",
        document_fields.has_pii_signals(SAMPLE_GARBLED_PASSPORT_OCR),
    )
    check(
        "vehicle registration sample (VIN pattern) flagged",
        document_fields.has_pii_signals(SAMPLE_VEHICLE_OCR),
    )
    check(
        "a passport keyword deep in an unrelated form still flags (safety net "
        "is deliberately more permissive than detect_document_type's title-only rule)",
        document_fields.has_pii_signals(
            "Адрес жапсырмасы\nSome field\nSome field\nSome field\nSome field\nSome field\n"
            "паспорт номер 1111 222222 выдан УФМС"
        ),
    )
    check(
        "an ordinary whiteboard-note screenshot is NOT flagged",
        not document_fields.has_pii_signals(
            "Newton's laws\n1. An object in motion stays in motion.\n"
            "2. F = ma\n3. Every action has an equal and opposite reaction."
        ),
    )

    print("\nScreenshot note labelling (structured-output parsing):")
    valid_json = json.dumps({"title": "Physics notes", "tags": ["physics"]})
    router = LLMRouter([JsonStubProvider(valid_json)])
    note = await media_notes.format_screenshot(router, "raw ocr text", user_message="")
    check("title parsed", note.title == "Physics notes")
    check("tags parsed", note.tags == ["physics"])
    check("the model returns no body — OCR text is never rewritten", not hasattr(note, "markdown"))

    print("\nScreenshot note formatting (invalid JSON is a clean error, not a crash):")
    bad_router = LLMRouter([JsonStubProvider("not json at all")])
    try:
        await media_notes.format_screenshot(bad_router, "raw ocr text", user_message="")
    except media_notes.ScreenshotFormatError:
        check("invalid JSON raises ScreenshotFormatError", True)
    else:
        check("invalid JSON raises ScreenshotFormatError", False)

    print("\nScreenshot note formatting (empty fields also rejected):")
    empty_router = LLMRouter([JsonStubProvider(json.dumps({"title": ""}))])
    try:
        await media_notes.format_screenshot(empty_router, "raw ocr text", user_message="")
    except media_notes.ScreenshotFormatError:
        check("empty title raises ScreenshotFormatError", True)
    else:
        check("empty title raises ScreenshotFormatError", False)


async def note_store_offline_check() -> None:
    """Staging-table save/count roundtrip (CLAUDE.md §7 Phase 4) — no search
    method exists on purpose, see `storage/notes.py`."""
    print("\nNote staging store:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = NoteStore(Path(tmp) / "notes.db")
        await store.connect()
        try:
            record = await store.save(
                telegram_id=1,
                kind="screenshot",
                title="Physics notes",
                content_md="# Newton's laws",
                content_json=json.dumps({"title": "Physics notes"}),
                tags=["physics", "notes"],
            )
            check("save assigns an id", record.id > 0)
            check("tags round-trip", record.tags == ["physics", "notes"])
            check("count reflects the save", await store.count(telegram_id=1) == 1)
            check("count is per-user", await store.count(telegram_id=2) == 0)
        finally:
            await store.close()


class CapturingProvider(LLMProvider):
    """Records every prompt it is sent and answers with fixed Markdown."""

    def __init__(self, reply: str) -> None:
        super().__init__(api_key="stub", model="stub")
        self.name = "capturing-stub"
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, *, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        return self.reply


def language_offline_checks() -> None:
    """Regression for the /note and /download language leaks: command words
    and URLs are Latin script and must never count as "the user wrote English"."""
    from tools.language import reply_language, user_words

    print("\nReply language (command prefix + URLs stripped by one shared helper):")
    cases = [
        ("/download https://youtu.be/dQw4w9WgXcQ", "Kazakh"),
        ("/download@HarnessBot https://youtu.be/dQw4w9WgXcQ", "Kazakh"),
        ("/download www.youtube.com/watch?v=dQw4w9WgXcQ", "Kazakh"),
        ("/summarize https://example.com/post", "Kazakh"),
        ("/note", "Kazakh"),
        ("https://example.com/a_b?x=1", "Kazakh"),
        ("/note осы конспектіні сақта", "Kazakh"),
        ("/note мне нужен конспект", "Kazakh"),  # Russian -> Kazakh, never Russian
        ("/summarize https://example.com please summarize this", "English"),
        ("what is this about? https://youtu.be/x", "English"),
    ]
    for text, expected in cases:
        check(f"{text!r} -> {expected}", reply_language(text) == expected)
    check(
        "user_words removes command and URL only",
        user_words("/download@Bot https://x.y/z hello") == "hello",
    )


async def routing_offline_checks() -> None:
    """Feed real aiogram Updates through the real root router (no network):
    `/download <url>` must reach the download handler, not the Phase 2 URL
    catch-all, and its description prompt must ask for Kazakh."""
    from datetime import datetime

    from aiogram import Bot
    from aiogram.client.session.base import BaseSession
    from aiogram.types import Chat, Message, Update, User

    import bot.handlers.links as links_handlers
    import bot.handlers.media as media_handlers
    from harness import LinkSummary, MediaDownload
    from tools.downloader import VideoInfo
    from tools.urls import classify_url

    chat = Chat(id=1, type="private")

    class RecordingSession(BaseSession):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list = []

        async def make_request(self, bot, method, timeout=None):
            self.calls.append(method)
            if method.__returning__ is User:  # getMe, for "/cmd@BotName" mention checks
                return User(id=42, is_bot=True, first_name="Harness", username="HarnessBot")
            if method.__returning__ is Message:
                return Message(message_id=len(self.calls) + 100, date=datetime.now(), chat=chat)
            return True

        async def stream_content(self, *args, **kwargs):  # pragma: no cover
            yield b""

        async def close(self) -> None:
            pass

    description_llm = CapturingProvider("## Overview\n**Bold** point & <tag>\n- item one")

    class FakeMedia:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, bool]] = []

        async def download(self, url, *, convert_to_mp3, user_message, on_progress=None):
            self.calls.append((url, user_message, convert_to_mp3))
            info = VideoInfo(
                url=url, id="x", title="Some English Title", duration=10.0,
                uploader=None, is_live=False, extractor="Youtube",
            )  # fmt: skip
            description = await media_notes.describe_media(
                LLMRouter([description_llm]), info, user_message=user_message
            )
            return MediaDownload(
                filename="media.mp3" if convert_to_mp3 else "media.mp4",
                data=b"fake media bytes",
                title=info.title,
                description=description,
                is_audio=convert_to_mp3,
                width=1280,
                height=720,
                duration=10.0,
            )

    class FakeHarness:
        def __init__(self) -> None:
            self.summarized: list[tuple[str, str]] = []

        def link_kind(self, url):
            return classify_url(url)

        async def summarize_link(self, *, telegram_id, chat_id, url, user_message, **_):
            self.summarized.append((url, user_message))
            return LinkSummary(url, classify_url(url), "Title", "**Summary**", "article")

        async def start_lesson(self, **_):
            raise AssertionError("tutor must not receive this message")

    class FakeArchive:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.deleted: list[tuple[int, int]] = []

        async def search(self, *, telegram_id, query):
            self.queries.append(query)
            return None

        async def delete(self, *, telegram_id, document_id):
            self.deleted.append((telegram_id, document_id))
            return None  # "not found" path; the store-level checks cover real deletion

    session = RecordingSession()
    fake_bot = Bot(token="42:offline-selfcheck", session=session)
    from harness.knowledge import AnswerStatus, KnowledgeAnswer
    from storage.knowledge import KnowledgeSource

    class FakeKnowledge:
        def __init__(self) -> None:
            self.asked: list[tuple[int, str, str]] = []
            self.next = KnowledgeAnswer(status=AnswerStatus.NOTHING_RELEVANT)

        async def ask(self, *, telegram_id, question, user_message):
            self.asked.append((telegram_id, question, user_message))
            return self.next

    media, harness, archive = FakeMedia(), FakeHarness(), FakeArchive()
    knowledge = FakeKnowledge()
    dp = shared_dispatcher()
    services = {
        "harness": harness,
        "media": media,
        "archive": archive,
        "knowledge": knowledge,
        "flashcards": None,
        "settings": None,
    }

    update_id = 0

    async def send(text: str) -> None:
        nonlocal update_id
        update_id += 1
        message = Message(
            message_id=update_id,
            date=datetime.now(),
            chat=chat,
            from_user=User(id=1, is_bot=False, first_name="Test"),
            text=text,
        )
        session.calls.clear()
        await dp.feed_update(fake_bot, Update(update_id=update_id, message=message), **services)
        pending = media_handlers._background_tasks | links_handlers._background_tasks
        await asyncio.gather(*pending)

    def sent(name: str) -> list:
        return [call for call in session.calls if type(call).__name__ == name]

    print("\nRouting (/download must not be swallowed by the Phase 2 URL catch-all):")
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    await send(f"/download {url}")
    check("'/download <url>' reached the download handler", len(media.calls) == 1)
    check("... and NOT the link summarizer", harness.summarized == [])
    check("the downloaded URL is the one sent", media.calls[0][0] == url)
    check(
        "description prompt asks for Kazakh",
        "(Reply language: Kazakh.)" in description_llm.prompts[-1],
    )
    videos = sent("SendVideo")
    check("mp4 sent via send_video", len(videos) == 1 and not sent("SendDocument"))
    check("... with supports_streaming", videos[0].supports_streaming is True)
    check("... with dimensions", (videos[0].width, videos[0].height) == (1280, 720))
    check("caption uses HTML parse mode", videos[0].parse_mode == "HTML")
    check("caption Markdown converted, not raw", "<b>Bold</b>" in videos[0].caption)
    check(
        "caption HTML-escaped",
        "&amp; &lt;tag&gt;" in videos[0].caption and "##" not in videos[0].caption,
    )

    await send(f"/download@HarnessBot {url} mp3")
    check("'/download@Bot <url> mp3' reached the download handler", len(media.calls) == 2)
    check("... mp3 flag parsed", media.calls[1][2] is True)
    check(
        "... 'mp3' option is not counted as English",
        "(Reply language: Kazakh.)" in description_llm.prompts[-1],
    )
    check("... sent as audio", len(sent("SendAudio")) == 1)

    await send(f"/download {url} please describe it")
    check(
        "English words after the URL still switch to English",
        "(Reply language: English.)" in description_llm.prompts[-1],
    )

    print("\nRouting (other commands with a URL also beat the catch-all):")
    await send("/find https://example.com/x")
    check("'/find <url>' reached the document search", archive.queries == ["https://example.com/x"])
    check("... and NOT the link summarizer", harness.summarized == [])

    await send(f"/summarize {url}")
    check("'/summarize <url>' still summarizes", len(harness.summarized) == 1)

    await send("https://example.com/article")
    check("a plain URL still goes to the Phase 2 summarizer", len(harness.summarized) == 2)
    answers = sent("SendMessage")
    check(
        "summary sent as HTML",
        any(a.parse_mode == "HTML" and "<b>Summary</b>" in a.text for a in answers),
    )
    check("/find and plain URLs triggered no download", len(media.calls) == 3)

    print("\nRouting (Phase 5a /ask):")
    import strings

    await send("/ask фотосинтез қайда өтеді?")
    check("'/ask <q>' reached the knowledge handler", len(knowledge.asked) == 1)
    check(
        "... question passed without the command",
        knowledge.asked[0][1] == "фотосинтез қайда өтеді?",
    )
    check("... scoped to the sender's user id", knowledge.asked[0][0] == 1)
    check(
        "nothing above threshold -> static Kazakh notice, no answer text",
        [m.text for m in sent("SendMessage")] == [strings.KB_NOTHING_RELEVANT],
    )
    knowledge.next = KnowledgeAnswer(status=AnswerStatus.NOT_IN_NOTES)
    await send("/ask Кальвин циклі")
    check(
        "not in notes -> static Kazakh notice",
        [m.text for m in sent("SendMessage")] == [strings.KB_NOT_IN_NOTES],
    )

    page = KnowledgeSource(
        id=7, telegram_id=1, kind="page", title="Биология <9>", origin=None,
        photo_filename="x.enc", created_at="2026-10-04 08:00:00",
    )  # fmt: skip
    knowledge.next = KnowledgeAnswer(
        status=AnswerStatus.ANSWERED,
        answer="Жарық фазасы **тилакоид** мембранасында өтеді [1].",
        sources=[page],
        photo=b"jpeg",
        photo_source=page,
    )
    await send("/ask фотосинтез")
    answer = sent("SendMessage")[0]
    check("answer sent as HTML", answer.parse_mode == "HTML" and "<b>тилакоид</b>" in answer.text)
    check(
        "source reference rendered in code",
        strings.KB_SOURCES_HEADER in answer.text and "2026-10-04" in answer.text,
    )
    check("source title HTML-escaped", "Биология &lt;9&gt;" in answer.text)
    check("original page photo sent after the answer", len(sent("SendPhoto")) == 1)

    await send("/ask")
    check("'/ask' alone -> usage", [m.text for m in sent("SendMessage")] == [strings.KB_ASK_USAGE])
    await send("/page")
    check(
        "'/page' without a photo -> usage",
        [m.text for m in sent("SendMessage")] == [strings.KB_PAGE_USAGE],
    )

    print("\nRouting (/delete):")
    await send("/delete #8")
    check("'/delete #8' reached the archive, scoped to the sender", archive.deleted == [(1, 8)])
    check(
        "unknown id -> not-found notice",
        [m.text for m in sent("SendMessage")] == [strings.DOCUMENT_DELETE_NOT_FOUND.format(id=8)],
    )
    await send("/delete abc")
    check("non-numeric id -> usage, nothing deleted", len(archive.deleted) == 1)

    await fake_bot.session.close()


def formatting_offline_checks() -> None:
    from bot.formatting import TELEGRAM_CAPTION_LIMIT, _balanced, render_chunks

    print("\nTelegram HTML formatter:")
    md = (
        "### Heading\n**bold** and *italic* and `a_b<c>`\n- bullet with [link](https://x.y/a_b)\n"
        "```python\nif a < b and c & d:\n    pass\n```\n> quoted\n***weird***"
    )
    html = render_chunks(md)[0]
    check("heading -> <b>", "<b>Heading</b>" in html and "###" not in html)
    check("bold/italic converted", "<b>bold</b>" in html and "<i>italic</i>" in html)
    check("inline code escaped, not italicised", "<code>a_b&lt;c&gt;</code>" in html)
    check("bullet rendered", "• bullet" in html)
    check("link kept intact", '<a href="https://x.y/a_b">link</a>' in html)
    check(
        "code block escaped",
        '<pre><code class="language-python">if a &lt; b and c &amp; d:' in html,
    )
    check("quote rendered", "<blockquote>quoted</blockquote>" in html)
    check("output is balanced HTML", _balanced(html))

    long_md = "\n".join(f"**Point {i}** — " + "detail & more " * 20 for i in range(200))
    chunks = render_chunks(long_md, header="Title <x>")
    check("long text split into several chunks", len(chunks) > 1)
    check("every chunk within Telegram's limit", all(len(c) <= 4096 for c in chunks))
    check("every chunk balanced on its own", all(_balanced(c) for c in chunks))
    check("header escaped and bold", chunks[0].startswith("<b>Title &lt;x&gt;</b>"))
    caption = render_chunks(long_md, limit=TELEGRAM_CAPTION_LIMIT)
    check("caption chunks within 1024", all(len(c) <= TELEGRAM_CAPTION_LIMIT for c in caption))

    big_code = "```\n" + "x = 1 < 2\n" * 2000 + "```"
    check(
        "an oversized code block is split into balanced <pre> chunks",
        all(len(c) <= 4096 and _balanced(c) for c in render_chunks(big_code)),
    )

    print("\n/note keeps OCR text verbatim:")
    ocr_text = "Pronunciation: [See-ee-oh]\n**not bold** # not a heading\n<b>"
    verbatim = render_chunks(ocr_text, header="📝 Title", verbatim=True)[0]
    check("bracketed text untouched", "[See-ee-oh]" in verbatim)
    check("Markdown-looking OCR text left literal", "**not bold** # not a heading" in verbatim)
    check("but still HTML-escaped", "&lt;b&gt;" in verbatim)


def download_format_offline_checks() -> None:
    """Format selection on a synthetic yt-dlp info dict — no network."""
    from tools.downloader import estimated_size, select_video_format

    print("\nDownload format selection (mp4 first, size-capped):")
    mb = 1024 * 1024

    def fmt(format_id, ext, vcodec, acodec, height, size_mb):
        return {
            "format_id": format_id, "ext": ext, "vcodec": vcodec, "acodec": acodec,
            "height": height, "width": height * 16 // 9 if height else None,
            "filesize": int(size_mb * mb), "url": f"https://example.invalid/{format_id}",
            "protocol": "https",
        }  # fmt: skip

    raw = {
        "_type": "video",
        "id": "x",
        "title": "t",
        "extractor": "youtube",
        "extractor_key": "Youtube",
        "webpage_url": "https://www.youtube.com/watch?v=x",
        "duration": 600,
        "formats": [
            fmt("140", "m4a", "none", "mp4a.40.2", None, 5),
            fmt("251", "webm", "none", "opus", None, 6),
            fmt("137", "mp4", "avc1.640028", "none", 1080, 80),
            fmt("248", "webm", "vp9", "none", 1080, 60),
            fmt("136", "mp4", "avc1.4d401f", "none", 720, 30),
            fmt("247", "webm", "vp9", "none", 720, 25),
            fmt("135", "mp4", "avc1.4d401e", "none", 480, 15),
        ],
    }
    opts = {"quiet": True, "no_warnings": True, "merge_output_format": "mp4"}
    _, selected = select_video_format(raw, opts, 45 * mb)
    ids = [f["format_id"] for f in selected.get("requested_formats") or [selected]]
    check("1080p (85 MB) skipped, 720p H.264 + m4a picked", ids == ["136", "140"])
    check("estimated size under the cap", estimated_size(selected) <= 45 * mb)
    _, unlimited = select_video_format(raw, opts, None)
    check(
        "with no cap the best H.264 wins over VP9",
        [f["format_id"] for f in unlimited["requested_formats"]] == ["137", "140"],
    )


_SHARED_DP = None


def shared_dispatcher():
    """One Dispatcher holding the full root router for every routing check:
    aiogram lets a router be attached to a single parent only. Each check
    passes its own services as feed_update() keyword data."""
    global _SHARED_DP
    if _SHARED_DP is None:
        from aiogram import Dispatcher

        from bot.handlers import build_root_router

        _SHARED_DP = Dispatcher()
        _SHARED_DP.include_router(build_root_router())
    return _SHARED_DP


class HashEmbedder:
    """Deterministic offline stand-in for llm_router.Embedder: hashed
    bag-of-words, L2-normalized. Same words -> high cosine; disjoint -> ~0."""

    dimensions = 64
    model_id = "hash-stub@64"

    def __init__(self) -> None:
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        import hashlib
        import math
        import re

        vector = [0.0] * self.dimensions
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[digest[0] % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def embed_documents(self, texts):
        self.calls += 1
        return [self._vector(text) for text in texts]

    async def embed_query(self, text):
        self.calls += 1
        return self._vector(text)


class CountingProvider(LLMProvider):
    """Counts calls and records everything sent; replies with a fixed string.

    `prompts` holds system prompt + user turn per call, so "text X was never
    sent" checks see the whole request — source text lives in the system
    prompt, not the user turn."""

    def __init__(self, name: str, reply: str) -> None:
        super().__init__(api_key="stub", model="stub")
        self.name = name
        self.reply = reply
        self.calls = 0
        self.prompts: list[str] = []
        self.images_seen = 0

    async def complete(self, *, prompt: str, system=None, images=None, **_kwargs) -> str:
        self.calls += 1
        self.prompts.append(f"{system or ''}\n\n{prompt}")
        self.images_seen += len(images or [])
        return self.reply


async def knowledge_offline_checks() -> None:
    """Phase 5a: isolation, threshold, grounding, PII gate, migration — no network."""
    from harness.documents import DocumentArchive
    from harness.knowledge import AnswerStatus, KnowledgeBase, KnowledgeError
    from providers.base import ImagePart, ProviderError, UnsupportedFeature
    from providers.openai_compatible import GroqProvider
    from storage.knowledge import KnowledgeStore
    from tools import rag

    print("\nRAG chunking:")
    page = "\n".join(f"Line {i}: " + "word " * 30 for i in range(20))
    chunks = rag.chunk_text(page, max_chars=400)
    check("long text split into several chunks", len(chunks) > 1)
    check("every chunk within the size limit", all(len(c) <= 400 for c in chunks))
    check(
        "consecutive chunks overlap by a line",
        chunks[0].splitlines()[-1] == chunks[1].splitlines()[0],
    )
    check("blank text -> no chunks", rag.chunk_text("  \n \n") == [])

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        key = Fernet.generate_key()
        embedder = HashEmbedder()
        store = KnowledgeStore(
            db_path=Path(tmp) / "knowledge.db",
            page_dir=Path(tmp) / "pages",
            encryption_key=key,
            embedding_model=embedder.model_id,
            dimensions=embedder.dimensions,
        )
        await store.connect()
        notes = NoteStore(Path(tmp) / "notes.db")
        await notes.connect()
        documents = DocumentStore(
            db_path=Path(tmp) / "documents.db", scan_dir=Path(tmp) / "scans", encryption_key=key
        )
        await documents.connect()
        try:
            answer_json = json.dumps({"answerable": True, "answer": "Тилакоид [1]", "sources": [1]})
            answer_llm = CountingProvider("answer-stub", answer_json)
            kb = KnowledgeBase(
                store=store,
                embedder=embedder,
                router=LLMRouter([answer_llm]),
                vision_router=None,
                note_store=notes,
                top_k=5,
                min_score=0.5,
            )

            user_a, user_b = 111, 222
            photo = b"\xff\xd8 fake jpeg bytes of user A's notebook page"
            src_a = await store.add_source(
                telegram_id=user_a,
                kind="page",
                title="photosynthesis",
                chunks=["photosynthesis light phase thylakoid membrane chlorophyll"],
                embeddings=await embedder.embed_documents(
                    ["photosynthesis light phase thylakoid membrane chlorophyll"]
                ),
                photo=photo,
            )
            secret_b = "user B private diary entry secret password hint"
            await store.add_source(
                telegram_id=user_b,
                kind="note",
                title="diary",
                chunks=[secret_b, "photosynthesis light phase thylakoid membrane chlorophyll"],
                embeddings=await embedder.embed_documents(
                    [secret_b, "photosynthesis light phase thylakoid membrane chlorophyll"]
                ),
            )

            print("\nKnowledge base: per-user isolation:")
            a_chunks: set[int] = set()
            for query in (secret_b, "photosynthesis thylakoid", "diary password", "anything"):
                hits = await store.search(
                    telegram_id=user_a, embedding=embedder._vector(query), k=50
                )
                a_chunks |= {h.chunk_id for h in hits}
                check(
                    f"user A's search for {query!r} returns only A's chunks",
                    all(h.source.telegram_id == user_a for h in hits),
                )
            check("A never sees B's identical-text chunk or secret", len(a_chunks) == 1)
            b_hits = await store.search(
                telegram_id=user_b, embedding=embedder._vector(secret_b), k=50
            )
            check("B still finds B's own secret", b_hits and b_hits[0].text == secret_b)
            check(
                "B's search never returns A's chunk",
                all(h.source.telegram_id == user_b for h in b_hits),
            )
            check(
                "B cannot load A's page photo by id",
                await store.load_photo(telegram_id=user_b, source_id=src_a.id) is None,
            )
            empty = await kb.ask(telegram_id=333, question="photosynthesis", user_message="")
            check(
                "a user with no notes gets nothing (no fallback to others' data)",
                empty.status is AnswerStatus.NOTHING_RELEVANT and not empty.hits,
            )

            print("\nKnowledge base: threshold and grounding:")
            calls_before = answer_llm.calls
            off_topic = await kb.ask(
                telegram_id=user_a, question="football match winner", user_message=""
            )
            check(
                "off-topic question -> NOTHING_RELEVANT",
                off_topic.status is AnswerStatus.NOTHING_RELEVANT,
            )
            check("... without calling the LLM at all", answer_llm.calls == calls_before)

            result = await kb.ask(
                telegram_id=user_a,
                question="photosynthesis light phase thylakoid",
                user_message="/ask фотосинтез жарық фазасы",
            )
            check("relevant question -> ANSWERED", result.status is AnswerStatus.ANSWERED)
            check("source reference is A's page", [s.id for s in result.sources] == [src_a.id])
            check("original page photo returned, decrypted", result.photo == photo)
            check(
                "page photo is encrypted at rest",
                photo not in (Path(tmp) / "pages" / src_a.photo_filename).read_bytes(),
            )
            prompt = answer_llm.prompts[-1]
            check("Kazakh question -> reply language Kazakh", "(Reply language: Kazakh.)" in prompt)
            await kb.ask(
                telegram_id=user_a,
                question="фотосинтез световая фаза",
                user_message="/ask фотосинтез световая фаза",
            )
            check(
                "Russian question -> Kazakh, never Russian",
                "(Reply language: Kazakh.)" in answer_llm.prompts[-1],
            )
            await kb.ask(
                telegram_id=user_a,
                question="photosynthesis light phase",
                user_message="/ask where does the photosynthesis light phase happen",
            )
            check(
                "English question -> English",
                "(Reply language: English.)" in answer_llm.prompts[-1],
            )

            answer_llm.reply = json.dumps({"answerable": False, "answer": "", "sources": []})
            refused = await kb.ask(
                telegram_id=user_a, question="photosynthesis thylakoid", user_message=""
            )
            check(
                "model finds no answer -> NOT_IN_NOTES (static reply)",
                refused.status is AnswerStatus.NOT_IN_NOTES,
            )

            print("\nKnowledge base: staging-note migration:")
            for i in range(2):
                await notes.save(
                    telegram_id=user_a,
                    kind="screenshot",
                    title=f"note {i}",
                    content_md=f"Ohm law current voltage resistance {i}",
                    content_json="{}",
                    tags=[],
                )
            check("first migration indexes both notes", await kb.migrate_staging_notes() == 2)
            check("second migration is a no-op (idempotent)", await kb.migrate_staging_notes() == 0)
            hits = await store.search(
                telegram_id=user_a, embedding=embedder._vector("Ohm law voltage"), k=5
            )
            check("migrated notes are searchable", hits[0].source.kind == "note")

            print("\nKnowledge base: PII gate before any cloud vision call:")
            import harness.knowledge as knowledge_module

            vision_llm = CountingProvider("vision-stub", "Some handwritten text")
            archive = DocumentArchive(store=documents, tesseract_cmd=None, ocr_lang="eng")
            gated = KnowledgeBase(
                store=store,
                embedder=embedder,
                router=LLMRouter([answer_llm]),
                vision_router=LLMRouter([vision_llm]),
                document_archive=archive,
            )
            gate_text = {"value": SAMPLE_PASSPORT_OCR}

            async def fake_local_ocr(_image, **_kwargs):
                return gate_text["value"]

            real_extract = knowledge_module.ocr.extract_text
            knowledge_module.ocr.extract_text = fake_local_ocr
            import tools.ocr as ocr_module

            ocr_module_extract = ocr_module.extract_text
            ocr_module.extract_text = fake_local_ocr  # DocumentArchive.ingest uses it too
            try:
                gated._tesseract_path = "fake-tesseract"
                archive._tesseract_path = "fake-tesseract"
                capture = await gated.ingest_page(telegram_id=user_a, image_bytes=b"passport photo")
                check(
                    "passport page redirected to the encrypted archive",
                    capture.redirected_to_documents,
                )
                check("... and the vision chain was NEVER called", vision_llm.calls == 0)
                check("... nothing about it reached the index", not capture.sent_to_vision)

                gate_text["value"] = "Биология\nфотосинтез"
                vision_llm.reply = "Биология 9 класс\nСветовая фаза: тилакоиды"
                capture = await gated.ingest_page(
                    telegram_id=user_a, image_bytes=b"notebook photo", title_hint=""
                )
                check(
                    "clean page goes to the vision chain",
                    vision_llm.calls == 1 and vision_llm.images_seen == 1,
                )
                check(
                    "clean page indexed with its photo",
                    capture.source and capture.source.photo_filename,
                )
                check("title taken from the first line", capture.source.title == "Биология 9 класс")

                vision_llm.reply = "Менің ЖСН: 900101300123"  # handwritten IIN Tesseract missed
                capture = await gated.ingest_page(telegram_id=user_a, image_bytes=b"page with iin")
                check(
                    "PII found only by vision -> archive, not index",
                    capture.redirected_to_documents and capture.sent_to_vision,
                )

                gated._tesseract_path = None
                calls = vision_llm.calls
                try:
                    await gated.ingest_page(telegram_id=user_a, image_bytes=b"any page")
                except KnowledgeError:
                    check("no local OCR -> page refused (fails closed)", vision_llm.calls == calls)
                else:
                    check("no local OCR -> page refused (fails closed)", False)
            finally:
                knowledge_module.ocr.extract_text = real_extract
                ocr_module.extract_text = ocr_module_extract

            print("\nKnowledge base: pinned embedding model:")
            await store.close()
            other = KnowledgeStore(
                db_path=Path(tmp) / "knowledge.db",
                page_dir=Path(tmp) / "pages",
                encryption_key=key,
                embedding_model="another-model@64",
                dimensions=64,
            )
            try:
                await other.connect()
            except Exception as exc:
                check("opening with a different model is refused", "not comparable" in str(exc))
            else:
                await other.close()
                check("opening with a different model is refused", False)
        finally:
            await store.close()
            await notes.close()
            await documents.close()

    print("\nProviders (Phase 5a hardening):")
    groq = GroqProvider(api_key="stub", model="openai/gpt-oss-120b", reasoning_effort="low")
    try:
        groq._extract_text(
            {
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"completion_tokens_details": {"reasoning_tokens": 512}},
            }
        )
    except ProviderError as exc:
        check("reasoning ate max_tokens -> explicit error", "max_tokens exhausted" in str(exc))
    try:
        await groq.complete(prompt="x", images=[ImagePart(b"img")])
    except UnsupportedFeature:
        check("text-only model refuses images locally (no request sent)", True)
    else:
        check("text-only model refuses images locally (no request sent)", False)


LIVE_STAGING_NOTES = [
    # (title, verbatim OCR-style text) — typed screenshots, like Phase 4 /note rows.
    (
        "Физика: закон Ома",
        "Закон Ома для участка цепи:\nI = U / R\nСопротивление измеряется в омах (Ом).",
    ),
    (
        "Алгебра: дискриминант",
        "Квадрат теңдеу ax² + bx + c = 0\nD = b² − 4ac\nD < 0 болса, нақты түбір жоқ.",
    ),
]


async def knowledge_live_check(image_paths: list[str], questions: list[str]) -> None:
    """Real pipeline in a throwaway temp dir: local Tesseract PII gate -> real
    vision chain -> real Gemini embeddings -> real answer chain. Prints every
    retrieved chunk with its score, then the answer exactly as /ask would."""
    from config import load_settings
    from harness.documents import DocumentArchive
    from harness.knowledge import AnswerStatus, KnowledgeBase
    from llm_router import build_embedder, build_router, build_vision_router
    from storage.knowledge import KnowledgeStore

    settings = load_settings()
    router = build_router(settings)
    vision = build_vision_router(settings)
    embedder = build_embedder(settings)
    assert embedder is not None, "GEMINI_API_KEY is required for the knowledge base"
    user = 1  # throwaway id in a throwaway store
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = KnowledgeStore(
            db_path=Path(tmp) / "knowledge.db",
            page_dir=Path(tmp) / "pages",
            encryption_key=settings.documents_encryption_key,
            embedding_model=embedder.model_id,
            dimensions=settings.embedding_dim,
        )
        notes = NoteStore(Path(tmp) / "notes.db")
        documents = DocumentStore(
            db_path=Path(tmp) / "documents.db",
            scan_dir=Path(tmp) / "scans",
            encryption_key=settings.documents_encryption_key,
        )
        for resource in (store, notes, documents):
            await resource.connect()
        archive = DocumentArchive(
            store=documents,
            tesseract_cmd=settings.tesseract_cmd,
            ocr_lang=settings.ocr_lang,
            ocr_tessdata_dir=settings.ocr_tessdata_dir,
        )
        kb = KnowledgeBase(
            store=store,
            embedder=embedder,
            router=router,
            vision_router=vision,
            note_store=notes,
            document_archive=archive,
            tesseract_cmd=settings.tesseract_cmd,
            ocr_lang=settings.ocr_lang,
            ocr_tessdata_dir=settings.ocr_tessdata_dir,
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
        )
        print("\n" + kb.describe())
        try:
            print("\n== Ingestion ==")
            for path in image_paths:
                capture = await kb.ingest_page(
                    telegram_id=user, image_bytes=Path(path).read_bytes()
                )
                if capture.redirected_to_documents:
                    print(
                        f"\n[{Path(path).name}] PII gate fired -> document archive "
                        f"(type={capture.document.document_type}, "
                        f"sent_to_vision={capture.sent_to_vision})"
                    )
                    continue
                print(
                    f"\n[{Path(path).name}] indexed as {capture.source.title!r} via "
                    f"{capture.transcribed_by}, {capture.chunk_count} chunk(s):"
                )
                print("    " + capture.transcription.replace("\n", "\n    "))
            for title, text in LIVE_STAGING_NOTES:
                await notes.save(
                    telegram_id=user,
                    kind="screenshot",
                    title=title,
                    content_md=text,
                    content_json="{}",
                    tags=[],
                )
            print(f"\nmigrated {await kb.migrate_staging_notes()} staging note(s)")

            for question in questions:
                result = await kb.ask(
                    telegram_id=user, question=question, user_message=f"/ask {question}"
                )
                print(f"\n== /ask {question}")
                for hit in result.hits[:3]:
                    mark = "PASS" if hit.score >= kb.min_score else "below"
                    text = hit.text.replace("\n", " / ")
                    print(
                        f"    {hit.score:.3f} [{mark}] "
                        f"{hit.source.kind}:{hit.source.title!r}: {text[:90]}"
                    )
                print(f"  status: {result.status.value}")
                if result.status is AnswerStatus.ANSWERED:
                    print("  answer: " + result.answer.replace("\n", "\n          "))
                    print(
                        "  sources: " + "; ".join(f"{s.kind} {s.title!r}" for s in result.sources)
                    )
                    print(f"  page photo returned: {len(result.photo or b'')} bytes")
        finally:
            for resource in (store, notes, documents):
                await resource.close()
            await router.aclose()
            await embedder.aclose()
            if vision is not None:
                await vision.aclose()


async def archive_fallback_and_delete_checks() -> None:
    """/find raw-text fallback for field-less documents, and /delete scoping."""
    print("\nDocument archive: /find falls back to raw OCR text:")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = DocumentStore(
            db_path=Path(tmp) / "documents.db",
            scan_dir=Path(tmp) / "scans",
            encryption_key=Fernet.generate_key(),
        )
        await store.connect()
        try:
            text = SYNTHETIC_TEXTBOOK_PAGES["kk chemistry worksheet"]
            unknown = await store.save(
                telegram_id=1,
                document_type="unknown",
                fields={},
                raw_text=text,
                image_bytes=b"worksheet scan",
            )
            other_users = await store.save(
                telegram_id=2,
                document_type="unknown",
                fields={},
                raw_text=text,
                image_bytes=b"someone else's scan",
            )
            match = await store.search(telegram_id=1, query="молярлық көлем есебі")
            check(
                "field-less document found via its OCR text",
                match and match.record.id == unknown.id,
            )
            check(
                "matched lines returned for display",
                match is not None and any("көлемі" in line for line in match.matched_lines),
            )
            check(
                "fallback never returns another user's document",
                match is not None and match.record.telegram_id == 1,
            )
            weak = await store.search(telegram_id=1, query="футбол чемпионаты нәтижесі көлем")
            check("one shared word out of four is not a match", weak is None)

            print("\nDocument archive: /delete:")
            check(
                "user 1 cannot delete user 2's document",
                await store.delete(telegram_id=1, document_id=other_users.id) is None,
            )
            check("... which is still there", other_users.scan_path.exists())
            deleted = await store.delete(telegram_id=1, document_id=unknown.id)
            check("owner deletes their document", deleted is not None and deleted.id == unknown.id)
            check("... encrypted scan file removed", not unknown.scan_path.exists())
            check(
                "... and it is no longer findable",
                await store.search(telegram_id=1, query="молярлық көлем есебі") is None,
            )
            check(
                "deleting again -> not found",
                await store.delete(telegram_id=1, document_id=unknown.id) is None,
            )
        finally:
            await store.close()


async def ask_outcome_logging_checks() -> None:
    """Every /ask logs best score and whether a refusal came from the
    threshold or from the model's "unanswerable" flag."""
    import logging

    from harness.knowledge import KnowledgeBase
    from storage.knowledge import KnowledgeStore

    print("\n/ask outcome logging:")
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    kb_logger = logging.getLogger("harness.knowledge")
    handler = Capture()
    kb_logger.addHandler(handler)
    previous_level = kb_logger.level
    kb_logger.setLevel(logging.INFO)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        embedder = HashEmbedder()
        store = KnowledgeStore(
            db_path=Path(tmp) / "k.db",
            page_dir=Path(tmp) / "p",
            encryption_key=Fernet.generate_key(),
            embedding_model=embedder.model_id,
            dimensions=embedder.dimensions,
        )
        await store.connect()
        try:
            chunk = "ohm law current voltage resistance"
            await store.add_source(
                telegram_id=1,
                kind="note",
                title="ohm",
                chunks=[chunk],
                embeddings=await embedder.embed_documents([chunk]),
            )
            llm = CountingProvider(
                "stub", json.dumps({"answerable": True, "answer": "I = U/R [1]", "sources": [1]})
            )
            kb = KnowledgeBase(
                store=store, embedder=embedder, router=LLMRouter([llm]), vision_router=None,
                min_score=0.5,
            )  # fmt: skip

            def last_outcome() -> str:
                lines = [r.getMessage() for r in records if "kb /ask" in r.getMessage()]
                return lines[-1] if lines else ""

            await kb.ask(telegram_id=1, question="football", user_message="")
            line = last_outcome()
            check(f"threshold refusal logged: {line!r}", "outcome=refused:threshold" in line)
            check("... with the best score", "best=" in line and "best=none" not in line)
            llm.reply = json.dumps({"answerable": False, "answer": "", "sources": []})
            await kb.ask(telegram_id=1, question="ohm law voltage", user_message="")
            line = last_outcome()
            check(f"model refusal logged: {line!r}", "outcome=refused:model" in line)
            llm.reply = json.dumps({"answerable": True, "answer": "I = U/R [1]", "sources": [1]})
            await kb.ask(telegram_id=1, question="ohm law voltage", user_message="")
            line = last_outcome()
            check(f"answer logged: {line!r}", "outcome=answered" in line and "cited=[1]" in line)
            await kb.ask(telegram_id=99, question="ohm", user_message="")
            check("empty index logs best=none", "best=none" in last_outcome())
            check(
                "question text never logged",
                not any("ohm law voltage" in r.getMessage() for r in records),
            )
        finally:
            await store.close()
            kb_logger.removeHandler(handler)
            kb_logger.setLevel(previous_level)


# -- Phase 5b: flashcards, SM-2, daily quiz ---------------------------------------


async def _first_turn(flashcard_tools) -> str:
    """The user turn of a Kazakh-target generation call (for prompt checks)."""
    stub = CountingProvider(
        "first-turn", json.dumps({"cards": [{"front": "Сұрақ?", "back": "Жауап"}]})
    )
    await flashcard_tools.generate_cards(LLMRouter([stub]), "text", user_message="")
    return stub.prompts[0]


def tiny_text_pdf(lines: list[str]) -> bytes:
    """A minimal valid one-page PDF with a real text layer (ASCII, Helvetica),
    built by hand so the checks need no PDF-writing dependency."""
    escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    ops = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(f"({line}) '" for line in escaped) + " ET"
    stream = ops.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def blank_pdf(pages: int) -> bytes:
    """A 'scanned-like' PDF: pages with no text layer at all."""
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def sm2_and_clock_checks() -> None:
    from datetime import datetime

    from tools import quiz_clock, sm2

    print("\nSM-2:")
    state = sm2.CardState()
    s1 = sm2.review(state, 4)
    check("1st success -> interval 1, reps 1", (s1.interval_days, s1.repetitions) == (1, 1))
    check("grade 4 keeps EF at 2.5", abs(s1.ease - 2.5) < 1e-9)
    s2 = sm2.review(s1, 4)
    check("2nd success -> interval 6", s2.interval_days == 6)
    s3 = sm2.review(s2, 4)
    check("3rd success -> round(6 * EF 2.5) = 15", s3.interval_days == 15)
    s3_easy = sm2.review(s2, 5)
    check("grade 5 raises EF by 0.1", abs(s3_easy.ease - 2.6) < 1e-9)
    check("... interval still uses the EF from before the review (15)", s3_easy.interval_days == 15)
    s3_hard = sm2.review(s2, 3)
    check("grade 3 lowers EF by 0.14", abs(s3_hard.ease - 2.36) < 1e-9)
    lapse = sm2.review(s3, 1)
    check("lapse -> reps 0, interval 1 day", (lapse.repetitions, lapse.interval_days) == (0, 1))
    check("lapse counted", lapse.lapses == 1)
    check("lapse (grade 1) lowers EF by 0.54", abs(lapse.ease - (2.5 - 0.54)) < 1e-9)
    relearn = sm2.review(sm2.review(lapse, 4), 4)
    check("relearning restarts 1 -> 6", relearn.interval_days == 6)
    floor = state
    for _ in range(10):
        floor = sm2.review(floor, 0)
    check("EF never drops below 1.3", abs(floor.ease - 1.3) < 1e-9 and floor.lapses == 10)
    try:
        sm2.review(state, 6)
    except ValueError:
        check("grade outside 0..5 rejected", True)
    else:
        check("grade outside 0..5 rejected", False)
    check(
        "due_after adds the interval",
        sm2.due_after(datetime(2026, 10, 4, 15, 0), s2) == datetime(2026, 10, 10, 15, 0),
    )

    print("\nDaily quiz clock (default Almaty UTC+5):")
    check("'7:05' -> '07:05'", quiz_clock.parse_quiz_time("7:05") == "07:05")
    check("'24:00' rejected", quiz_clock.parse_quiz_time("24:00") is None)
    check("'+5' -> 300", quiz_clock.parse_utc_offset("+5") == 300)
    check("'UTC+05:30' -> 330", quiz_clock.parse_utc_offset("UTC+05:30") == 330)
    check("'-3' -> -180", quiz_clock.parse_utc_offset("-3") == -180)
    check("'+15' rejected", quiz_clock.parse_utc_offset("+15") is None)

    def due(now, last=None, time="20:00", enabled=True):
        return quiz_clock.daily_due(
            now_utc=now, quiz_time=time, utc_offset_minutes=300, enabled=enabled,
            last_daily_date=last,
        )  # fmt: skip

    check("14:59 UTC (19:59 Almaty) -> not yet", not due(datetime(2026, 10, 4, 14, 59)))
    check("15:00 UTC (20:00 Almaty) -> due", due(datetime(2026, 10, 4, 15, 0)))
    check("already sent today -> not due", not due(datetime(2026, 10, 4, 16, 0), "2026-10-04"))
    check(
        "bot back at 23:30 the same day -> still sent once (gentle catch-up)",
        due(datetime(2026, 10, 4, 18, 30), "2026-10-03"),
    )
    check(
        "back after midnight, before quiz time -> waits (no 02:00 quiz)",
        not due(datetime(2026, 10, 4, 21, 0), "2026-10-03"),
    )
    check(
        "local date, not UTC date: 20:30 UTC = 01:30 next day in Almaty -> waits for 20:00",
        not due(datetime(2026, 10, 4, 20, 30), "2026-10-04"),
    )
    check(
        "... and fires at the next local 20:00",
        due(datetime(2026, 10, 5, 15, 0), "2026-10-04"),
    )
    check("disabled -> never", not due(datetime(2026, 10, 4, 15, 0), enabled=False))


async def flashcard_service_checks() -> None:
    from datetime import UTC, datetime, timedelta

    from harness.flashcards import FlashcardError, FlashcardService
    from storage.flashcards import FlashcardStore, NewCard
    from storage.knowledge import KnowledgeStore
    from tools import flashcards as flashcard_tools

    def cards_json(prefix: str, n: int) -> str:
        return json.dumps(
            {
                "cards": [
                    {"front": f"{prefix} сұрақ {i}?", "back": f"{prefix} жауап {i}"}
                    for i in range(n)
                ]
            }
        )

    print("\nFlashcard extraction parsing:")
    drafts = flashcard_tools.parse_cards(
        json.dumps({"cards": [
            {"front": "Q1?", "back": "A1"}, {"front": " q1? ", "back": "dup"},
            {"front": "", "back": "x"}, {"front": "Same", "back": "same"},
        ] + [{"front": f"F{i}?", "back": f"B{i}"} for i in range(20)]})
    )  # fmt: skip
    check(
        "duplicates / empty / front==back dropped",
        drafts[0].front == "Q1?" and drafts[1].front == "F0?",
    )
    check("capped at 15 cards per call", len(drafts) == 15)
    try:
        flashcard_tools.parse_cards("not json")
    except flashcard_tools.FlashcardFormatError:
        check("invalid JSON -> FlashcardFormatError", True)

    print("\nFlashcard language enforced in code (§8):")
    english = json.dumps({"cards": [{"front": "What is ATP?", "back": "Energy currency"}]})
    russian = json.dumps({"cards": [{"front": "Что такое АТФ?", "back": "Источник энергии"}]})
    kazakh = json.dumps({"cards": [{"front": "АТФ дегеніміз не?", "back": "Энергия көзі"}]})

    class SequenceProvider(LLMProvider):
        def __init__(self, replies: list[str]) -> None:
            super().__init__(api_key="stub", model="stub")
            self.name = "sequence"
            self.replies = replies
            self.turns: list[str] = []

        async def complete(self, *, prompt: str, **_kwargs) -> str:
            self.turns.append(prompt)
            return self.replies[min(len(self.turns), len(self.replies)) - 1]

    seq = SequenceProvider([english, kazakh])
    cards = await flashcard_tools.generate_cards(LLMRouter([seq]), "text", user_message="/cards")
    check("English reply for a Kazakh request -> one corrective retry", len(seq.turns) == 2)
    check("... retry asks again explicitly", "NOT in Kazakh" in seq.turns[1])
    check("... and the Kazakh cards are kept", cards[0].front == "АТФ дегеніміз не?")
    for label, reply in (("English", english), ("Russian", russian)):
        try:
            await flashcard_tools.generate_cards(
                LLMRouter([SequenceProvider([reply])]), "text", user_message="/cards"
            )
        except flashcard_tools.FlashcardLanguageError:
            check(f"always {label} for a Kazakh request -> rejected, never stored", True)
        else:
            check(f"always {label} for a Kazakh request -> rejected, never stored", False)
    seq = SequenceProvider([english])
    await flashcard_tools.generate_cards(
        LLMRouter([seq]), "text", user_message="make these in English please"
    )
    check("English request + English cards -> accepted first time", len(seq.turns) == 1)
    check(
        "first request already names the language natively",
        "қазақ тілінде" in (await _first_turn(flashcard_tools)),
    )
    check(
        "card budget sized for reasoning models (>= 4096 tokens)",
        flashcard_tools.FLASHCARD_MAX_TOKENS >= 4096,
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        key = Fernet.generate_key()
        embedder = HashEmbedder()
        kstore = KnowledgeStore(
            db_path=Path(tmp) / "k.db", page_dir=Path(tmp) / "p", encryption_key=key,
            embedding_model=embedder.model_id, dimensions=embedder.dimensions,
        )  # fmt: skip
        await kstore.connect()
        store = FlashcardStore(Path(tmp) / "cards.db")
        await store.connect()
        llm = CountingProvider("card-stub", cards_json("A", 6))
        service = FlashcardService(
            store=store, router=LLMRouter([llm]), knowledge_store=kstore,
            default_daily_cap=3,
        )  # fmt: skip
        user_a, user_b = 111, 222
        try:
            print("\nFlashcards: PII gate before the LLM:")
            for title, text in (
                ("biology", "Mitosis has four phases\nprophase metaphase anaphase telophase"),
                ("passport photo", SAMPLE_PASSPORT_OCR),
            ):
                await kstore.add_source(
                    telegram_id=user_a, kind="page", title=title, chunks=[text],
                    embeddings=await embedder.embed_documents([text]),
                )  # fmt: skip
            report = await service.make_from_knowledge(
                telegram_id=user_a, chat_id=user_a, user_message="/cards make"
            )
            check("clean source sent to the LLM once", llm.calls == 1)
            check(
                "PII source NEVER sent (no prompt contains the IIN)",
                not any("900101300123" in p for p in llm.prompts),
            )
            check("... and reported as withheld", report.segments_skipped_pii == 1)
            check("6 cards created from the clean source", len(report.cards) == 6)
            check(
                "'/cards make' -> Kazakh cards ('make' is not English text)",
                "(Reply language: Kazakh.)" in llm.prompts[-1],
            )
            try:
                await service.make_from_knowledge(
                    telegram_id=user_a, chat_id=user_a, user_message="/cards make"
                )
            except FlashcardError as exc:
                check(
                    "second run: every source already carded",
                    str(exc) == strings.CARDS_ALL_SOURCES_DONE,
                )

            print("\nFlashcards: PDF sources:")
            llm.reply = cards_json("PDF", 5)
            pdf = tiny_text_pdf(
                [f"Photosynthesis fact number {i}: chlorophyll absorbs light." for i in range(12)]
            )
            report = await service.make_from_pdf(
                telegram_id=user_a, chat_id=user_a, data=pdf, filename="bio.pdf", user_message=""
            )
            check("text PDF -> cards", len(report.cards) == 5)
            check("PDF text reached the prompt", "chlorophyll absorbs light" in llm.prompts[-1])
            for label, data, expected in (
                (
                    "scanned PDF -> Kazakh /page hint, no LLM call",
                    blank_pdf(3),
                    strings.CARDS_PDF_SCANNED,
                ),
                ("same PDF again -> already done", pdf, strings.CARDS_PDF_ALREADY_DONE),
            ):
                calls = llm.calls
                try:
                    await service.make_from_pdf(
                        telegram_id=user_a,
                        chat_id=user_a,
                        data=data,
                        filename="x.pdf",
                        user_message="",
                    )
                except FlashcardError as exc:
                    check(label, str(exc) == expected and llm.calls == calls)
                else:
                    check(label, False)
            try:
                await service.make_from_pdf(
                    telegram_id=user_a,
                    chat_id=user_a,
                    data=b"not a pdf",
                    filename="x.pdf",
                    user_message="",
                )
            except FlashcardError:
                check("corrupt PDF -> clean error", True)
            llm.reply = cards_json("PDF", 5)
            pii_pdf = tiny_text_pdf(
                [
                    "REPUBLIC OF KAZAKHSTAN - PASSPORT",
                    "IIN: 900101300123",
                    "Surname: TESTOV  Given names: TESTBEK",
                    "Date of birth: 01.01.1990  Date of expiry: 01.01.2030",
                ]
            )
            calls = llm.calls
            report = await service.make_from_pdf(
                telegram_id=user_a, chat_id=user_a, data=pii_pdf, filename="id.pdf", user_message=""
            )
            check("PDF with passport text: NOT sent to the LLM", llm.calls == calls)
            check(
                "... withheld and reported", report.segments_skipped_pii == 1 and not report.cards
            )

            print("\nFlashcards: per-user isolation:")
            card_a = report_card = await store.next_due_card(
                telegram_id=user_a, now=datetime(2100, 1, 1)
            )
            assert report_card is not None
            check(
                "B cannot see A's card",
                await store.get_card(telegram_id=user_b, card_id=card_a.id) is None,
            )
            check(
                "B has nothing due",
                await store.next_due_card(telegram_id=user_b, now=datetime(2100, 1, 1)) is None,
            )
            prompt_a = await service.start_session(
                telegram_id=user_a, chat_id=user_a, kind="manual"
            )
            assert prompt_a is not None
            check(
                "B cannot answer A's question",
                await service.answer_choice(telegram_id=user_b, item_id=prompt_a.item.id, choice=0)
                is None,
            )
            check(
                "B cannot rate A's question",
                await service.rate(telegram_id=user_b, item_id=prompt_a.item.id, grade=4) is None,
            )
            check(
                "B cannot reveal A's question",
                await service.reveal(telegram_id=user_b, item_id=prompt_a.item.id) is None,
            )
            check(
                "B cannot delete A's card",
                await service.delete_card(telegram_id=user_b, card_id=card_a.id) is None,
            )
            check(
                "... A's card still there",
                await store.get_card(telegram_id=user_a, card_id=card_a.id) is not None,
            )
            check(
                "B cannot write a review onto A's card",
                not await store.save_review(
                    telegram_id=user_b, card_id=card_a.id, state=card_a.state,
                    due_at=datetime(2100, 1, 1), grade=5, mode="choice", now=datetime(2026, 1, 1),
                ),
            )  # fmt: skip
            stats_b, _ = await service.stats(telegram_id=user_b, chat_id=user_b)
            check("B's stats don't count A's cards", stats_b.total == 0)

            print("\nQuiz session (multiple choice, SM-2, cap):")
            check(
                "enough cards -> multiple choice",
                prompt_a.item.mode == "choice" and len(prompt_a.item.choices) == 4,
            )
            correct = prompt_a.item.correct_index
            assert correct is not None
            check(
                "correct answer among the choices",
                prompt_a.item.choices[correct] == prompt_a.card.back,
            )
            check(
                "distractors are A's own other answers",
                all(c.startswith(("A жауап", "PDF жауап")) for c in prompt_a.item.choices),
            )
            now = datetime.now(UTC).replace(tzinfo=None)
            result = await service.answer_choice(
                telegram_id=user_a, item_id=prompt_a.item.id, choice=correct, now=now
            )
            assert result is not None
            check("correct choice -> grade 4", result.correct is True and result.grade == 4)
            check("new card -> next review in 1 day", result.next_due_at - now == timedelta(days=1))
            try:
                await service.answer_choice(
                    telegram_id=user_a, item_id=prompt_a.item.id, choice=correct
                )
            except FlashcardError as exc:
                check("double tap can't grade twice", str(exc) == strings.QUIZ_ALREADY_ANSWERED)
            second = result.next_prompt
            assert second is not None
            check("next card is a different card", second.card.id != prompt_a.card.id)
            wrong = next(i for i in range(4) if i != second.item.correct_index)
            result = await service.answer_choice(
                telegram_id=user_a, item_id=second.item.id, choice=wrong, now=now
            )
            assert result is not None
            check("wrong choice -> grade 1, lapse", result.correct is False and result.grade == 1)
            third = result.next_prompt
            assert third is not None
            result = await service.rate(telegram_id=user_a, item_id=third.item.id, grade=5, now=now)
            assert result is not None
            check(
                "daily cap 3 reached -> session over",
                result.session_done and result.next_prompt is None,
            )
            check("session score counted", (result.session_asked, result.session_good) == (3, 2))

            print("\nReveal + self-rating mode:")
            long_back = "A long explanatory answer " * 5
            await store.add_cards(
                telegram_id=user_b, source_key="kb:x", source_title="t",
                cards=[NewCard("Long question?", long_back, "long question?")], now=now,
            )  # fmt: skip
            prompt_b = await service.start_session(
                telegram_id=user_b, chat_id=user_b, kind="manual"
            )
            assert prompt_b is not None
            check("long answer / few cards -> reveal mode", prompt_b.item.mode == "reveal")
            revealed = await service.reveal(telegram_id=user_b, item_id=prompt_b.item.id)
            check(
                "reveal returns the answer", revealed is not None and revealed[1].back == long_back
            )
            check(
                "rating outside Again/Hard/Good/Easy rejected",
                await service.rate(telegram_id=user_b, item_id=prompt_b.item.id, grade=2) is None,
            )
            result = await service.rate(
                telegram_id=user_b, item_id=prompt_b.item.id, grade=1, now=now
            )
            assert result is not None
            check("'Again' -> lapse, due tomorrow", result.next_due_at - now == timedelta(days=1))

            print("\nDaily quiz scheduling (from the DB, gentle catch-up):")
            # Make many of A's cards due: a backlog after "downtime".
            await store.db.execute(
                "UPDATE cards SET due_at = '2026-01-01 00:00:00' WHERE telegram_id = ?", (user_a,)
            )
            await store.db.commit()
            base = datetime(2026, 10, 4)  # Almaty quiz time 20:00 = 15:00 UTC
            started = await service.due_daily_sessions(base.replace(hour=14, minute=59))
            check("before quiz time: nothing sent", not started)
            started = await service.due_daily_sessions(base.replace(hour=15, minute=0))
            users = {s.telegram_id for s, _ in started}
            check("at quiz time: one session per user", users == {user_a, user_b})
            check(
                "A's session starts with one card",
                any(s.telegram_id == user_a and p is not None for s, p in started),
            )
            check(
                "next tick same day: nothing more",
                not await service.due_daily_sessions(base.replace(hour=15, minute=1)),
            )
            prompt = next(p for s, p in started if s.telegram_id == user_a)
            asked = 0
            while prompt is not None:
                asked += 1
                result = await service.rate(
                    telegram_id=user_a,
                    item_id=prompt.item.id,
                    grade=4,
                    now=base.replace(hour=15, minute=2),
                )
                prompt = result.next_prompt if result else None
            check(f"backlog of {12} due cards -> only the daily cap (3) asked", asked == 3)
            later = base + timedelta(days=3, hours=16)  # down 3 days, back at 21:00 Almaty
            started = await service.due_daily_sessions(later)
            check(
                "after 3 days down: exactly one session per user, not three",
                len([s for s, _ in started if s.telegram_id == user_a]) == 1,
            )

            print("\nRestart survival:")
            pending = next(p for s, p in started if s.telegram_id == user_a)
            assert pending is not None
            await store.close()
            store2 = FlashcardStore(Path(tmp) / "cards.db")
            await store2.connect()
            service2 = FlashcardService(
                store=store2, router=LLMRouter([llm]), knowledge_store=kstore, default_daily_cap=3
            )
            check(
                "after restart: no re-send of today's quiz",
                not await service2.due_daily_sessions(later + timedelta(minutes=5)),
            )
            result = await service2.rate(
                telegram_id=user_a, item_id=pending.item.id, grade=4, now=later
            )
            check("after restart: the button sent before it still works", result is not None)
            deleted = await service2.delete_card(telegram_id=user_a, card_id=card_a.id)
            check("owner can delete their card", deleted is not None)
            check(
                "... and it's gone",
                await store2.get_card(telegram_id=user_a, card_id=card_a.id) is None,
            )
            await store2.close()
        finally:
            await store.close()
            await kstore.close()

    print("\nGroq structured-output flake is retried, not dropped:")
    import httpx

    from providers.base import ProviderUnavailable
    from providers.openai_compatible import GroqProvider

    groq = GroqProvider(api_key="stub", model="openai/gpt-oss-120b")
    flake = httpx.Response(
        400,
        text='{"error":{"code":"json_validate_failed",'
        '"message":"Generated JSON does not match the expected schema."}}',
    )
    try:
        groq._raise_for_status(flake)
    except ProviderUnavailable:
        check("json_validate_failed 400 -> retryable (ProviderUnavailable)", True)
    else:
        check("json_validate_failed 400 -> retryable (ProviderUnavailable)", False)


async def flashcard_routing_checks() -> None:
    """Commands + inline-button callbacks through the real router and the real
    FlashcardService (temp store, stub LLM) — no network."""
    from datetime import UTC, datetime

    from aiogram import Bot
    from aiogram.client.session.base import BaseSession
    from aiogram.types import CallbackQuery, Chat, Message, Update, User

    from harness.flashcards import FlashcardService
    from storage.flashcards import FlashcardStore, NewCard

    print("\nFlashcard commands and buttons (Telegram routing):")

    class Session(BaseSession):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list = []

        async def make_request(self, bot, method, timeout=None):
            self.calls.append(method)
            if method.__returning__ is User:
                return User(id=42, is_bot=True, first_name="Harness", username="HarnessBot")
            if method.__returning__ is Message or "Message" in str(method.__returning__):
                return Message(
                    message_id=len(self.calls) + 500,
                    date=datetime.now(),
                    chat=Chat(id=1, type="private"),
                    text="x",
                )
            return True

        async def stream_content(self, *args, **kwargs):  # pragma: no cover
            yield b""

        async def close(self) -> None:
            pass

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = FlashcardStore(Path(tmp) / "cards.db")
        await store.connect()
        service = FlashcardService(
            store=store, router=LLMRouter([CountingProvider("x", "{}")]), default_daily_cap=10
        )
        now = datetime.now(UTC).replace(tzinfo=None)
        cards = await store.add_cards(
            telegram_id=1, source_key="kb:1", source_title="Биология",
            cards=[NewCard(f"Сұрақ {i}?", f"Жауап {i}", f"сұрақ {i}?") for i in range(5)], now=now,
        )  # fmt: skip
        try:
            session = Session()
            bot = Bot(token="42:offline-selfcheck", session=session)
            # Full root router (real handler order — e.g. the tutor's catch-all
            # must not swallow these), with only the flashcard service wired.
            dp = shared_dispatcher()
            services = {
                "flashcards": service,
                "harness": None,
                "media": None,
                "archive": None,
                "knowledge": None,
                "settings": None,
            }
            update_id = 0

            async def feed(update_kwargs):
                nonlocal update_id
                update_id += 1
                session.calls.clear()
                await dp.feed_update(bot, Update(update_id=update_id, **update_kwargs), **services)

            def message(text, user=1):
                return {"message": Message(
                    message_id=update_id + 1, date=datetime.now(),
                    chat=Chat(id=user, type="private"),
                    from_user=User(id=user, is_bot=False, first_name="T"), text=text,
                )}  # fmt: skip

            def press(data, user=1):
                return {"callback_query": CallbackQuery(
                    id=str(update_id), from_user=User(id=user, is_bot=False, first_name="T"),
                    chat_instance="ci", data=data,
                    message=Message(
                        message_id=900, date=datetime.now(),
                        chat=Chat(id=user, type="private"), text="q",
                    ),
                )}  # fmt: skip

            def sent(name):
                return [c for c in session.calls if type(c).__name__ == name]

            await feed(message("/cards"))
            check("/cards shows stats (Kazakh)", "Карточкалар: 5" in sent("SendMessage")[0].text)
            await feed(message("/quiz"))
            quiz = sent("SendMessage")
            check(
                "/quiz sends a card with inline buttons", quiz and quiz[0].reply_markup is not None
            )
            buttons = [b for row in quiz[0].reply_markup.inline_keyboard for b in row]
            check(
                "multiple choice: 4 answer buttons",
                len(buttons) == 4 and all(b.callback_data.startswith("fc:a:") for b in buttons),
            )
            item_id = int(buttons[0].callback_data.split(":")[2])
            await feed(press(buttons[0].callback_data, user=2))
            alerts = sent("AnswerCallbackQuery")
            check(
                "another user pressing A's button -> alert, nothing graded",
                alerts
                and alerts[0].show_alert
                and (await store.get_item(telegram_id=1, item_id=item_id)).answered is False,
            )
            await feed(press(buttons[0].callback_data))
            check(
                "owner's press grades it and edits the message",
                sent("EditMessageText")
                and (await store.get_item(telegram_id=1, item_id=item_id)).answered,
            )
            check("... and the next card follows", any(m.reply_markup for m in sent("SendMessage")))
            await feed(press(buttons[0].callback_data))
            check(
                "pressing again -> 'already answered' alert",
                sent("AnswerCallbackQuery")[0].text == strings.QUIZ_ALREADY_ANSWERED,
            )

            await feed(message("/quiztime 07:30 +6"))
            settings = await store.get_settings(telegram_id=1)
            check(
                "/quiztime 07:30 +6 saved",
                settings and (settings.quiz_time, settings.utc_offset_minutes) == ("07:30", 360),
            )
            await feed(message("/quiztime cap 4"))
            check("/quiztime cap 4 saved", (await store.get_settings(telegram_id=1)).daily_cap == 4)
            await feed(message("/quiztime 25:00"))
            check("bad time -> Kazakh usage", sent("SendMessage")[0].text == strings.QUIZTIME_BAD)
            await feed(message("/quiztime off"))
            check(
                "/quiztime off disables", (await store.get_settings(telegram_id=1)).enabled is False
            )

            target = cards[-1]
            await feed(message(f"/delcard {target.id}", user=2))
            check(
                "/delcard of another user's card -> not found",
                sent("SendMessage")[0].text == strings.DELCARD_NOT_FOUND.format(id=target.id),
            )
            await feed(message(f"/delcard {target.id}"))
            confirm = sent("SendMessage")[0]
            check(
                "/delcard asks for confirmation first",
                "#" in confirm.text and confirm.reply_markup is not None,
            )
            check(
                "... nothing deleted yet",
                await store.get_card(telegram_id=1, card_id=target.id) is not None,
            )
            await feed(press(f"fc:d:{target.id}:y", user=2))
            check(
                "another user pressing 'Yes' deletes nothing",
                await store.get_card(telegram_id=1, card_id=target.id) is not None,
            )
            await feed(press(f"fc:d:{target.id}:n"))
            check(
                "'No' keeps the card",
                await store.get_card(telegram_id=1, card_id=target.id) is not None,
            )
            await feed(press(f"fc:d:{target.id}:y"))
            check(
                "'Yes' deletes it", await store.get_card(telegram_id=1, card_id=target.id) is None
            )
        finally:
            await store.close()
            if "bot" in locals():
                await bot.session.close()


async def quiz_mode_checks() -> None:
    """Phase 5b.1: why a card is multiple choice or show-answer, short answers +
    generated wrong options, migration, PII on generated text, truncation."""
    import logging
    import sqlite3
    from datetime import UTC, datetime

    import httpx

    from harness.flashcards import FlashcardService
    from providers.base import ProviderError
    from providers.gemini import GeminiProvider
    from providers.openai_compatible import GroqProvider
    from storage.flashcards import FlashcardStore, NewCard
    from tools import flashcards as flashcard_tools
    from tools import quiz_mode

    print("\nWrong-option validation:")
    ok = quiz_mode.option_ok
    check(
        "an option equal to the answer is rejected (case/punctuation-insensitive)",
        not ok("Митоз", " митоз. "),
    )
    check("a plausible same-kind option is accepted", ok("Митоз", "Мейоз"))
    check("a year for a year: '1838' vs '1855' accepted", ok("1838", "1855"))
    check("shape mismatch (number vs word) rejected", not ok("1838", "Шлейден"))
    check("script mismatch (Cyrillic vs Latin) rejected", not ok("Митоз", "Mitosis"))
    check(
        "far longer than the answer rejected", not ok("АТФ", "Аденозинтрифосфат қышқылының қалдығы")
    )
    check("over 60 chars rejected", not ok("Жауап", "ж" * 61))
    picked = quiz_mode.pick_options(
        "Митоз", ["Мейоз", "мейоз", "Митоз", "Амитоз", "Мейоз!", "Эндомитоз"]
    )
    check(
        f"duplicates and the answer removed, 3 kept: {picked}",
        picked == ["Мейоз", "Амитоз", "Эндомитоз"],
    )

    print("\nGeneration output validation:")
    stats = flashcard_tools.ParseStats()
    drafts = flashcard_tools.parse_cards(
        json.dumps({"cards": [
            {"front": "Жасушаның бөліну түрі?", "back": "Митоз", "explanation": "",
             "wrong_options": ["Митоз", "Мейоз", "мейоз", "Амитоз", "Mitosis", "Эндомитоз"]},
            {"front": "Ұзақ жауап?", "back": "Бұл жауап өте ұзақ, өйткені модель түсіндірмені де жауапқа жазып жіберді.",  # noqa: E501
             "explanation": "", "wrong_options": ["А", "Б", "В"]},
        ]}),
        stats,
    )  # fmt: skip
    check(
        "valid options stored with the card", drafts[0].options == ("Мейоз", "Амитоз", "Эндомитоз")
    )
    check("long 'back' kept as a card but with no options", drafts[1].options == ())
    check(
        "long answers and rejected options counted",
        stats.long_answers == 1 and stats.options_rejected >= 3,
    )
    check(
        "schema asks for explanation + wrong_options and requires them (strict mode)",
        set(flashcard_tools._CARDS_SCHEMA["properties"]["cards"]["items"]["required"])
        == {"front", "back", "explanation", "wrong_options"},
    )

    print("\nMode decision (pure):")
    d = quiz_mode.choose_mode("Бұл жауап тым ұзақ: " + "сөз " * 15, [], ["Мейоз"] * 5, [])
    check(
        f"long answer -> show_answer/answer_too_long (len {d.answer_len})",
        (d.mode, d.reason) == ("show_answer", "answer_too_long"),
    )
    d = quiz_mode.choose_mode("Митоз", ["Мейоз", "Амитоз", "Эндомитоз"], [], [])
    check(
        "short answer + 3 stored options, empty deck -> multiple_choice",
        d.mode == "multiple_choice" and d.stored_options == 3,
    )
    d = quiz_mode.choose_mode("Митоз", ["Мейоз"], ["Амитоз"], ["Эндомитоз", "Шванн"])
    check(
        "stored first, then same source, then others",
        d.options[:2] == ["Мейоз", "Амитоз"] and d.mode == "multiple_choice",
    )
    d = quiz_mode.choose_mode("Митоз", ["Митоз", "митоз"], [], ["1838"])
    check(
        f"only invalid candidates -> show_answer/too_few_distractors ({d.candidates} valid)",
        (d.mode, d.reason) == ("show_answer", "too_few_distractors"),
    )

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    fc_logger = logging.getLogger("harness.flashcards")
    handler = Capture()
    fc_logger.addHandler(handler)
    previous = fc_logger.level
    fc_logger.setLevel(logging.INFO)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = FlashcardStore(Path(tmp) / "cards.db")
        await store.connect()
        try:
            now = datetime.now(UTC).replace(tzinfo=None)
            print("\nQuiz mode is logged per card (no card text):")
            short, long_ = await store.add_cards(
                telegram_id=7, source_key="kb:1", source_title="Биология",
                cards=[
                    NewCard("Жасушаның бөліну түрі?", "Митоз", "жасушаның бөліну түрі?",
                            explanation="Соматикалық жасушалар митоз арқылы бөлінеді.",
                            options=("Мейоз", "Амитоз", "Эндомитоз")),
                    NewCard("Ұзақ сұрақ?", "Бұл жауап өте ұзақ, өйткені ол бір емес бірнеше сөйлемнен тұрады.", "ұзақ сұрақ?"),  # noqa: E501
                ],
                now=now,
            )  # fmt: skip
            service = FlashcardService(store=store, router=LLMRouter([CountingProvider("x", "{}")]))
            prompt = await service.start_session(telegram_id=7, chat_id=7, kind="manual", now=now)
            assert prompt is not None
            lines = [r.getMessage() for r in records if r.getMessage().startswith("quiz mode")]
            check(
                f"short answer with stored options -> multiple_choice: {lines[-1]!r}",
                prompt.item.mode == "choice" and "mode=multiple_choice reason=-" in lines[-1],
            )
            check(
                "... only one other card exists, so the stored options did it",
                "stored_options=3" in lines[-1],
            )
            check(
                "... and the 4 buttons are the answer + its stored options",
                sorted(prompt.item.choices) == sorted(["Митоз", "Мейоз", "Амитоз", "Эндомитоз"]),
            )
            result = await service.answer_choice(
                telegram_id=7, item_id=prompt.item.id, choice=prompt.item.correct_index, now=now
            )
            lines = [r.getMessage() for r in records if r.getMessage().startswith("quiz mode")]
            check(
                f"long answer -> show_answer with reason: {lines[-1]!r}",
                result.next_prompt.item.mode == "reveal"
                and "mode=show_answer reason=answer_too_long" in lines[-1],
            )
            check("answer length is in the log line", f"answer_len={len(long_.back)}" in lines[-1])
            check(
                "no card text in any quiz-mode log line",
                not any(t in line for line in lines for t in ("Митоз", "Мейоз", "Ұзақ", "жауап")),
            )
            from bot.quiz import render_result

            check(
                "explanation shown after the answer",
                "Соматикалық жасушалар" in render_result(result),
            )
            check(
                "/cards multiple-choice count: 1 of 2",
                await service.multiple_choice_count(telegram_id=7) == 1,
            )

            print("\nPII gate on generated cards and options:")
            gen = CountingProvider("gen", json.dumps({"cards": [
                {"front": "Тәуелсіздік жылы?", "back": "1991", "explanation": "",
                 "wrong_options": ["1986", "900101300123", "1995", "1990"]},
                {"front": "Шетелге шығу құжаты?", "back": "Төлқұжат", "explanation": "", "wrong_options": []},  # noqa: E501
            ]}))  # fmt: skip
            service = FlashcardService(store=store, router=LLMRouter([gen]))
            pdf = tiny_text_pdf(
                [f"Independence history fact number {i} for the lesson." for i in range(10)]
            )
            report = await service.make_from_pdf(
                telegram_id=7, chat_id=7, data=pdf, filename="h.pdf", user_message=""
            )
            stored = [c for c in report.cards if c.back == "1991"]
            check(
                'card whose own text trips the gate ("Төлқұжат") is not stored',
                report.cards_dropped_pii == 1 and len(report.cards) == 1,
            )
            check(
                f"IIN-shaped option dropped, a spare takes its place: {stored[0].options}",
                stored[0].options == ["1986", "1995", "1990"],
            )
        finally:
            await store.close()
            fc_logger.removeHandler(handler)
            fc_logger.setLevel(previous)

    print("\nMigration of existing cards (no rewriting):")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "old.db"
        old = sqlite3.connect(path)
        old.executescript(
            "CREATE TABLE cards (id INTEGER PRIMARY KEY AUTOINCREMENT, telegram_id INTEGER NOT NULL, "  # noqa: E501
            "front TEXT NOT NULL, back TEXT NOT NULL, front_key TEXT NOT NULL, source_key TEXT NOT NULL, "  # noqa: E501
            "source_title TEXT NOT NULL, ease REAL NOT NULL DEFAULT 2.5, interval_days INTEGER NOT NULL DEFAULT 0, "  # noqa: E501
            "repetitions INTEGER NOT NULL DEFAULT 0, lapses INTEGER NOT NULL DEFAULT 0, due_at TEXT NOT NULL, "  # noqa: E501
            "last_reviewed_at TEXT, created_at TEXT NOT NULL, UNIQUE (telegram_id, front_key));"
        )  # fmt: skip
        long_text = "Ескі ұзын жауап: " + "түсіндірме " * 8
        for front, back in (("Қысқа?", "Митоз"), ("Ұзын?", long_text)):
            old.execute(
                "INSERT INTO cards (telegram_id, front, back, front_key, source_key, source_title, due_at, created_at) "  # noqa: E501
                "VALUES (1, ?, ?, ?, 'kb:1', 't', '2026-01-01 00:00:00', '2026-01-01 00:00:00')",
                (front, back, front.lower()),
            )  # fmt: skip
        old.commit()
        old.close()
        store = FlashcardStore(path)
        await store.connect()
        try:
            cards = {c.front: c for c in await store.all_cards(telegram_id=1)}
            check(
                "long answer copied into explanation verbatim",
                cards["Ұзын?"].explanation == long_text,
            )
            check("... and the answer itself not rewritten", cards["Ұзын?"].back == long_text)
            check(
                "short answer: no explanation added",
                cards["Қысқа?"].explanation is None and cards["Қысқа?"].back == "Митоз",
            )
            check("existing cards start with no stored options", cards["Қысқа?"].options == [])
        finally:
            await store.close()
        store = FlashcardStore(path)
        await store.connect()
        await store.close()
        check("migration is idempotent (second open is a no-op)", True)

    print("\nTruncated structured output is rejected, not silently shortened:")

    def mock(payload: dict) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=payload))
        )

    groq = GroqProvider(api_key="stub", model="openai/gpt-oss-120b")
    groq._client = mock(
        {"choices": [{"message": {"content": '{"cards": []}'}, "finish_reason": "length"}]}
    )
    try:
        await groq.complete(prompt="x", json_schema={"type": "object"})
    except ProviderError as exc:
        check("Groq finish_reason=length with a schema -> ProviderError", "truncated" in str(exc))
    else:
        check("Groq finish_reason=length with a schema -> ProviderError", False)
    groq._client = mock(
        {"choices": [{"message": {"content": "plain text"}, "finish_reason": "length"}]}
    )
    check("... plain-text replies are unaffected", await groq.complete(prompt="x") == "plain text")
    await groq.aclose()
    gemini = GeminiProvider(api_key="stub", model="m")
    gemini._client = mock(
        {
            "candidates": [
                {"content": {"parts": [{"text": '{"cards": []}'}]}, "finishReason": "MAX_TOKENS"}
            ]
        }
    )
    try:
        await gemini.complete(prompt="x", json_schema={"type": "object"})
    except ProviderError as exc:
        check(
            "Gemini finishReason=MAX_TOKENS with a schema -> ProviderError", "truncated" in str(exc)
        )
    else:
        check("Gemini finishReason=MAX_TOKENS with a schema -> ProviderError", False)
    await gemini.aclose()
    check(
        "card budget raised for the larger schema (8192)",
        flashcard_tools.FLASHCARD_MAX_TOKENS == 8192,
    )


async def download_check(url: str, *, convert_to_mp3: bool) -> None:
    """Run the Phase 4 generalized downloader (no Telegram) and print what
    would be sent — the real-network counterpart of `phase4_offline_checks`."""
    from config import load_settings
    from harness.media import MediaError, MediaPipeline
    from llm_router import build_router

    print(f"\nMedia download: {url} (mp3={convert_to_mp3})")
    settings = load_settings()
    router = build_router(settings)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        notes = NoteStore(Path(tmp) / "notes.db")
        await notes.connect()
        media = MediaPipeline(
            router=router,
            download_dir=settings.download_dir,
            note_store=notes,
            ffmpeg_path=settings.ffmpeg_path,
            tesseract_cmd=settings.tesseract_cmd,
            ocr_lang=settings.ocr_lang,
            ocr_tessdata_dir=settings.ocr_tessdata_dir,
            max_download_mb=settings.media_max_download_mb,
        )
        print("  " + media.describe())

        async def progress(text: str) -> None:
            print(f"  ... {text}")

        try:
            result = await media.download(
                url, convert_to_mp3=convert_to_mp3, user_message=url, on_progress=progress
            )
        except MediaError as exc:
            check(f"download failed: {exc}", False)
        finally:
            await router.aclose()
            await notes.close()
    check("file downloaded", len(result.data) > 0)
    check("description returned", bool(result.description.strip()))
    print(f"\n--- {result.filename} ({len(result.data) / 1e6:.1f} MB) ---")
    print(result.description)
    print("-------------------")


async def screenshot_check(image_path: str) -> None:
    """Run the Phase 4 screenshot pipeline (real OCR, no Telegram) and print
    the result — the real-OCR counterpart of `phase4_offline_checks`.

    Wires up a real (temp, throwaway) encrypted DocumentArchive alongside the
    notes staging store, same as `main.py`, so the PII safety net's redirect
    path (`document_fields.has_pii_signals`) is exercised for real if the
    image trips it — not just the offline stub-based check."""
    from config import load_settings
    from harness.documents import DocumentArchive
    from harness.media import MediaError, MediaPipeline
    from llm_router import build_router
    from storage.documents import DocumentStore

    print(f"\nScreenshot pipeline: {image_path}")
    settings = load_settings()
    router = build_router(settings)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        notes = NoteStore(Path(tmp) / "notes.db")
        await notes.connect()
        documents = DocumentStore(
            db_path=Path(tmp) / "documents.db",
            scan_dir=Path(tmp) / "scans",
            encryption_key=settings.documents_encryption_key,
        )
        await documents.connect()
        archive = DocumentArchive(
            store=documents,
            tesseract_cmd=settings.tesseract_cmd,
            ocr_lang=settings.ocr_lang,
            ocr_tessdata_dir=settings.ocr_tessdata_dir,
        )
        media = MediaPipeline(
            router=router,
            download_dir=settings.download_dir,
            note_store=notes,
            tesseract_cmd=settings.tesseract_cmd,
            ocr_lang=settings.ocr_lang,
            ocr_tessdata_dir=settings.ocr_tessdata_dir,
            document_archive=archive,
        )
        print("  " + media.describe())
        image_bytes = Path(image_path).read_bytes()
        try:
            result = await media.capture_screenshot(
                telegram_id=0, image_bytes=image_bytes, user_message=""
            )

            if result.redirected_to_documents:
                assert result.document is not None
                check("redirected to the encrypted document archive, not notes", True)
                check(
                    "nothing landed in the unencrypted notes table",
                    await notes.count(telegram_id=0) == 0,
                )
                print(f"\n--- redirected: document_type={result.document.document_type} ---")
                print(result.document.fields)
                print("-------------------")
            else:
                assert result.note is not None
                check("note saved", result.note.id > 0)
                print(f"\n--- {result.note.title} (tags: {result.note.tags}) ---")
                print(result.note.content_md)
                print("-------------------")
        except MediaError as exc:
            check(f"screenshot pipeline failed: {exc}", False)
        finally:
            await router.aclose()
            await notes.close()
            await documents.close()


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
    if "--download" in sys.argv:
        index = sys.argv.index("--download")
        if index + 1 >= len(sys.argv):
            raise SystemExit("usage: selfcheck.py --download <url> [mp3]")
        convert_to_mp3 = "mp3" in sys.argv[index + 2 :]
        await download_check(sys.argv[index + 1], convert_to_mp3=convert_to_mp3)
        print("\nDownload check passed.")
        return
    if "--kb" in sys.argv:
        # --kb page1.jpg page2.jpg ... --q "question 1" --q "question 2"
        args = sys.argv[sys.argv.index("--kb") + 1 :]
        images = [a for a in args[: args.index("--q")] if a] if "--q" in args else args
        questions = [args[i + 1] for i, a in enumerate(args) if a == "--q" and i + 1 < len(args)]
        await knowledge_live_check(images, questions)
        print("\nKnowledge-base live check finished.")
        return
    if "--screenshot" in sys.argv:
        index = sys.argv.index("--screenshot")
        if index + 1 >= len(sys.argv):
            raise SystemExit("usage: selfcheck.py --screenshot <image_path>")
        await screenshot_check(sys.argv[index + 1])
        print("\nScreenshot check passed.")
        return
    await offline_checks()
    phase2_offline_checks()
    phase3_offline_checks()
    await document_store_offline_check()
    await phase4_offline_checks()
    await note_store_offline_check()
    language_offline_checks()
    await routing_offline_checks()
    formatting_offline_checks()
    download_format_offline_checks()
    await knowledge_offline_checks()
    pii_gate_regression_checks()
    await archive_fallback_and_delete_checks()
    await ask_outcome_logging_checks()
    sm2_and_clock_checks()
    await flashcard_service_checks()
    await flashcard_routing_checks()
    await quiz_mode_checks()
    if "--live" in sys.argv:
        await live_check()
    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
