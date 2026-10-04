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

    from aiogram import Bot, Dispatcher
    from aiogram.client.session.base import BaseSession
    from aiogram.types import Chat, Message, Update, User

    import bot.handlers.links as links_handlers
    import bot.handlers.media as media_handlers
    from bot.handlers import build_root_router
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

        async def search(self, *, telegram_id, query):
            self.queries.append(query)
            return None

    session = RecordingSession()
    fake_bot = Bot(token="42:offline-selfcheck", session=session)
    media, harness, archive = FakeMedia(), FakeHarness(), FakeArchive()
    dp = Dispatcher(harness=harness, media=media, archive=archive, settings=None)
    dp.include_router(build_root_router())

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
        await dp.feed_update(fake_bot, Update(update_id=update_id, message=message))
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
    if "--live" in sys.argv:
        await live_check()
    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
