# Harness — family Telegram bot

Agent harness over free-tier LLMs. See `CLAUDE.md` for the full spec and roadmap.

**Current status: Phase 3** — tutor + link/video summarizer + document archive.

## Layout

```
main.py              entry point: wires storage -> router -> harness -> bot
config.py            all env/.env reading happens here and nowhere else
llm_router.py        the ONLY way feature code calls an LLM; fallback chain
providers/           one adapter per vendor (gemini, groq, openrouter, openrouter-paid)
harness/
  orchestrator.py    the harness: the only thing that talks to bot + LLM + storage
  links.py           Phase 2 route: article vs video, subtitles vs STT, progress
  documents.py       Phase 3 route: OCR -> field extraction -> encrypted storage -> search
  prompts.py         system prompts, verbatim from CLAUDE.md §9
strings.py           every static, non-LLM user-facing string (CLAUDE.md §8) — Kazakh
bot/
  __init__.py        Bot/Dispatcher construction
  handlers/          Telegram handlers — translate events into Harness calls
    links.py         URL messages + /summarize; video jobs run in the background
    documents.py     photo uploads + /find; document archive only
  middlewares.py     whitelist gate + user tracking
  states.py          FSM states
storage/
  db.py              plain SQLite: users + chat history
  documents.py        SEPARATE encrypted SQLite: document fields/OCR text + scan references
tools/               standalone pipelines, no bot/harness imports
  urls.py            URL detection + article/video classification
  article.py         httpx fetch -> trafilatura extraction
  downloader.py      yt-dlp: probe, subtitles, audio
  transcript.py      VTT parsing, [mm:ss] transcript rendering
  transcriber.py     speech-to-text: Groq Whisper -> local faster-whisper
  summarizer.py      builds the §9 prompts and calls llm_router
  ocr.py             local Tesseract OCR — never a cloud vision API (CLAUDE.md §5)
  document_fields.py regex/heuristic field extraction + search-query scoring
scripts/selfcheck.py local sanity check, no Telegram needed
```

The dependency direction is one-way: `bot` → `harness` → (`llm_router`, `storage`).
Handlers never call a provider or touch SQL directly.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env      # then fill it in
```

At minimum `.env` needs:

- `BOT_TOKEN` — from [@BotFather](https://t.me/BotFather)
- `GEMINI_API_KEY` — from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free, no card)
- `ALLOWED_USER_IDS` — your Telegram ID from [@userinfobot](https://t.me/userinfobot).
  Leave empty while testing to allow anyone; set it before the bot goes public.

`GROQ_API_KEY` and `OPENROUTER_API_KEY` are optional — providers without a key are
skipped, so the chain quietly shortens to whatever you configured.

### Phase 2 extras (link / video summarizer)

- **Videos with subtitles** need nothing extra — yt-dlp reads the captions.
- **Videos without subtitles** need speech-to-text: set `GROQ_API_KEY`
  (Whisper on Groq's free tier) *or* `pip install faster-whisper` for a local,
  slower, fully private fallback. With neither, such videos get a clear
  "not configured" reply.
- **ffmpeg** (`winget install Gyan.FFmpeg`, or `apt install ffmpeg` on the VPS)
  lets the bot compress and split audio so long videos fit Groq's 25 MB limit.
  Without it only videos under roughly 25 minutes can be transcribed.
- **deno** (`winget install DenoLand.Deno`) is what yt-dlp now wants as a JS
  runtime for YouTube; it works without one today but yt-dlp warns that some
  formats may be missing. Keep `yt-dlp` itself fresh: `pip install -U yt-dlp`.

Setting `OPENROUTER_API_KEY` also enables `openrouter-paid`, a **paid** last-resort
tier (`OPENROUTER_PAID_MODEL`, default `google/gemini-3.8-flash`) that only fires
after Gemini, Groq and the OpenRouter free pool have all failed. Each paid call is
logged at WARNING with a `[PAID]` marker — `grep '\[PAID\]'` the logs to see how
much credit is being used. Drop `openrouter-paid` from `LLM_PROVIDER_CHAIN` to
stay strictly free.

### Phase 3 extras (document archive) — REQUIRED, not optional

Unlike Phase 2's extras, `DOCUMENTS_ENCRYPTION_KEY` is **required**: `main.py`
refuses to start without it, the same way it refuses to start without `BOT_TOKEN`.

**1. Generate the encryption key** (one-time, per deployment):

```powershell
.\.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Paste the output into `.env` as `DOCUMENTS_ENCRYPTION_KEY`. See "Where the key
lives" below for exactly what this key protects and why it's not in the DB itself.

**2. Install the Tesseract OCR binary** (not a pip package — `pytesseract` is
just a thin wrapper around it):

- Windows: `winget install UB-Mannheim.TesseractOCR`, or download from
  [the UB-Mannheim build page](https://github.com/UB-Mannheim/tesseract/wiki).
- Debian/Ubuntu VPS: `apt install tesseract-ocr`.
- If it's not on `PATH`, set `TESSERACT_CMD` in `.env` to the binary (or its
  folder) — same pattern as `FFMPEG_PATH` in Phase 2.

**3. Install a language pack for Kazakh/Russian documents.** Tesseract only
ships `eng.traineddata` by default, which reads Cyrillic text poorly. For real
Kazakhstani documents (passports, vehicle registration, receipts), install:

- Windows: the UB-Mannheim installer has a language-pack checklist — tick
  Kazakh and Russian during install.
- Debian/Ubuntu: `apt install tesseract-ocr-kaz tesseract-ocr-rus`.

Then set `OCR_LANG=kaz+rus+eng` in `.env`. Until you do this, `OCR_LANG`
defaults to `eng` — OCR will still run, just poorly on Cyrillic documents
(garbled text, and `document_type` stuck at `"unknown"` for anything without
an MRZ block to fall back on).

If you can't write to Tesseract's own `tessdata` folder (on Windows this is
under `Program Files`, which needs admin rights), download the `.traineddata`
files yourself instead of using the installer/`apt` package:

- `kaz.traineddata` / `rus.traineddata` / `eng.traineddata` from
  [tesseract-ocr/tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast)
  into any folder you own (e.g. `%LOCALAPPDATA%\tessdata_kaz_rus`).
- Set `OCR_TESSDATA_DIR` in `.env` to that folder. It replaces Tesseract's
  search path for that call, so the folder needs **all** languages named in
  `OCR_LANG`, not just the new ones.

**4. Field extraction is regex/heuristics by default** (CLAUDE.md §9) — no
LLM, cloud or local, is needed for the common case. `DOCUMENT_LOCAL_LLM_ENABLED`
is an optional fallback for documents where regex genuinely finds nothing
(neither a recognized document type nor any field); it talks to a local
Ollama-compatible server directly, never `llm_router.py`, and stays off by default.

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

On startup it logs the bot username and the active provider chain, e.g.
`LLM chain: gemini(gemini-2.5-flash) -> groq(llama-3.3-70b-versatile)`.
Stop with Ctrl+C. Only one instance may poll at a time — Telegram returns a
409 conflict if a second one starts.

## Test

**Without Telegram** — storage, harness wiring and router fallback:

```powershell
.\.venv\Scripts\python.exe scripts\selfcheck.py
```

**Check your API key** actually works (one real call, prints the reply):

```powershell
.\.venv\Scripts\python.exe scripts\selfcheck.py --live
```

**Phase 2 pipeline on one URL**, no Telegram (runs the real fetch/yt-dlp/LLM
steps and prints the summary; add `--link` for either an article or a video):

```powershell
.\.venv\Scripts\python.exe scripts\selfcheck.py --link https://en.wikipedia.org/wiki/Spaced_repetition
.\.venv\Scripts\python.exe scripts\selfcheck.py --link https://www.youtube.com/watch?v=jNQXAC9IVRw
```

**Phase 3 document pipeline**, no Telegram — regex field extraction, search
scoring and the encrypted store's save/search/decrypt roundtrip all run as
part of the default `scripts\selfcheck.py` call above (see "Document field
extraction" / "Document store (encrypted)" in its output). This does **not**
exercise real OCR, since it doesn't need a Tesseract install to verify the
storage/extraction/search logic. To check OCR itself works end to end, use
Telegram (below) — or run this one-off with any document-like photo:

```powershell
.\.venv\Scripts\python.exe -c "
import asyncio
from config import load_settings
from tools.ocr import extract_text

async def main():
    s = load_settings()
    text = await extract_text(open('path/to/photo.jpg', 'rb').read(), tesseract_cmd=s.tesseract_cmd, lang=s.ocr_lang, tessdata_dir=s.ocr_tessdata_dir)
    print(text)

asyncio.run(main())
"
```

**In Telegram**, after `python main.py`:

| You send | Expected |
|---|---|
| `/start` | greeting, asks for a topic |
| `рекурсия` | 2-3 sentence explanation + metaphor + one worked example + a check-question |
| a wrong answer | says exactly where the reasoning broke, gives a new example at the same level |
| a correct answer | raises difficulty or offers a related concept |
| `/tutor закон Ома` | starts a fresh lesson immediately |
| `/reset` | wipes history, confirms how many messages were deleted |
| `/cancel` | leaves the lesson (next message starts a new one) |
| an article link | 📄 title, one-sentence takeaway, 3-6 bullets, unsupported-claim note — within a few seconds |
| a YouTube link with captions | "⏳ Бейне қабылданды, өңдеп жатырмын…" placeholder, edited into 🎬 title + timestamped key points |
| a video with no captions | placeholder shows progress (downloading → transcribing → summarizing), then the summary; needs STT configured |
| `/summarize <url>` | same as sending the link |
| a text-only tweet / post | falls back to the article path |
| a photo of a document (passport, vehicle registration, contract, receipt) | "⏳ Құжатты өңдеп жатырмын…" then ✅ with the detected document type + extracted fields |
| `/find <query>` (e.g. `/find көлік нөмірі`) | best-matching stored document's fields, then the original scan photo |
| `/find` with no match | "no matching document" reply |
| a voice message or a non-image file | "text, links and document photos only" |

Replies follow the CLAUDE.md language policy: Kazakh by default, English if
your own message (not the article) is in English, never Russian. The reply
language is decided in code from your message, not left to the model. This
now also covers every static reply (command help, error/status notices, the
Telegram command menu) via `strings.py`, not just LLM output (CLAUDE.md §8).

## Notes / known limits

- **Replies are plain text.** Model output often contains Markdown (`**bold**`),
  which will show literally. Sending it raw is deliberate: Telegram's MarkdownV2
  parser rejects unescaped `_ * [ ] ( )` and would make the send fail outright.
- **FSM state is in memory**, so a restart drops the "am I mid-lesson" flag. The
  conversation itself is in SQLite and is reloaded on the next message, so the
  tutor picks up where it left off.
- **History is per (user, chat)** and capped at `HISTORY_LIMIT` turns per request.
- `data/harness.db` is gitignored, as is `.env`; yt-dlp scratch files go to
  `DOWNLOAD_DIR` (default `data/downloads/<job id>/`) and are deleted after each job.
- **Videos run in the background**, at most two at a time (`LinkSummarizer`
  semaphore). A restart mid-job loses that job; the user just resends the link.
- **Source text is capped** at ~120k characters before it reaches the model
  (`tools/summarizer.py: MAX_INPUT_CHARS`) — plenty for Gemini, and keeps the
  Groq/OpenRouter fallbacks from choking on a two-hour transcript.
- A link summary is stored in the chat history like any other turn, so a
  follow-up question ("explain point 2") lands in the tutor with context.
- One link per message; extra links are ignored with a note.

### Phase 3: document archive — encryption, storage, and privacy

- **Encryption choice: AES via `cryptography`'s `Fernet`** (AES-128-CBC +
  HMAC-SHA256, authenticated), not SQLCipher. SQLCipher needs a
  non-standard SQLite build (`pysqlcipher3` / `sqlcipher3-binary`), which is
  an extra native-dependency headache on both Windows dev and a small VPS for
  no real benefit at this scale — a single-key, single-user encrypted-blob
  scheme is simpler to reason about and just as private. Two things are
  encrypted: the scanned image (as a standalone `<uuid>.enc` file under
  `DOCUMENTS_SCAN_DIR`) and the extracted field values + raw OCR text (as BLOB
  columns in `DOCUMENTS_DB_PATH`, a database file **separate** from
  `DB_PATH`/chat history, per the CLAUDE.md architecture). `document_type` and
  field *names* (not values — e.g. `vin` is stored in the clear but `1HGCM...`
  is not) stay in plaintext columns so `/find` can search without decrypting
  every document, or another user's documents, on every query.
- **Where the key lives:** `DOCUMENTS_ENCRYPTION_KEY` in `.env` — nowhere
  else. It is read once in `config.py` (the only module that touches
  `os.environ`, per CLAUDE.md §5) and handed to `storage/documents.py`'s
  `Fernet` instance; it is never logged, never written to the database, and
  the app refuses to start without it (`ConfigError`, same as a missing
  `BOT_TOKEN`). Back it up outside the VPS — there is no recovery path if it's
  lost, by design (that's what makes it real encryption rather than obfuscation).
- **OCR is 100% local** (`tools/ocr.py`, Tesseract via `pytesseract`) — a
  document photo's pixels are decoded and OCR'd in a worker thread on this
  machine and never touch `llm_router.py` or any cloud vision API (CLAUDE.md
  §5). Field extraction (`tools/document_fields.py`) is regex/heuristics by
  default for the same reason; the optional local-LLM fallback
  (`DOCUMENT_LOCAL_LLM_ENABLED`) talks to a local Ollama-compatible server
  directly, never the cloud LLM chain.
- **Search is also non-LLM.** `/find` matches your query against
  `document_type` and field names via a small keyword/synonym table
  (`tools/document_fields.py: score_query`) — deterministic, auditable, and
  keeps PII values from ever being sent anywhere to answer a search.
- `data/documents.db` and `data/document_scans/` are gitignored along with
  everything else under `data/`.
