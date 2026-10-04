# Harness — family Telegram bot

Agent harness over free-tier LLMs. See `CLAUDE.md` for the full spec and roadmap.

**Current status: Phase 5a** — tutor + link/video summarizer + document archive +
generalized media downloader + screenshot notes + RAG knowledge base (`/ask`,
notebook pages via `/page`). Flashcards/quizzes are Phase 5b, not built yet.

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
  media.py           Phase 4 route: any-URL download/convert + screenshot OCR -> note staging
  knowledge.py       Phase 5a route: PII gate -> vision transcription -> embed/index; /ask
  prompts.py         system prompts, verbatim from CLAUDE.md §9 (+ Phase 4's own, see file)
strings.py           every static, non-LLM user-facing string (CLAUDE.md §8) — Kazakh
bot/
  __init__.py        Bot/Dispatcher construction
  handlers/          Telegram handlers — translate events into Harness calls
    links.py         URL messages + /summarize; video jobs run in the background
    documents.py     photo uploads + /find; document archive only
    media.py         /download (background job) + photos captioned /note
    knowledge.py     /ask + photos captioned /page (notebook pages)
  middlewares.py     whitelist gate + user tracking
  states.py          FSM states
storage/
  db.py              plain SQLite: users + chat history
  documents.py        SEPARATE encrypted SQLite: document fields/OCR text + scan references
  notes.py           plain SQLite staging table for Phase 4 screenshot notes (source for 5a)
  knowledge.py       sqlite-vec index: sources/chunks/vectors, partitioned per user
  encrypted_files.py Fernet-encrypted files: document scans + notebook page photos
tools/               standalone pipelines, no bot/harness imports
  urls.py            URL detection + article/video classification
  article.py         httpx fetch -> trafilatura extraction
  downloader.py      yt-dlp: probe, subtitles, audio, and (Phase 4) any-file download/mp3
  transcript.py      VTT parsing, [mm:ss] transcript rendering
  transcriber.py     speech-to-text: Groq Whisper -> local faster-whisper
  summarizer.py      builds the §9 prompts and calls llm_router (Phase 2)
  media_notes.py     Phase 4: media description + screenshot title/tags via llm_router
  language.py        reply-language decision (command word + URLs stripped first)
  ocr.py             local Tesseract OCR — never a cloud vision API (CLAUDE.md §5)
  vision_ocr.py      handwriting transcription via the vision chain (after the PII gate)
  rag.py             chunking + grounded answer with structured output
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

### Phase 4 extras (media pipeline) — no new required config

Reuses Phase 2's `ffmpeg`/yt-dlp setup (for `/download`) and Phase 3's Tesseract
setup (for `/note` screenshots) as-is — nothing new to install if you already
did those. Two optional settings, both with working defaults:

- `NOTES_DB_PATH` (default `data/notes.db`) — plain SQLite staging table for
  screenshot notes; not encrypted (screenshots are the explicit non-sensitive
  case, CLAUDE.md §5), not searchable yet (Phase 5's vector store will index
  it later).
- `MEDIA_MAX_DOWNLOAD_MB` (default `45`) — `/download` refuses anything over
  this; Telegram bot uploads cap at ~50 MB.

### Phase 5a extras (knowledge base) — one new package, no new keys

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # adds sqlite-vec
```

- **Vector store: `sqlite-vec`.** A prebuilt loadable SQLite extension per
  platform (pip wheel, no compiler, no admin, no server) on the stdlib sqlite3
  we already use; Chroma would add onnxruntime/hnswlib and a server-ish stack —
  the native-build pain we hit with SQLCipher/tessdata. Requirement: a Python
  whose sqlite3 allows extension loading (python.org Windows builds, Debian/
  Ubuntu `apt` Python and `uv` Pythons do; some `pyenv` builds don't — the bot
  refuses to start with a clear message in that case).
- **Embeddings: `gemini-embedding-2`, 768 dims** (`EMBEDDING_MODEL`,
  `EMBEDDING_DIM`). Chosen by a retrieval test: 7/7 top-1 on Kazakh→Kazakh and
  Kazakh→Russian/English questions, with a ~0.17 gap between the weakest
  correct hit and an off-topic query. Local `multilingual-e5-small` (ONNX) also
  got 7/7 but with 0.01 margins and a 470 MB download — it is the fallback if
  notes must never leave the server. The index is pinned to one model: changing
  either setting needs a rebuild (move `data/knowledge.db` aside; staging notes
  re-migrate on the next start, page photos would need re-sending).
- `RAG_MIN_SCORE` (default `0.65`) — below this cosine similarity a chunk is
  ignored; if nothing passes, `/ask` says so in Kazakh without calling the LLM.
  `RAG_TOP_K` (default `5`).
- **Vision chain for handwriting**: `VISION_PROVIDER_CHAIN` (default
  `gemini,groq,openrouter`), models `GEMINI_VISION_MODEL` (defaults to
  `GEMINI_MODEL`), `GROQ_VISION_MODEL` (`qwen/qwen3.8-27b`),
  `OPENROUTER_VISION_MODEL` (`qwen/qwen3.8-27b:free`); Tesseract is the last
  resort (printed text only).
- **PII gate**: every `/page` photo is OCR'd **locally** with Tesseract first and
  checked with `has_pii_signals`; if it fires, the photo goes to the encrypted
  document archive and never to a cloud vision API. If Tesseract is missing or
  fails, the page is refused (fails closed).
- `GROQ_REASONING_EFFORT` (default `low`) — Groq's `gpt-oss` spends hidden
  reasoning tokens out of `max_tokens`; measured 124-221 at default effort vs
  23-49 at `low` on the RAG prompt (default effort + `max_tokens=128` → HTTP 400
  "Failed to validate JSON"). Leave empty for non-reasoning Groq models.
- `KNOWLEDGE_DB_PATH` (`data/knowledge.db`), `KNOWLEDGE_PAGE_DIR`
  (`data/knowledge_pages`, Fernet-encrypted photos, same key as documents).
- Existing `/note` staging rows are migrated into the index on every start
  (idempotent, in the background); new `/note`s are indexed as they are saved.

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

**Phase 4 generalized downloader**, no Telegram (runs the real yt-dlp fetch +
LLM description and prints it; add `mp3` to also exercise the ffmpeg conversion):

```powershell
.\.venv\Scripts\python.exe scripts\selfcheck.py --download https://www.youtube.com/watch?v=jNQXAC9IVRw
.\.venv\Scripts\python.exe scripts\selfcheck.py --download https://www.youtube.com/watch?v=jNQXAC9IVRw mp3
```

**Phase 4 screenshot pipeline**, no Telegram (runs real OCR + the LLM
formatting call on a local image and prints the resulting note). The JSON
parsing itself (valid/invalid/empty replies) is covered offline, without OCR
or a real LLM call, by the default `scripts\selfcheck.py` run above (see
"Screenshot note formatting" in its output):

```powershell
.\.venv\Scripts\python.exe scripts\selfcheck.py --screenshot path\to\screenshot.png
```

**Phase 5a knowledge base**, real pipeline in a throwaway temp store (local
Tesseract gate, real vision chain, real embeddings) — prints each retrieved
chunk with its score and the final answer. Keep test photos outside the repo:

```powershell
.\.venv\Scripts\python.exe scripts\selfcheck.py --kb page1.jpg page2.jpg --q "Абылай хан қай жылы хан сайланды?" --q "Ом заңының формуласы қандай?"
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
| `/find <words from the text>` on a document stored as "unknown" | found via its encrypted OCR text: matching lines + the scan (unknown documents store no guessed fields) |
| `/delete <id>` (id shown as `#N` when saved and in `/find`) | 🗑 document and its encrypted scan removed; another user's id behaves like a missing one |
| `/download <url>` | "⏳ processing…" placeholder, then an inline-playable mp4 video (H.264/AAC, highest resolution under `MEDIA_MAX_DOWNLOAD_MB`) with a Kazakh LLM description as the caption |
| `/download <url> mp3` | same, but audio-only, converted to mp3 (needs ffmpeg) |
| a photo captioned `/note` | "⏳ processing…" placeholder, then 📝 bold title, #tags, and the OCR'd text verbatim (the LLM only writes title + tags) |
| `/note` with no photo | usage reminder: caption a photo with `/note` |
| a notebook-page photo captioned `/page` (optional title after it) | "⏳ Дәптер бетін оқып жатырмын…" then 📚 added, with title + chunk count |
| a passport photo captioned `/page` | ⚠️ redirected to the encrypted document archive, never sent to a vision API |
| `/ask <question>` | Kazakh answer with [n] citations, 📎 source line(s), then the original page photo if it came from a page |
| `/ask` about something not in your notes | static Kazakh "nothing found in your notes" — no answer from model knowledge |
| a voice message or a non-image file | "text, links and document photos only" |

Replies follow the CLAUDE.md language policy: Kazakh by default, English if
your own message (not the article) is in English, never Russian. The reply
language is decided in code from your message, not left to the model. This
now also covers every static reply (command help, error/status notices, the
Telegram command menu) via `strings.py`, not just LLM output (CLAUDE.md §8).

## Notes / known limits

- **Summaries, `/download` descriptions and `/note` are sent as Telegram HTML**
  (`bot/formatting.py`): model Markdown is converted and escaped there, never
  MarkdownV2, whose parser rejects unescaped `_ * [ ] ( )` and fails the send.
  If Telegram still rejects a chunk, it is resent as plain text. Tutor replies
  are still plain text.
- **Reply language** is decided by `tools/language.py`: the leading bot command
  and URLs are stripped first, so `/download <url>` defaults to Kazakh.
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

### Phase 4: media pipeline extras

- **`/download` is deliberately separate from `/summarize`.** `/summarize`
  only treats the hardcoded hosts in `tools/urls.py` (`_VIDEO_HOSTS`) as
  video, everything else as an article to read; `/download` doesn't do that
  classification at all — it just asks yt-dlp to fetch whatever's at the URL,
  so it also works on hosts `/summarize` would have treated as an article.
- **The description is metadata-based, not a transcript summary.** Unlike
  `/summarize`'s video path, `/download` doesn't fetch subtitles or run STT —
  it asks the LLM to describe the file from its title/uploader/description/
  chapters alone. If you want an actual transcript summary, use `/summarize`.
- **A download is read fully into memory** before being sent to Telegram
  (not streamed from disk), so `MEDIA_MAX_DOWNLOAD_MB` is a hard cap, checked
  after the file lands and before it's ever handed to the LLM or Telegram.
- **`/note` vs. a plain document photo:** `bot/handlers/media.py`'s router is
  included *before* `bot/handlers/documents.py`'s in `bot/handlers/__init__.py`
  specifically so a `/note`-captioned photo is claimed first; any other photo
  still goes to the Phase 3 PII pipeline unchanged — default photo behavior
  did not change.
- **The staging table has no search on purpose** (CLAUDE.md §7 Phase 4) —
  `storage/notes.py` only saves, counts and (Phase 5a) iterates for the
  one-way migration. Search happens in the knowledge index, never here.
