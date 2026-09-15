# Harness — family Telegram bot

Agent harness over free-tier LLMs. See `CLAUDE.md` for the full spec and roadmap.

**Current status: Phase 2** — tutor + link/video summarizer.

## Layout

```
main.py              entry point: wires storage -> router -> harness -> bot
config.py            all env/.env reading happens here and nowhere else
llm_router.py        the ONLY way feature code calls an LLM; fallback chain
providers/           one adapter per vendor (gemini, groq, openrouter, openrouter-paid)
harness/
  orchestrator.py    the harness: the only thing that talks to bot + LLM + storage
  links.py           Phase 2 route: article vs video, subtitles vs STT, progress
  prompts.py         system prompts, verbatim from CLAUDE.md §9
bot/
  __init__.py        Bot/Dispatcher construction
  handlers/          Telegram handlers — translate events into Harness calls
    links.py         URL messages + /summarize; video jobs run in the background
  middlewares.py     whitelist gate + user tracking
  states.py          FSM states
storage/db.py        SQLite: users + chat history
tools/               standalone pipelines, no bot/harness imports
  urls.py            URL detection + article/video classification
  article.py         httpx fetch -> trafilatura extraction
  downloader.py      yt-dlp: probe, subtitles, audio
  transcript.py      VTT parsing, [mm:ss] transcript rendering
  transcriber.py     speech-to-text: Groq Whisper -> local faster-whisper
  summarizer.py      builds the §9 prompts and calls llm_router
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
| a YouTube link with captions | "⏳ обрабатываю…" placeholder, edited into 🎬 title + timestamped key points |
| a video with no captions | placeholder shows progress (downloading → transcribing → summarizing), then the summary; needs STT configured |
| `/summarize <url>` | same as sending the link |
| a text-only tweet / post | falls back to the article path |
| a photo or file | "text and links only for now" |

Replies follow the CLAUDE.md language policy: Kazakh by default, English if
your own message (not the article) is in English, never Russian. The reply
language is decided in code from your message, not left to the model.

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
