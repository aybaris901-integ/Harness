# Harness — family Telegram bot

Agent harness over free-tier LLMs. See `CLAUDE.md` for the full spec and roadmap.

**Current status: Phase 1** — bot skeleton + tutor. No tools yet.

## Layout

```
main.py              entry point: wires storage -> router -> harness -> bot
config.py            all env/.env reading happens here and nowhere else
llm_router.py        the ONLY way feature code calls an LLM; fallback chain
providers/           one adapter per vendor (gemini, groq, openrouter)
harness/
  orchestrator.py    the harness: the only thing that talks to bot + LLM + storage
  prompts.py         system prompts, verbatim from CLAUDE.md §9
bot/
  __init__.py        Bot/Dispatcher construction
  handlers/          Telegram handlers — translate events into Harness calls
  middlewares.py     whitelist gate + user tracking
  states.py          FSM states
storage/db.py        SQLite: users + chat history
tools/               empty until Phase 2
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
| a photo or file | "text only for now" |

The tutor replies in whatever language you write in, Russian by default.

## Notes / known limits in Phase 1

- **Replies are plain text.** Model output often contains Markdown (`**bold**`),
  which will show literally. Sending it raw is deliberate: Telegram's MarkdownV2
  parser rejects unescaped `_ * [ ] ( )` and would make the send fail outright.
- **FSM state is in memory**, so a restart drops the "am I mid-lesson" flag. The
  conversation itself is in SQLite and is reloaded on the next message, so the
  tutor picks up where it left off.
- **History is per (user, chat)** and capped at `HISTORY_LIMIT` turns per request.
- `data/harness.db` is gitignored, as is `.env`.
