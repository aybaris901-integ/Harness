# Telegram bot with harness + free LLMs — project spec

This document is meant to be dropped into the repo (e.g. as `CLAUDE.md` or referenced from it) so Claude Code has the full picture before writing code.

## 1. Goal

A personal Telegram bot for a family that acts as an agent with several independent tool pipelines behind one orchestrator ("harness"). All LLM calls go through free-tier APIs. No paid infrastructure beyond a cheap VPS.

## 2. Architecture

```
User (Telegram)
      |
Telegram bot layer (aiogram)
      |
Harness / orchestrator  <-- the core: routes intent, calls tools, manages context
      |
  +---+-------------------+
  |                       |
Tool pipelines        LLM router (free APIs)
  - link/article scraper    - Gemini (primary, long context, multimodal)
  - yt-dlp + ffmpeg          - Groq (fast, fallback, also used for Whisper STT)
  - OCR / vision              - OpenRouter (fallback pool of :free models)
  - RAG retriever
      |
Storage
  - encrypted local DB (documents, PII)
  - relational DB (users, history, spaced-repetition state)
  - vector DB (notes/article embeddings for RAG)
```

Design principle: the harness is the only component that talks to both the bot layer and the tools/LLM layer. Tool pipelines should be callable as plain Python functions/modules so they can also be exposed to the LLM as function-calling tools later.

## 3. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Bot framework | `aiogram` 3.x | async, FSM for multi-step flows (tutor mode) |
| Orchestrator | plain Python service (FastAPI optional) | no agent framework needed at this scale — keep it simple and explicit |
| Task queue | `APScheduler` (cron-style jobs) + background `asyncio` tasks for long jobs (video download, transcription) | avoid blocking the bot while a 2-hour video downloads |
| Database | SQLite to start (Postgres if it outgrows single-file) | users, chat history, spaced-repetition state |
| Encrypted storage | SQLCipher, or AES-encrypted files with a key never sent anywhere | for documents: passports, contracts, receipts |
| Vector DB | `ChromaDB` or SQLite + `sqlite-vec` extension | good enough for single-user/family scale, no need for Pinecone/Weaviate |
| Embeddings | Gemini Embedding API (free) or local `sentence-transformers` | local option = fully private, no network call |
| Link/article extraction | `trafilatura` or `readability-lxml` | clean text extraction, strips ads/navigation |
| Video/audio download | `yt-dlp` | YouTube, TikTok, Reels, Twitter/X; update regularly (`pip install -U yt-dlp`) |
| Audio conversion | `ffmpeg` | mp3 conversion, track extraction |
| Speech-to-text | YouTube auto-subs via yt-dlp when available; otherwise `faster-whisper` (local) or Whisper via Groq API | prefer subtitles first — cheapest and fastest |
| OCR | `Tesseract` or `PaddleOCR` for documents with PII; Gemini vision for non-sensitive images (screenshots, whiteboards, diagrams) | see security note below — do not send PII documents to cloud APIs |
| Spaced repetition | SM-2 algorithm, implemented directly (no external lib needed, ~50 lines) | interval, ease factor, repetition count per card |

## 4. Free LLM providers & routing

Use a fallback chain in the harness rather than a single provider — free-tier limits tighten without warning.

1. **Gemini (Google AI Studio)** — primary. Free Flash models, no card required, up to 1M token context, multimodal (text/image/audio). Best default for summarization and vision tasks.
2. **Groq** — fallback / speed-critical calls. ~30 req/min, 1000/day on free tier. Also useful for fast Whisper transcription.
3. **OpenRouter** — secondary fallback pool of `:free` models (Llama, Nemotron, etc.), single API key routes to many providers.

Implementation note: wrap the LLM call in a small `llm_router.py` with a uniform interface (`complete(prompt, system, images=None, json_schema=None) -> str`), so swapping/adding providers doesn't touch the rest of the codebase. Catch rate-limit errors specifically and retry against the next provider in the chain.

## 5. Security & privacy requirements

The document-archive feature stores passports, vehicle registration, contracts, receipts — real PII. Non-negotiable requirements:

- Documents and their extracted text live only in the **encrypted local DB**, on a server you control (own VPS), never in a third-party cloud storage bucket.
- OCR for PII documents runs **locally** (Tesseract/PaddleOCR), not through a cloud vision API — the raw image should not leave the server.
- Bot token, encryption keys, and API keys live in `.env`, excluded via `.gitignore`, never hardcoded or committed.
- Non-sensitive OCR (whiteboard photos, code screenshots, class notes) can use cloud vision APIs — this distinction should be explicit in the routing logic, not left to the model's default choice.

## 6. Environment setup (manual, before coding starts)

- Telegram bot token via @BotFather
- Gemini API key — aistudio.google.com (no card required)
- Groq API key — console.groq.com (no card required)
- OpenRouter API key (optional, for the fallback pool)
- Hosting: needs to run persistently — a small VPS (~$5/mo) or Oracle Cloud's free tier (sufficient for this load)

## 7. Development phases

Build and test one feature at a time; each phase should be a separate Claude Code session so diffs stay reviewable.

### Phase 1 — Skeleton + Tutor
- Bot skeleton: aiogram setup, `/start`, message handling, SQLite for chat history
- Harness v1: single LLM call path (Gemini), tutor persona system prompt (see §9 "Tutor"), FSM for step-by-step dialogue
- No tools yet — pure prompting, validates the whole scaffolding end to end

### Phase 2 — Link/video summarizer
- URL detection in incoming messages
- Article path: `trafilatura` extraction → LLM summarization (see §9 "Article summarizer")
- Video path: `yt-dlp` → prefer existing subtitles → fallback to Whisper (Groq or local `faster-whisper`) → LLM summarization (see §9 "Video summarizer")
- Output format: concise summary + key points, sent back within a few seconds where possible; long video transcription should run as a background task with a "processing..." reply first

### Phase 3 — Document archive
- Encrypted storage setup (SQLCipher or AES file encryption)
- Photo upload handler → local OCR (Tesseract) → structured extraction (document type, key fields, see §9 "Document field extraction") → store encrypted
- Query handler: natural-language request ("send me the car registration number") → lookup in encrypted store → return text + original scan

### Phase 4 — Media pipeline extras
- Generalize the downloader: any supported URL → download → convert (mp3 if requested) → LLM-generated timestamps/description
- Screenshot OCR → Markdown/JSON formatting → save into the knowledge base (ties into Phase 5's vector store)

### Phase 5 — RAG knowledge base + spaced repetition
- Vector DB setup (Chroma or sqlite-vec), embedding pipeline for notes/articles
- Notebook photo → vision OCR (handwriting) → chunk → embed → index
- Query flow: question → embed → similarity search → top-N chunks into LLM prompt → answer + reference to original photo
- Flashcard generation: PDF/notes → LLM extracts terms/dates/formulas as structured JSON (see §9 "Flashcard extraction") → SM-2 scheduling
- Daily quiz job via APScheduler: pick due cards, send as Telegram poll/quiz

## 8. Conventions for Claude Code

- Keep tool pipelines as standalone, testable Python modules (`tools/summarizer.py`, `tools/ocr.py`, `tools/downloader.py`, `tools/rag.py`) — the harness only orchestrates, it shouldn't contain pipeline logic inline
- All LLM calls go through `llm_router.py`, never call a provider SDK directly from feature code
- Secrets only via `.env` / environment variables
- Prefer explicit routing logic over "let the LLM decide everything" for anything touching PII or file storage
- Language policy is shared across all prompts (tutor, summarizers, future features) — define it once as a `LANGUAGE_POLICY` constant in `prompts.py` and append it to every system prompt, rather than repeating the wording in each template. Current policy: **Kazakh by default, English if the user writes in English, never Russian.**
- LANGUAGE_POLICY applies to every user-facing string, not just LLM prompts. Static text that never goes through `llm_router.py` — command replies (`/start`, `/help`, `/cancel`, `/reset`, `/tutor`), unsupported-input notices, whitelist/access-rejection messages, rate-limit messages, BotCommand menu descriptions, and any `HarnessError`/`LinkError` message raised by pipeline code — must live in the single `strings.py` module, never inline in handlers or tools. Since these are triggered by non-text input (commands, photos, access checks) with no user text to detect language from, they default to Kazakh, never Russian, no exceptions.

## 9. Prompt templates

These are meant to be used close to verbatim as system prompts (or the instruction part of a single-turn prompt). Anywhere structured output is required, use the provider's JSON-schema / structured-output mode instead of just asking nicely in text — free-tier models are more likely to add stray prose or markdown fences without it.

### Tutor (Phase 1)

```
You are an excellent tutor for a teenager/engineer-level learner.

When explaining a topic:
1. Give the core idea in 2-3 sentences, using one concrete real-world metaphor. No jargon.
2. Give exactly one worked example.
3. End with a single check-question the student must solve themselves. Do not reveal the answer.

Wait for the student's answer before continuing.
- If wrong: point out exactly where the reasoning broke, then give one more example at the same difficulty. Do not just repeat the first explanation.
- If correct: either raise the difficulty slightly or ask if they want to move to a related concept.

Keep responses under ~150 words unless asked for more detail.
Respond in Kazakh by default. If the student's message is in English, respond in English instead. Never respond in Russian, even if the student writes in Russian — understand their intent and answer in Kazakh.
```

### Article summarizer (Phase 2)

```
Summarize the article below for someone who has not read it and has limited time.

Output format:
- One-sentence takeaway.
- 3-6 bullet points with the concrete content (names, numbers, recommendations — not vague generalities).
- If the article makes a claim the author doesn't support with evidence, note it in one line at the end.

Do not editorialize beyond that. Do not pad with "In this article, the author discusses...".
Respond in Kazakh by default, regardless of the article's source language. If the user's own message was in English, respond in English instead. Never respond in Russian.

Article text:
{article_text}
```

### Video summarizer (Phase 2)

```
Below is a transcript (from subtitles or speech-to-text, may contain minor errors) of a video titled "{video_title}".

Produce:
- One-sentence takeaway.
- Key points as a bulleted list, each tagged with an approximate timestamp if timestamps are present in the transcript (format: [mm:ss]).
- If it's a listicle/ranking video (e.g. "top 5 X"), preserve the list as a numbered list with the one-line reason for each item.

Keep it under 200 words unless the video is dense with distinct topics.
Respond in Kazakh by default, regardless of the transcript's language. If the user's own message was in English, respond in English instead. Never respond in Russian.

Transcript:
{transcript_text}
```

### Flashcard extraction (Phase 5)

```
Extract flashcard-worthy facts from the text below: key terms, dates, formulas, definitions, cause-effect relationships.

Return ONLY valid JSON, no prose, no markdown fences:
{"cards": [{"front": "...", "back": "..."}]}

Rules:
- 5-15 cards depending on text density.
- Prefer atomic facts (one fact per card) over broad summaries.
- "front" should be a question or a fill-in-the-blank, not just a restated heading.
- Skip anything too trivial or too context-dependent to make sense as a standalone card.

Text:
{source_text}
```

### Document field extraction (Phase 3)

Important: this step must **not** go through `llm_router.py` / the cloud fallback chain — that would send full PII (IIN, passport number, etc.) to a third-party API in text form, contradicting §5. Run it either as a **local LLM** (e.g. a small model via Ollama) or, more simply, as **per-document-type regex/heuristic parsers** (a passport has a predictable layout, so does a Kazakh vehicle registration) — regex is likely enough here and avoids needing a second local model. Use the prompt below only if you do route it through a local LLM.

Input is Tesseract's raw OCR text, not the image itself — the image never leaves local processing.

```
The text below is raw OCR output from a personal document (passport, vehicle registration, contract, receipt, etc). OCR errors are possible.

Return ONLY valid JSON, no prose, no markdown fences:
{
  "document_type": "...",
  "fields": {"field_name": "value", ...},
  "raw_text": "..."
}

Rules:
- Use field names a person would search for later (e.g. "iin", "document_number", "issue_date", "vin"), not generic labels like "field_1".
- If a field is unreadable or ambiguous due to OCR noise, set its value to null — do not guess.
- "raw_text" is the original OCR text, kept as a fallback for fields not explicitly modeled above.

OCR text:
{ocr_text}
```

