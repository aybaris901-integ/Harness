"""Reply-language decision (CLAUDE.md §8 LANGUAGE_POLICY), shared by every pipeline.

The policy is applied in code rather than left to the model: English only if
the user's own words are English, anything else -> Kazakh, never Russian.

"The user's own words" excludes things the user didn't write as language: the
leading bot command (`/download`, `/note@MyBot`) and links. Both are Latin
script, so left in they would flip every command to English — the bug behind
`/note` and `/download` replying in English. `user_words` is the single place
that strips them; `reply_language` always goes through it, so callers can pass
the raw message text.
"""

from __future__ import annotations

import re

from tools.urls import strip_urls

# A Telegram bot command at the very start: "/cmd" or "/cmd@BotName".
_COMMAND_RE = re.compile(r"^\s*/[A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)?(?=\s|$)")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def user_words(text: str | None) -> str:
    """`text` minus a leading bot command and any URLs, whitespace-collapsed."""
    if not text:
        return ""
    text = _COMMAND_RE.sub(" ", text, count=1)
    return " ".join(strip_urls(text).split())


def reply_language(user_message: str | None) -> str:
    """Return "English" if the user's own words are English, otherwise "Kazakh"."""
    words = user_words(user_message)
    if _CYRILLIC_RE.search(words):
        return "Kazakh"
    if _LATIN_WORD_RE.search(words):
        return "English"
    return "Kazakh"
