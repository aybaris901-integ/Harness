"""Model Markdown -> Telegram HTML (parse_mode="HTML"), shared by every handler
that shows LLM or note content: /summarize (and plain links), /download
descriptions, /note.

Why HTML and not MarkdownV2: MarkdownV2 needs ~18 characters escaped outside
entities and rejects the whole message on one miss, and model output is full
of `.`, `-`, `(`. HTML needs only `& < >` escaped.

The converter works block by block (a line, a fenced code block, a run of
quote lines), and every block renders to balanced HTML on its own. Chunks are
packed from whole blocks, so splitting for Telegram's length limits never cuts
through a tag. Inline formatting that would come out unbalanced (e.g.
`***x***`) falls back to the escaped plain line. If Telegram still rejects a
chunk, the send helpers below retry it as plain text.
"""

from __future__ import annotations

import html
import re

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from bot.utils import TELEGRAM_MESSAGE_LIMIT, split_message

# Telegram's caption limit (photos, documents, videos, audio).
TELEGRAM_CAPTION_LIMIT = 1024
PARSE_MODE = "HTML"

_FENCE_RE = re.compile(r"^\s*```\s*([\w+-]*)\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_RULE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")

_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__")
_ITALIC_RE = re.compile(
    r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])|(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])"
)
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_TAG_RE = re.compile(r"<(/?)([a-z]+)[^>]*>")
_PLACEHOLDER = "\x00{}\x00"
_PLACEHOLDER_RE = re.compile("\x00(\\d+)\x00")

# HTML escaping can grow text up to 5x ("&" -> "&amp;"); pieces of an oversized
# block are cut this small so each still fits once rendered.
_EXPANSION = 5


def escape(text: str) -> str:
    return html.escape(text, quote=False)


def to_plain(rendered: str) -> str:
    """Rendered HTML back to plain text — the fallback when Telegram rejects it."""
    return html.unescape(_TAG_RE.sub("", rendered))


def _balanced(rendered: str) -> bool:
    stack: list[str] = []
    for match in _TAG_RE.finditer(rendered):
        closing, name = match.groups()
        if not closing:
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def _inline(text: str) -> str:
    stash: list[str] = []

    def keep(rendered: str) -> str:
        stash.append(rendered)
        return _PLACEHOLDER.format(len(stash) - 1)

    # Code spans and links are rendered first and hidden from the emphasis
    # regexes, so `a_b` in code or "_" in a URL never turns into italics.
    text = _CODE_SPAN_RE.sub(lambda m: keep(f"<code>{escape(m.group(1))}</code>"), text)
    text = _LINK_RE.sub(
        lambda m: keep(f'<a href="{html.escape(m.group(2), quote=True)}">{escape(m.group(1))}</a>'),
        text,
    )
    out = escape(text)
    out = _BOLD_RE.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", out)
    out = _ITALIC_RE.sub(lambda m: f"<i>{m.group(1) or m.group(2)}</i>", out)
    out = _STRIKE_RE.sub(lambda m: f"<s>{m.group(1)}</s>", out)
    out = _PLACEHOLDER_RE.sub(lambda m: stash[int(m.group(1))], out)
    if not _balanced(out):
        return _PLACEHOLDER_RE.sub(lambda m: to_plain(stash[int(m.group(1))]), escape(text))
    return out


def _line(line: str) -> str:
    if _RULE_RE.match(line):
        return "———"
    if heading := _HEADING_RE.match(line):
        # Strip bold markers inside the heading: it's bold as a whole already.
        return f"<b>{_inline(heading.group(1).replace('**', ''))}</b>"
    if bullet := _BULLET_RE.match(line):
        indent = "  " * (len(bullet.group(1).expandtabs(4)) // 2)
        return f"{indent}• {_inline(bullet.group(2))}"
    return _inline(line)


def _code_blocks(code: str, lang: str, limit: int) -> list[str]:
    open_tag = f'<pre><code class="language-{lang}">' if lang else "<pre>"
    close_tag = "</code></pre>" if lang else "</pre>"
    budget = max(1, (limit - len(open_tag) - len(close_tag)) // _EXPANSION)
    return [f"{open_tag}{escape(piece)}{close_tag}" for piece in split_message(code, budget)]


def _text_blocks(source: str, render, limit: int) -> list[str]:
    rendered = render(source)
    if len(rendered) <= limit:
        return [rendered]
    return [render(piece) for piece in split_message(source, max(1, limit // _EXPANSION))]


def _blocks(text: str, *, verbatim: bool, limit: int) -> list[str]:
    lines = text.splitlines()
    if verbatim:
        blocks: list[str] = []
        for line in lines:
            blocks.extend(_text_blocks(line, escape, limit))
        return blocks

    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if fence := _FENCE_RE.match(line):
            body: list[str] = []
            i += 1
            while i < len(lines) and not _FENCE_RE.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # closing fence (or end of text if the model never closed it)
            blocks.extend(_code_blocks("\n".join(body) or " ", fence.group(1), limit))
            continue
        if _QUOTE_RE.match(line):
            quoted: list[str] = []
            while i < len(lines) and (quote := _QUOTE_RE.match(lines[i])):
                quoted.append(quote.group(1))
                i += 1
            inner_limit = limit - len("<blockquote></blockquote>")
            for piece in _text_blocks(
                "\n".join(quoted),
                lambda src: "\n".join(_inline(q) for q in src.splitlines()),
                inner_limit,
            ):
                blocks.append(f"<blockquote>{piece}</blockquote>")
            continue
        blocks.extend(_text_blocks(line, _line, limit))
        i += 1
    return blocks


def render_chunks(
    text: str,
    *,
    header: str = "",
    verbatim: bool = False,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    """Render `text` (model Markdown, or plain text with `verbatim=True`) to
    Telegram HTML chunks of at most `limit` characters each. `header` is plain
    text, escaped and shown in bold on top."""
    blocks = _blocks(text.strip("\n"), verbatim=verbatim, limit=limit)
    if header:
        blocks = [*_text_blocks(header, lambda h: f"<b>{escape(h)}</b>", limit), "", *blocks]

    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current.strip():
            chunks.append(current.strip("\n"))
        current = block
    if current.strip():
        chunks.append(current.strip("\n"))
    return chunks or [escape(text[:limit]) or "…"]


# -- sending -------------------------------------------------------------------


async def answer_html(message: Message, chunk: str) -> Message:
    try:
        return await message.answer(chunk, parse_mode=PARSE_MODE)
    except TelegramBadRequest:
        return await message.answer(to_plain(chunk), parse_mode=None)


async def edit_html(bot: Bot, status: Message, chunk: str) -> None:
    """Replace a status message with `chunk`; send a new message if it can't be edited."""
    for text, mode in ((chunk, PARSE_MODE), (to_plain(chunk), None)):
        try:
            await bot.edit_message_text(
                text, chat_id=status.chat.id, message_id=status.message_id, parse_mode=mode
            )
            return
        except TelegramBadRequest:
            continue
    try:
        await bot.send_message(status.chat.id, chunk, parse_mode=PARSE_MODE)
    except TelegramBadRequest:
        await bot.send_message(status.chat.id, to_plain(chunk), parse_mode=None)
