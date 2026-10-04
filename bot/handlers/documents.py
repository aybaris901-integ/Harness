"""Document archive (CLAUDE.md §7 Phase 3).

Any photo lands here — OCR and field extraction run entirely locally (§5),
and the scan plus extracted fields are stored encrypted, keyed per Telegram
user. `/find <query>` searches that store and returns the best match:
extracted fields (or, for an "unknown" document with no fields, the matching
lines of its OCR text) plus the original scan. `/delete <id>` removes one of
the user's own documents — the PII redirects from /note and /page can land a
false positive here.
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, Message

import strings
from harness import DocumentArchive, DocumentError
from storage.documents import DocumentMatch, DocumentRecord

logger = logging.getLogger(__name__)

router = Router(name="documents")


def _type_label(document_type: str) -> str:
    return strings.DOCUMENT_TYPE_LABELS.get(document_type, document_type)


def format_document_saved(record: DocumentRecord) -> str:
    """Shared by every handler that may store a photo in the archive (/note, /page).
    Always ends with the /delete hint: the PII redirects can land a false
    positive here, and the id is the only handle the user has on it."""
    doc_type = _type_label(record.document_type)
    if record.fields:
        lines = [
            strings.DOCUMENT_SAVED.format(id=record.id, type=doc_type, count=len(record.fields))
        ]
        lines.extend(f"• {name}: {value}" for name, value in record.fields.items())
    else:
        lines = [strings.DOCUMENT_SAVED_TEXT_ONLY.format(id=record.id, type=doc_type)]
    lines.append(strings.DOCUMENT_DELETE_HINT.format(id=record.id))
    return "\n".join(lines)


def _format_match(match: DocumentMatch) -> str:
    record = match.record
    lines = [
        strings.DOCUMENT_MATCH_HEADER.format(id=record.id, type=_type_label(record.document_type))
    ]
    for name, value in record.fields.items():
        marker = "➡️ " if name in match.matched_fields else "• "
        lines.append(f"{marker}{name}: {value}")
    if match.matched_lines:
        lines.append(strings.DOCUMENT_MATCH_TEXT_LINES)
        lines.extend(f"➡️ {line}" for line in match.matched_lines)
    return "\n".join(lines)


@router.message(F.photo)
async def handle_photo(message: Message, archive: DocumentArchive) -> None:
    assert message.from_user is not None and message.bot is not None
    status = await message.answer(strings.DOCUMENT_PROCESSING)

    # Telegram sends the same photo at several resolutions, smallest first —
    # the last entry is the highest-resolution one, best for OCR.
    photo = message.photo[-1]
    buffer = await message.bot.download(photo)
    image_bytes = buffer.read()

    try:
        record = await archive.ingest(telegram_id=message.from_user.id, image_bytes=image_bytes)
    except DocumentError as exc:
        await status.edit_text(str(exc))
        return
    except Exception:
        logger.exception("Unexpected failure ingesting a document photo")
        await status.edit_text(strings.DOCUMENT_GENERIC_ERROR)
        return

    await status.edit_text(format_document_saved(record))


@router.message(Command("find"))
async def handle_find(message: Message, command: CommandObject, archive: DocumentArchive) -> None:
    assert message.from_user is not None
    query = (command.args or "").strip()
    if not query:
        await message.answer(strings.DOCUMENT_FIND_USAGE)
        return

    try:
        match = await archive.search(telegram_id=message.from_user.id, query=query)
    except DocumentError as exc:
        await message.answer(str(exc))
        return
    except Exception:
        logger.exception("Unexpected failure searching documents")
        await message.answer(strings.DOCUMENT_GENERIC_ERROR)
        return

    if match is None:
        await message.answer(strings.DOCUMENT_NOT_FOUND)
        return

    await message.answer(_format_match(match))
    try:
        scan_bytes = await archive.load_scan(match.record)
    except DocumentError as exc:
        await message.answer(str(exc))
        return
    await message.answer_photo(
        BufferedInputFile(scan_bytes, filename=f"{match.record.document_type}.jpg")
    )


@router.message(Command("delete"))
async def handle_delete(message: Message, command: CommandObject, archive: DocumentArchive) -> None:
    """`/delete <id>` — remove one of the sender's own archived documents (row
    + encrypted scan). Scoped by Telegram user id in the store's SQL, so an id
    belonging to someone else behaves exactly like a missing one."""
    assert message.from_user is not None
    raw = (command.args or "").strip().lstrip("#")
    if not raw.isdigit():
        await message.answer(strings.DOCUMENT_DELETE_USAGE)
        return
    document_id = int(raw)

    try:
        record = await archive.delete(telegram_id=message.from_user.id, document_id=document_id)
    except DocumentError as exc:
        await message.answer(str(exc))
        return
    except Exception:
        logger.exception("Unexpected failure deleting a document")
        await message.answer(strings.DOCUMENT_GENERIC_ERROR)
        return

    if record is None:
        await message.answer(strings.DOCUMENT_DELETE_NOT_FOUND.format(id=document_id))
        return
    await message.answer(
        strings.DOCUMENT_DELETED.format(id=document_id, type=_type_label(record.document_type))
    )
