"""Document archive (CLAUDE.md §7 Phase 3).

Any photo lands here — OCR and field extraction run entirely locally (§5),
and the scan plus extracted fields are stored encrypted, keyed per Telegram
user. `/find <query>` searches that store and returns the best match:
extracted fields plus the original scan.
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


def _format_saved(record: DocumentRecord) -> str:
    header = strings.DOCUMENT_SAVED.format(
        type=_type_label(record.document_type), count=len(record.fields)
    )
    lines = [header]
    lines.extend(f"• {name}: {value}" for name, value in record.fields.items())
    return "\n".join(lines)


def _format_match(match: DocumentMatch) -> str:
    lines = [strings.DOCUMENT_MATCH_HEADER.format(type=_type_label(match.record.document_type))]
    for name, value in match.record.fields.items():
        marker = "➡️ " if name in match.matched_fields else "• "
        lines.append(f"{marker}{name}: {value}")
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

    await status.edit_text(_format_saved(record))


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
