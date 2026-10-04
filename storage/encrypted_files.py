"""Fernet-encrypted files on local disk (CLAUDE.md §5).

One `<uuid>.enc` file per blob, AES-128-CBC + HMAC-SHA256 via Fernet, with
the key from `.env` (DOCUMENTS_ENCRYPTION_KEY) — never logged. Shared by the
document archive (scans) and the knowledge base (notebook page photos), so
there is one implementation of "encrypted at rest" in the codebase.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


class EncryptedFileError(RuntimeError):
    """A file could not be decrypted (wrong/rotated key, corruption) or is missing."""


class EncryptedFileStore:
    def __init__(self, directory: Path, encryption_key: bytes) -> None:
        self.directory = directory
        self._fernet = Fernet(encryption_key)

    def ensure_dir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, filename: str) -> Path:
        # Filenames are always ours (uuid hex + ".enc"); refuse anything else
        # so a stored value can never point outside the directory.
        if Path(filename).name != filename or not filename.endswith(".enc"):
            raise EncryptedFileError(f"invalid encrypted filename {filename!r}")
        return self.directory / filename

    async def write(self, data: bytes) -> str:
        """Encrypt `data` into a new file; return its filename (not the full path)."""
        filename = f"{uuid.uuid4().hex}.enc"
        encrypted = self._fernet.encrypt(data)
        await asyncio.to_thread(self.path_for(filename).write_bytes, encrypted)
        return filename

    async def read(self, filename: str) -> bytes:
        path = self.path_for(filename)
        try:
            encrypted = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise EncryptedFileError(f"encrypted file {filename} is missing") from exc
        try:
            return self._fernet.decrypt(encrypted)
        except InvalidToken as exc:
            raise EncryptedFileError(
                f"could not decrypt {filename} — wrong DOCUMENTS_ENCRYPTION_KEY?"
            ) from exc

    async def delete(self, filename: str) -> None:
        await asyncio.to_thread(self.path_for(filename).unlink, True)
