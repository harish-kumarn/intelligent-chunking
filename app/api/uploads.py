"""
app.api.uploads
===============
`stage_upload` — stream an uploaded file to disk, never holding it in memory.

    stream : 1 MiB reads into `<workspace root>/_uploads/<uuid>.pdf`, hashing (SHA-256) as it goes
    limit  : the upload is rejected the moment it passes `max_bytes`, not after it was read whole

The staged file is handed to the channel, which moves it into the document's workspace
(`Channel.admit`). Whatever is left in the staging directory at startup belonged to requests a
restart cut off and is removed (`clear_staged_uploads`).
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.orchestration.errors import InvalidDocumentError

__all__ = ["StagedUpload", "clear_staged_uploads", "stage_upload"]

_READ_SIZE = 1024 * 1024


@dataclass(frozen=True)
class StagedUpload:
    path: Path
    filename: str
    sha256: str
    size: int

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


def stage_upload(stream: BinaryIO, filename: str, directory: Path, max_bytes: int) -> StagedUpload:
    """Copy `stream` to a new file in `directory`. Raises InvalidDocumentError when it exceeds `max_bytes`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.pdf"
    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "wb") as out:
            while block := stream.read(_READ_SIZE):
                size += len(block)
                if size > max_bytes:
                    raise InvalidDocumentError(f"File exceeds the {max_bytes}-byte upload limit")
                digest.update(block)
                out.write(block)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return StagedUpload(path=path, filename=filename, sha256=digest.hexdigest(), size=size)


def clear_staged_uploads(directory: Path) -> None:
    """Startup only: no request is in flight, so anything staged was abandoned."""
    shutil.rmtree(directory, ignore_errors=True)
