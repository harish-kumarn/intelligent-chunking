"""
app.utils.fs
============
Crash-safe file writes and moves.

Every artifact a resumed run may reuse is written through here: the content goes to a hidden
temp file beside the target and is renamed over it, so a crash leaves either the old file or
the new one — never a truncated file that a later run would mistake for finished work.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_bytes", "atomic_write_json", "atomic_write_text", "move_file"]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with open(temp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=indent))


def move_file(source: Path, target: Path) -> None:
    """Move `source` to `target` atomically; across filesystems it is copied to a temp name, then renamed."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, target)
        return
    except OSError:  # EXDEV: different filesystem
        pass
    temp = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temp)
    os.replace(temp, target)
    source.unlink(missing_ok=True)
