"""
app.orchestration.orchestrator
==============================
ChannelOrchestrator — maps a `source_type` to its Channel and dispatches.

A channel works on a source file already on disk (the API streams uploads there):

    validate        : reject what the channel cannot process, before any paid work
    admit           : move the source into the document's workspace (idempotent); returns its path there
    admitted_source : the admitted source of an earlier run, if its workspace still holds it (restart recovery)
    process         : run the channel's pipeline on an admitted source

Adding a source type means registering another Channel; nothing in the API layer or the
pipeline changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from app.orchestration.errors import UnsupportedSourceTypeError

__all__ = ["Channel", "ChannelOrchestrator"]


@runtime_checkable
class Channel(Protocol):
    source_type: str

    def validate(self, filename: str, source: Path) -> None:
        """Raise a `ChannelError` if the file is not processable by this channel."""

    def admit(self, *, document_id: str, source: Path, content_hash: str) -> Path:
        """Move `source` into the document's workspace (consuming it); return the admitted path."""

    def admitted_source(self, *, document_id: str, content_hash: str) -> Optional[Path]:
        """The admitted source for this document and content, or None when its workspace is gone."""

    def process(self, *, document_id: str, source: Path, content_hash: str, **kwargs: Any) -> Any:
        """Run the channel's pipeline on an admitted source and return its result."""


class ChannelOrchestrator:
    def __init__(self) -> None:
        self._channels: Dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        self._channels[channel.source_type] = channel

    def resolve(self, source_type: str) -> Channel:
        try:
            return self._channels[source_type]
        except KeyError as exc:
            raise UnsupportedSourceTypeError(f"No channel registered for source_type={source_type!r}") from exc

    def dispatch(self, source_type: str, *, document_id: str, filename: str, source: Path, content_hash: str, **kwargs: Any) -> Any:
        """validate ─► admit ─► process, in one call (the synchronous path)."""
        channel = self.resolve(source_type)
        channel.validate(filename, source)
        admitted = channel.admit(document_id=document_id, source=source, content_hash=content_hash)
        return channel.process(document_id=document_id, source=admitted, content_hash=content_hash, **kwargs)
