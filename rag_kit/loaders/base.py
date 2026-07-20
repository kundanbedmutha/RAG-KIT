"""Base abstractions for the document-loading stage.

Every loader turns "some source on disk" into a list of `Document` objects.
A `Document` is intentionally dumb: raw text plus metadata. All
segmentation happens later, in the chunking stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """A single loaded source document.

    Attributes:
        text: Full extracted text content of the document.
        metadata: Arbitrary provenance info (source path, page count,
            file type, etc.) that downstream stages may attach to chunks.
        doc_id: Stable identifier for this document, defaults to the
            source path if not explicitly provided.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None

    def __post_init__(self) -> None:
        if self.doc_id is None:
            self.doc_id = self.metadata.get("source", None)


class BaseLoader(ABC):
    """Abstract base class for all document loaders.

    Subclasses implement `load` for a single file path. Loaders are kept
    single-format on purpose (one loader per file type) so that adding
    support for a new format never risks breaking existing ones.
    """

    #: File extensions (lowercase, with leading dot) this loader supports.
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: str | Path) -> list[Document]:
        """Load a single file into one or more `Document` objects.

        Most loaders return exactly one `Document` per file, but a loader
        is free to return more (e.g. one per PDF page) if that is a more
        useful unit for downstream chunking.
        """
        raise NotImplementedError

    def supports(self, path: str | Path) -> bool:
        """Return True if this loader can handle the given file path."""
        return Path(path).suffix.lower() in self.extensions
