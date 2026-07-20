"""Base abstractions for the chunking stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_kit.loaders.base import Document


@dataclass
class Chunk:
    """A single retrievable unit produced from a `Document`.

    Attributes:
        text: The chunk's text content.
        metadata: Copied/augmented from the parent document, plus
            chunk-specific info (chunk_index, char_start, char_end).
        chunk_id: Stable identifier, `{doc_id}::chunk_{index}` by default.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies.

    A chunker takes documents in, chunks out. Chunk size/overlap semantics
    are strategy-specific (char count for `FixedSizeChunker`, but a
    subclass is free to define "size" in tokens or sentences).
    """

    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        """Split raw text into a list of chunk strings.

        This is the one method concrete strategies must implement; the
        Document-aware `chunk_document`/`chunk_documents` wrappers below
        are implemented once here and reused by every strategy.
        """
        raise NotImplementedError

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Chunk a single `Document`, attaching provenance metadata."""
        pieces = self.chunk_text(document.text)
        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = i
            metadata["doc_id"] = document.doc_id
            chunks.append(
                Chunk(
                    text=piece,
                    metadata=metadata,
                    chunk_id=f"{document.doc_id}::chunk_{i}",
                )
            )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a list of `Document`s into a flat list of `Chunk`s."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
