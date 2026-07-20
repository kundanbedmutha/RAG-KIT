"""Recursive, separator-hierarchy-aware chunking strategy.

Tries to split on "big" boundaries first (paragraphs), falling back to
progressively smaller ones (sentences, words, characters) only for pieces
that are still too large. This generally produces chunks that respect
natural document structure better than a naive fixed-size window.
"""

from __future__ import annotations

from rag_kit.chunking.base import BaseChunker

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """Splits text on a hierarchy of separators, merging back up to size.

    Args:
        chunk_size: Target maximum size of each chunk, in characters.
        chunk_overlap: Number of trailing characters from one chunk to
            prepend to the next, for retrieval-context continuity.
        separators: Ordered list of separators to try, from coarsest
            (e.g. "\\n\\n" for paragraphs) to finest (e.g. "" for a hard
            character split as a last resort).
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be in [0, chunk_size)"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else list(DEFAULT_SEPARATORS)

    def chunk_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        pieces = self._split(text, self.separators)
        merged = self._merge(pieces)
        return [m.strip() for m in merged if m.strip()]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split `text` until every piece is <= chunk_size."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Last resort: hard character split.
            return [
                text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
            ]

        sep, rest_separators = separators[0], separators[1:]
        parts = text.split(sep) if sep else list(text)

        results: list[str] = []
        for part in parts:
            if sep:
                part_with_sep = part + sep
            else:
                part_with_sep = part
            if len(part_with_sep) > self.chunk_size:
                results.extend(self._split(part_with_sep, rest_separators))
            elif part_with_sep.strip():
                results.append(part_with_sep)
        return results

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily merge small split pieces back up to ~chunk_size."""
        if not pieces:
            return []

        merged: list[str] = []
        current = ""
        for piece in pieces:
            candidate = current + piece
            if len(candidate) <= self.chunk_size or not current:
                current = candidate
            else:
                merged.append(current)
                overlap_tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = overlap_tail + piece
        if current.strip():
            merged.append(current)
        return merged
