"""Fixed-size sliding-window chunking strategy."""

from __future__ import annotations

from rag_kit.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Splits text into fixed-size windows with configurable overlap.

    Size and overlap are measured in characters by default. Set
    `unit="word"` to measure in whitespace-split words instead, which
    tends to produce more semantically stable chunk boundaries for
    languages that tokenize well on whitespace.

    Args:
        chunk_size: Target size of each chunk (in the configured unit).
        chunk_overlap: Number of units of overlap between consecutive
            chunks. Must be strictly less than chunk_size.
        unit: Either "char" or "word".

    Raises:
        ValueError: If chunk_overlap >= chunk_size, or chunk_size <= 0.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        unit: str = "char",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size}), or windows never advance"
            )
        if unit not in ("char", "word"):
            raise ValueError(f"unit must be 'char' or 'word', got {unit!r}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit

    def chunk_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        if self.unit == "char":
            return self._chunk_units(text, list(text), joiner="")
        # unit == "word"
        words = text.split()
        return self._chunk_units(text, words, joiner=" ")

    def _chunk_units(self, original_text: str, units: list[str], joiner: str) -> list[str]:
        if not units:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        i = 0
        n = len(units)
        while i < n:
            window = units[i : i + self.chunk_size]
            piece = joiner.join(window)
            stripped = piece.strip()
            if stripped:
                chunks.append(stripped)
            if i + self.chunk_size >= n:
                break
            i += step
        return chunks
