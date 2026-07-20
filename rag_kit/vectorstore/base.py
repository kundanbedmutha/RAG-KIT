"""Base abstraction for the vector store stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from rag_kit.chunking.base import Chunk


@dataclass
class ScoredChunk:
    """A `Chunk` paired with its similarity score from a vector search."""

    chunk: Chunk
    score: float


class BaseVectorStore(ABC):
    """Abstract base class for all vector store backends.

    A vector store owns two parallel pieces of data: the embedding
    vectors (for similarity search) and the corresponding `Chunk` objects
    (for returning human-readable results). Implementations are free to
    keep these together (Chroma) or separate (FAISS + a side list/dict).
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their pre-computed embeddings to the store."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[ScoredChunk]:
        """Return the `top_k` most similar chunks to `query_embedding`."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Number of chunks currently stored."""
        raise NotImplementedError

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk, e.g. for building a BM25 index.

        Default implementation raises; concrete stores that can cheaply
        enumerate their contents should override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support enumerating all chunks"
        )

    def persist(self, path: str) -> None:
        """Optionally persist the store to disk. No-op by default."""
        _ = path  # not all backends need this

    def config(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of this store's config."""
        return {}
