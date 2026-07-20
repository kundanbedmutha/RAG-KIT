"""Base abstraction for the retrieval stage."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_kit.vectorstore.base import ScoredChunk


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies.

    A retriever takes a query string in, ranked `ScoredChunk`s out. This
    is deliberately separate from `BaseVectorStore`: a vector store only
    knows how to do dense similarity search, whereas a retriever can
    compose multiple signals (dense, BM25, both fused) and optionally
    hand results to a reranker before returning them.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the top_k most relevant chunks for `query`."""
        raise NotImplementedError
