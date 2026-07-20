"""Sparse keyword retrieval strategy using BM25."""

from __future__ import annotations

import re

from rag_kit.chunking.base import Chunk
from rag_kit.retrieval.base import BaseRetriever
from rag_kit.vectorstore.base import ScoredChunk

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """Retrieves chunks via BM25 keyword scoring over their raw text.

    The BM25 index is built once, eagerly, from the chunks passed in.
    Call `rebuild(chunks)` if the underlying chunk set changes (e.g. more
    documents were indexed after this retriever was constructed).

    Args:
        chunks: All chunks to build the keyword index over. Typically
            `vectorstore.all_chunks()`.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "BM25Retriever requires 'rank-bm25'. "
                "Install with: pip install rag-kit[bm25]"
            ) from exc

        self._bm25_cls = BM25Okapi
        self._chunks: list[Chunk] = []
        self._bm25 = None
        self.rebuild(chunks)

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Rebuild the BM25 index from a fresh chunk list."""
        self._chunks = list(chunks)
        tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = self._bm25_cls(tokenized) if tokenized else None

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top_k = min(top_k, len(self._chunks))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in ranked_indices
        ]
