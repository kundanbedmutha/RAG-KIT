"""Hybrid (dense + keyword) retrieval via score/rank fusion."""

from __future__ import annotations

from rag_kit.retrieval.base import BaseRetriever
from rag_kit.retrieval.bm25_retriever import BM25Retriever
from rag_kit.retrieval.dense import DenseRetriever
from rag_kit.vectorstore.base import ScoredChunk


class HybridRetriever(BaseRetriever):
    """Combines a `DenseRetriever` and a `BM25Retriever` via fusion.

    Args:
        dense_retriever: The dense (embedding similarity) retriever.
        bm25_retriever: The sparse (keyword) retriever.
        method: Fusion method — "rrf" (Reciprocal Rank Fusion, the
            default; scale-free and robust when dense/BM25 scores live on
            very different numeric ranges) or "weighted" (min-max
            normalize each side's scores, then blend with `alpha`).
        alpha: Only used when method="weighted". Weight given to the
            dense retriever's normalized score; BM25 gets `1 - alpha`.
        rrf_k: Only used when method="rrf". Standard RRF damping
            constant; higher values reduce the influence of rank
            position (60 is the commonly used default in IR literature).
        fetch_k: How many candidates to pull from *each* underlying
            retriever before fusing, typically larger than the final
            `top_k` passed to `retrieve` so fusion has enough to work with.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        method: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        fetch_k: int = 20,
    ) -> None:
        if method not in ("rrf", "weighted"):
            raise ValueError(f"method must be 'rrf' or 'weighted', got {method!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.method = method
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        dense_results = self.dense_retriever.retrieve(query, top_k=self.fetch_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=self.fetch_k)

        if self.method == "rrf":
            fused = self._reciprocal_rank_fusion(dense_results, bm25_results)
        else:
            fused = self._weighted_fusion(dense_results, bm25_results)

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[ScoredChunk],
        bm25_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        combined_scores: dict[str, float] = {}
        chunk_by_id = {}

        for rank, result in enumerate(dense_results):
            key = result.chunk.chunk_id or result.chunk.text
            chunk_by_id[key] = result.chunk
            combined_scores[key] = combined_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, result in enumerate(bm25_results):
            key = result.chunk.chunk_id or result.chunk.text
            chunk_by_id[key] = result.chunk
            combined_scores[key] = combined_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(combined_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [ScoredChunk(chunk=chunk_by_id[key], score=score) for key, score in ranked]

    def _weighted_fusion(
        self,
        dense_results: list[ScoredChunk],
        bm25_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        dense_norm = self._min_max_normalize(dense_results)
        bm25_norm = self._min_max_normalize(bm25_results)

        combined_scores: dict[str, float] = {}
        chunk_by_id = {}

        for key, (chunk, score) in dense_norm.items():
            chunk_by_id[key] = chunk
            combined_scores[key] = combined_scores.get(key, 0.0) + self.alpha * score

        for key, (chunk, score) in bm25_norm.items():
            chunk_by_id[key] = chunk
            combined_scores[key] = combined_scores.get(key, 0.0) + (1 - self.alpha) * score

        ranked = sorted(combined_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [ScoredChunk(chunk=chunk_by_id[key], score=score) for key, score in ranked]

    @staticmethod
    def _min_max_normalize(results: list[ScoredChunk]) -> dict[str, tuple]:
        if not results:
            return {}
        scores = [r.score for r in results]
        lo, hi = min(scores), max(scores)
        span = hi - lo if hi > lo else 1.0
        normalized = {}
        for r in results:
            key = r.chunk.chunk_id or r.chunk.text
            normalized[key] = (r.chunk, (r.score - lo) / span)
        return normalized
