"""Cross-encoder reranking step.

Applied *after* a retriever (dense, BM25, or hybrid) has already
narrowed the corpus down to a small candidate set. A cross-encoder
jointly attends over (query, chunk) pairs, which is far more accurate
than embedding-similarity but too slow to run over an entire corpus —
hence "retrieve broad, then rerank narrow."
"""

from __future__ import annotations

from rag_kit.vectorstore.base import ScoredChunk


class CrossEncoderReranker:
    """Reranks a candidate list of `ScoredChunk`s using a cross-encoder.

    This is a standalone, optional post-processing step rather than a
    `BaseRetriever` itself — `RAGPipeline` calls it on a retriever's
    output only when reranking is enabled, so it's a one-line toggle to
    turn on/off without changing the retrieval strategy underneath.

    Args:
        model_name: Any cross-encoder model loadable by
            `sentence_transformers.CrossEncoder`, e.g.
            "cross-encoder/ms-marco-MiniLM-L-6-v2".
        device: Torch device string. Defaults to auto-detected.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "CrossEncoderReranker requires 'sentence-transformers'. "
                "Install with: pip install rag-kit[rerank]"
            ) from exc

        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Rerank `candidates` for `query`, returning the top `top_k`.

        Args:
            query: The original query string.
            candidates: Chunks to rerank, typically the output of a
                `BaseRetriever.retrieve` call with a generous `top_k`.
            top_k: Number of chunks to return after reranking. Defaults
                to returning all candidates, just reordered.
        """
        if not candidates:
            return []

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)

        reranked = [
            ScoredChunk(chunk=c.chunk, score=float(score))
            for c, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda sc: sc.score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked
