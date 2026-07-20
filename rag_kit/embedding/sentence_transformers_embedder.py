"""Embedding backend using local sentence-transformers models."""

from __future__ import annotations

import numpy as np

from rag_kit.embedding.base import BaseEmbedder


class SentenceTransformersEmbedder(BaseEmbedder):
    """Embeds text locally using a sentence-transformers model.

    Args:
        model_name: Any model name loadable by `sentence_transformers.
            SentenceTransformer` (e.g. "all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5").
        device: Torch device string, e.g. "cpu", "cuda". Defaults to
            whatever sentence-transformers auto-detects.
        normalize: Whether to L2-normalize embeddings. Recommended when
            the vector store uses inner-product/cosine similarity.
        batch_size: Batch size used when embedding multiple texts.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "SentenceTransformersEmbedder requires 'sentence-transformers'. "
                "Install with: pip install rag-kit[sentence-transformers]"
            ) from exc

        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension
