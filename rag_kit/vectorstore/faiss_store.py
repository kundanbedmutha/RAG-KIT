"""FAISS-backed vector store.

FAISS itself only stores vectors and returns integer ids, so this class
keeps a parallel `list[Chunk]` indexed by insertion order and maps FAISS
ids straight back to it.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from rag_kit.chunking.base import Chunk
from rag_kit.vectorstore.base import BaseVectorStore, ScoredChunk


class FAISSVectorStore(BaseVectorStore):
    """Vector store backed by a FAISS index.

    Args:
        dimension: Dimensionality of the vectors to be stored.
        metric: Either "cosine" (implemented as inner product over
            L2-normalized vectors) or "l2" (Euclidean distance).
    """

    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "FAISSVectorStore requires 'faiss-cpu'. "
                "Install with: pip install rag-kit[faiss]"
            ) from exc

        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")

        self._faiss = faiss
        self.dimension = dimension
        self.metric = metric
        self._index = (
            faiss.IndexFlatIP(dimension) if metric == "cosine" else faiss.IndexFlatL2(dimension)
        )
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of "
                f"embeddings ({embeddings.shape[0]})"
            )
        embeddings = self._prepare(embeddings)
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[ScoredChunk]:
        if len(self) == 0:
            return []
        query = self._prepare(query_embedding.reshape(1, -1))
        top_k = min(top_k, len(self))
        scores, indices = self._index.search(query, top_k)

        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(ScoredChunk(chunk=self._chunks[idx], score=float(score)))
        return results

    def _prepare(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        return embeddings

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def persist(self, path: str) -> None:
        """Persist the FAISS index and chunk list to `path` (a directory)."""
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(out_dir / "index.faiss"))
        with open(out_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    @classmethod
    def load(cls, path: str, dimension: int, metric: str = "cosine") -> "FAISSVectorStore":
        """Load a previously persisted FAISS store from `path`."""
        store = cls(dimension=dimension, metric=metric)
        in_dir = Path(path)
        store._index = store._faiss.read_index(str(in_dir / "index.faiss"))
        with open(in_dir / "chunks.pkl", "rb") as f:
            store._chunks = pickle.load(f)
        return store

    def config(self) -> dict:
        return {"backend": "faiss", "dimension": self.dimension, "metric": self.metric}
