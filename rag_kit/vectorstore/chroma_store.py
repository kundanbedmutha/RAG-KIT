"""Chroma-backed vector store.

Unlike FAISS, Chroma natively stores documents + metadata alongside
vectors and assigns string ids, so this implementation leans on Chroma's
own storage rather than keeping a parallel chunk list.
"""

from __future__ import annotations

import json

import numpy as np

from rag_kit.chunking.base import Chunk
from rag_kit.vectorstore.base import BaseVectorStore, ScoredChunk


class ChromaVectorStore(BaseVectorStore):
    """Vector store backed by a Chroma collection.

    Args:
        collection_name: Name of the Chroma collection to use/create.
        persist_directory: If set, uses a persistent on-disk Chroma
            client rooted at this path. If None, uses an in-memory
            (ephemeral) client — convenient for tests and quickstarts.
        distance: Chroma's `hnsw:space` setting, e.g. "cosine", "l2", "ip".
    """

    def __init__(
        self,
        collection_name: str = "rag_kit",
        persist_directory: str | None = None,
        distance: str = "cosine",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "ChromaVectorStore requires 'chromadb'. "
                "Install with: pip install rag-kit[chroma]"
            ) from exc

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )
        self._next_id = self._collection.count()

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of "
                f"embeddings ({embeddings.shape[0]})"
            )
        if not chunks:
            return

        ids = [f"chunk_{self._next_id + i}" for i in range(len(chunks))]
        # Chroma requires flat, JSON-scalar metadata values; nested dicts
        # or None values are serialized to strings to stay compatible.
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks]
        documents = [c.text for c in chunks]

        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents,
        )
        self._next_id += len(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[ScoredChunk]:
        if len(self) == 0:
            return []
        top_k = min(top_k, len(self))
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        scored: list[ScoredChunk] = []
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        # Chroma returns *distances*; convert to a similarity-style score
        # (higher = more similar) so callers get consistent semantics
        # regardless of vector store backend.
        distances = result["distances"][0]
        ids = result["ids"][0]

        for doc, meta, dist, id_ in zip(docs, metas, distances, ids):
            chunk = Chunk(text=doc, metadata=dict(meta), chunk_id=id_)
            scored.append(ScoredChunk(chunk=chunk, score=1.0 - dist))
        return scored

    def all_chunks(self) -> list[Chunk]:
        raw = self._collection.get()
        chunks = []
        for doc, meta, id_ in zip(raw["documents"], raw["metadatas"], raw["ids"]):
            chunks.append(Chunk(text=doc, metadata=dict(meta), chunk_id=id_))
        return chunks

    def __len__(self) -> int:
        return self._collection.count()

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        clean = {}
        for key, value in metadata.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = json.dumps(value)
        return clean

    def config(self) -> dict:
        return {
            "backend": "chroma",
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }
