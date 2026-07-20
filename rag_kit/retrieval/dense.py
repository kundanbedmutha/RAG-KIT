"""Pure dense (embedding-similarity) retrieval strategy."""

from __future__ import annotations

from rag_kit.embedding.base import BaseEmbedder
from rag_kit.retrieval.base import BaseRetriever
from rag_kit.vectorstore.base import BaseVectorStore, ScoredChunk


class DenseRetriever(BaseRetriever):
    """Retrieves chunks via nearest-neighbor search in embedding space.

    Args:
        vectorstore: A populated `BaseVectorStore` to search against.
        embedder: The same embedder used to build the vector store (must
            match, or query/document vectors won't be comparable).
    """

    def __init__(self, vectorstore: BaseVectorStore, embedder: BaseEmbedder) -> None:
        self.vectorstore = vectorstore
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        query_embedding = self.embedder.embed_query(query)
        return self.vectorstore.search(query_embedding, top_k=top_k)
