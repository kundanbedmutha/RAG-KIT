"""Vector store stage: stores chunk embeddings and answers similarity search."""

from rag_kit.vectorstore.base import BaseVectorStore, ScoredChunk
from rag_kit.vectorstore.chroma_store import ChromaVectorStore
from rag_kit.vectorstore.faiss_store import FAISSVectorStore

#: String-keyed registry so `RAGPipeline` can resolve a vector store by name.
VECTORSTORE_REGISTRY: dict[str, type[BaseVectorStore]] = {
    "faiss": FAISSVectorStore,
    "chroma": ChromaVectorStore,
}


def get_vectorstore(name: str, **kwargs) -> BaseVectorStore:
    """Instantiate a registered vector store by name.

    Example:
        >>> store = get_vectorstore("faiss", dimension=384)
    """
    if name not in VECTORSTORE_REGISTRY:
        raise ValueError(
            f"Unknown vector store {name!r}. Available: {list(VECTORSTORE_REGISTRY)}"
        )
    return VECTORSTORE_REGISTRY[name](**kwargs)


__all__ = [
    "BaseVectorStore",
    "ScoredChunk",
    "FAISSVectorStore",
    "ChromaVectorStore",
    "VECTORSTORE_REGISTRY",
    "get_vectorstore",
]
