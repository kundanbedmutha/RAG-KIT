"""Embedding stage: turns chunk text into dense vectors."""

from rag_kit.embedding.base import BaseEmbedder
from rag_kit.embedding.openai_embedder import OpenAIEmbedder
from rag_kit.embedding.sentence_transformers_embedder import (
    SentenceTransformersEmbedder,
)

#: String-keyed registry so `RAGPipeline` can resolve an embedder by name.
EMBEDDER_REGISTRY: dict[str, type[BaseEmbedder]] = {
    "sentence-transformers": SentenceTransformersEmbedder,
    "openai": OpenAIEmbedder,
}


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    """Instantiate a registered embedder by name.

    Example:
        >>> embedder = get_embedder("sentence-transformers", model_name="all-MiniLM-L6-v2")
    """
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(
            f"Unknown embedder {name!r}. Available: {list(EMBEDDER_REGISTRY)}"
        )
    return EMBEDDER_REGISTRY[name](**kwargs)


__all__ = [
    "BaseEmbedder",
    "SentenceTransformersEmbedder",
    "OpenAIEmbedder",
    "EMBEDDER_REGISTRY",
    "get_embedder",
]
