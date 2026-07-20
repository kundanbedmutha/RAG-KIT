"""Chunking stage: turns `Document`s into a flat list of `Chunk`s."""

from rag_kit.chunking.base import BaseChunker, Chunk
from rag_kit.chunking.fixed_size import FixedSizeChunker
from rag_kit.chunking.recursive import RecursiveChunker

#: String-keyed registry so `RAGPipeline` can resolve a chunker by name.
CHUNKER_REGISTRY: dict[str, type[BaseChunker]] = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
}


def get_chunker(name: str, **kwargs) -> BaseChunker:
    """Instantiate a registered chunker by name.

    Example:
        >>> chunker = get_chunker("recursive", chunk_size=300, chunk_overlap=30)
    """
    if name not in CHUNKER_REGISTRY:
        raise ValueError(
            f"Unknown chunker {name!r}. Available: {list(CHUNKER_REGISTRY)}"
        )
    return CHUNKER_REGISTRY[name](**kwargs)


__all__ = [
    "BaseChunker",
    "Chunk",
    "FixedSizeChunker",
    "RecursiveChunker",
    "CHUNKER_REGISTRY",
    "get_chunker",
]
