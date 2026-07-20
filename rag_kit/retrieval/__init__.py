"""Retrieval stage: query -> ranked chunks, with optional hybrid search
and cross-encoder reranking."""

from rag_kit.retrieval.base import BaseRetriever
from rag_kit.retrieval.bm25_retriever import BM25Retriever
from rag_kit.retrieval.dense import DenseRetriever
from rag_kit.retrieval.hybrid import HybridRetriever
from rag_kit.retrieval.reranker import CrossEncoderReranker

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "CrossEncoderReranker",
]
