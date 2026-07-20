"""Tests for the retrieval stage: dense, BM25, hybrid fusion, reranking."""

from __future__ import annotations

import numpy as np
import pytest

from rag_kit.chunking.base import Chunk
from rag_kit.retrieval.bm25_retriever import BM25Retriever
from rag_kit.retrieval.dense import DenseRetriever
from rag_kit.retrieval.hybrid import HybridRetriever
from rag_kit.vectorstore.base import ScoredChunk
from rag_kit.vectorstore.faiss_store import FAISSVectorStore

CORPUS = [
    "The Amazon rainforest is home to millions of species of plants and animals.",
    "Python is a popular programming language known for readability.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The stock market fluctuated heavily during the financial crisis.",
    "Guido van Rossum created the Python programming language in 1991.",
]


def _make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(text=t, metadata={"source": f"doc{i}.txt"}, chunk_id=f"chunk_{i}")
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def indexed_store(fake_embedder):
    chunks = _make_chunks(CORPUS)
    embeddings = fake_embedder.embed([c.text for c in chunks])
    store = FAISSVectorStore(dimension=fake_embedder.dimension)
    store.add(chunks, embeddings)
    return store, chunks


class TestDenseRetriever:
    def test_returns_requested_top_k(self, indexed_store, fake_embedder):
        store, _ = indexed_store
        retriever = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        results = retriever.retrieve("Python programming language", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, ScoredChunk) for r in results)

    def test_scores_are_descending(self, indexed_store, fake_embedder):
        store, _ = indexed_store
        retriever = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        results = retriever.retrieve("Python programming language", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_most_relevant_chunk_ranks_first(self, indexed_store, fake_embedder):
        store, _ = indexed_store
        retriever = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        results = retriever.retrieve("Python programming language readability", top_k=1)
        assert "python" in results[0].chunk.text.lower()

    def test_top_k_larger_than_corpus_is_clamped(self, indexed_store, fake_embedder):
        store, _ = indexed_store
        retriever = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        results = retriever.retrieve("plants", top_k=1000)
        assert len(results) == len(CORPUS)

    def test_empty_store_returns_no_results(self, fake_embedder):
        store = FAISSVectorStore(dimension=fake_embedder.dimension)
        retriever = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        assert retriever.retrieve("anything", top_k=5) == []


class TestBM25Retriever:
    def test_keyword_match_ranks_first(self):
        chunks = _make_chunks(CORPUS)
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("Guido van Rossum Python", top_k=1)
        assert "guido" in results[0].chunk.text.lower()

    def test_returns_requested_top_k(self):
        chunks = _make_chunks(CORPUS)
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("plants energy", top_k=2)
        assert len(results) == 2

    def test_rebuild_reflects_new_chunks(self):
        chunks = _make_chunks(CORPUS[:2])
        retriever = BM25Retriever(chunks)
        assert len(retriever.retrieve("stock market crisis", top_k=5)) <= 2

        retriever.rebuild(_make_chunks(CORPUS))
        results = retriever.retrieve("stock market crisis", top_k=1)
        assert "stock market" in results[0].chunk.text.lower()

    def test_empty_corpus_returns_no_results(self):
        retriever = BM25Retriever([])
        assert retriever.retrieve("anything", top_k=5) == []


class TestHybridRetriever:
    def test_rrf_fusion_returns_top_k(self, indexed_store, fake_embedder):
        store, chunks = indexed_store
        dense = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25, method="rrf", fetch_k=5)
        results = hybrid.retrieve("Python programming", top_k=3)
        assert len(results) == 3

    def test_weighted_fusion_returns_top_k(self, indexed_store, fake_embedder):
        store, chunks = indexed_store
        dense = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25, method="weighted", alpha=0.5, fetch_k=5)
        results = hybrid.retrieve("Python programming", top_k=3)
        assert len(results) == 3

    def test_keyword_only_match_still_surfaces_via_bm25_signal(self, indexed_store, fake_embedder):
        # "Guido van Rossum" is a strong exact-keyword signal that BM25
        # should rank highly, contributing to the fused result even if
        # the fake embedder's similarity signal is weak/noisy.
        store, chunks = indexed_store
        dense = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25, method="rrf", fetch_k=5)
        results = hybrid.retrieve("Guido van Rossum", top_k=5)
        top_texts = [r.chunk.text.lower() for r in results[:2]]
        assert any("guido" in t for t in top_texts)

    def test_invalid_method_raises(self, indexed_store, fake_embedder):
        store, chunks = indexed_store
        dense = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        bm25 = BM25Retriever(chunks)
        with pytest.raises(ValueError):
            HybridRetriever(dense, bm25, method="bogus")

    def test_invalid_alpha_raises(self, indexed_store, fake_embedder):
        store, chunks = indexed_store
        dense = DenseRetriever(vectorstore=store, embedder=fake_embedder)
        bm25 = BM25Retriever(chunks)
        with pytest.raises(ValueError):
            HybridRetriever(dense, bm25, alpha=1.5)


class TestCrossEncoderReranker:
    """Reranker logic is tested with a stubbed cross-encoder model so the
    suite never needs to download real weights."""

    def test_rerank_reorders_by_stub_scores(self, monkeypatch):
        import sys
        import types

        from rag_kit.retrieval import reranker as reranker_module

        class StubCrossEncoder:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, pairs):
                # Score higher for pairs where the chunk mentions "python".
                return np.array(
                    [1.0 if "python" in chunk_text.lower() else 0.1 for _, chunk_text in pairs]
                )

        # `CrossEncoderReranker.__init__` does a *lazy* `from
        # sentence_transformers import CrossEncoder`. Rather than importing
        # the real package (which pulls in torch and is slow/heavy, and on
        # some machines fails for unrelated environment reasons -- e.g. a
        # Windows paging-file limit tripping up torch's DLL loading), we
        # inject a fake module into sys.modules so the import inside
        # __init__ resolves to our stub instead of ever touching torch.
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = StubCrossEncoder
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        rr = reranker_module.CrossEncoderReranker(model_name="stub-model")
        candidates = [
            ScoredChunk(chunk=Chunk(text="The stock market crashed."), score=0.9),
            ScoredChunk(chunk=Chunk(text="Python is a great language."), score=0.1),
        ]
        results = rr.rerank("Tell me about Python", candidates, top_k=2)
        assert "python" in results[0].chunk.text.lower()
        assert results[0].score > results[1].score