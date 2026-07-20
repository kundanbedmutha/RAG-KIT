"""End-to-end tests: index a toy document set, then query it.

Uses the FakeEmbedder fixture so the full loader -> chunker -> embedder
-> vectorstore -> retriever chain is exercised without any network
access or real model downloads.
"""

from __future__ import annotations

import pytest

from rag_kit.pipeline import RAGPipeline

FIXTURES_DIR = "tests/fixtures"


def make_pipeline(fake_embedder, **kwargs):
    """Build a RAGPipeline wired to the fake embedder + FAISS store."""
    defaults = dict(
        embedder=fake_embedder,
        vectorstore="faiss",
        vectorstore_kwargs={"dimension": fake_embedder.dimension},
        chunker="recursive",
        chunker_kwargs={"chunk_size": 200, "chunk_overlap": 20},
    )
    defaults.update(kwargs)
    return RAGPipeline(**defaults)


class TestIndexing:
    def test_index_directory_returns_chunk_count(self, fake_embedder):
        pipeline = make_pipeline(fake_embedder)
        n_chunks = pipeline.index_directory(FIXTURES_DIR)
        assert n_chunks > 0
        assert len(pipeline) == n_chunks

    def test_index_file_adds_to_existing_store(self, fake_embedder):
        pipeline = make_pipeline(fake_embedder)
        n1 = pipeline.index_file(f"{FIXTURES_DIR}/doc1.txt")
        n2 = pipeline.index_file(f"{FIXTURES_DIR}/doc2.md")
        assert len(pipeline) == n1 + n2

    def test_index_texts_bypasses_loaders(self, fake_embedder):
        pipeline = make_pipeline(fake_embedder)
        n_chunks = pipeline.index_texts(
            ["A short standalone note about cats.", "Another note about dogs."]
        )
        assert n_chunks == 2
        assert len(pipeline) == 2

    def test_empty_document_list_indexes_nothing(self, fake_embedder):
        pipeline = make_pipeline(fake_embedder)
        assert pipeline.index_documents([]) == 0
        assert len(pipeline) == 0


class TestDenseQuery:
    @pytest.fixture
    def pipeline(self, fake_embedder):
        p = make_pipeline(fake_embedder, retrieval_mode="dense")
        p.index_directory(FIXTURES_DIR)
        return p

    def test_query_returns_top_k_results(self, pipeline):
        results = pipeline.query("Who created the Python language?", top_k=2)
        assert len(results) == 2

    def test_query_surfaces_relevant_chunk(self, pipeline):
        results = pipeline.query("Guido van Rossum Python programming", top_k=1)
        assert "python" in results[0].chunk.text.lower()

    def test_query_result_has_source_metadata(self, pipeline):
        results = pipeline.query("rainforest trees Brazil", top_k=1)
        assert "source" in results[0].chunk.metadata
        assert results[0].chunk.metadata["source"].endswith("doc1.txt")


class TestHybridQuery:
    @pytest.fixture
    def pipeline(self, fake_embedder):
        p = make_pipeline(
            fake_embedder,
            retrieval_mode="hybrid",
            hybrid_kwargs={"method": "rrf", "fetch_k": 10},
        )
        p.index_directory(FIXTURES_DIR)
        return p

    def test_hybrid_query_returns_top_k(self, pipeline):
        results = pipeline.query("photosynthesis light energy", top_k=3)
        assert len(results) == 3

    def test_hybrid_surfaces_keyword_match(self, pipeline):
        results = pipeline.query("Calvin cycle stroma", top_k=1)
        assert "calvin" in results[0].chunk.text.lower()

    def test_bm25_index_rebuilds_after_reindexing(self, fake_embedder):
        pipeline = make_pipeline(
            fake_embedder, retrieval_mode="hybrid", hybrid_kwargs={"fetch_k": 10}
        )
        pipeline.index_file(f"{FIXTURES_DIR}/doc1.txt")
        pipeline.query("rainforest", top_k=1)  # builds BM25 cache
        pipeline.index_file(f"{FIXTURES_DIR}/doc3.txt")  # should invalidate cache
        results = pipeline.query("Guido van Rossum", top_k=1)
        assert "guido" in results[0].chunk.text.lower()


class TestBuildPrompt:
    def test_prompt_contains_context_and_question(self, fake_embedder):
        pipeline = make_pipeline(fake_embedder)
        pipeline.index_directory(FIXTURES_DIR)
        prompt = pipeline.build_prompt("What is photosynthesis?", top_k=2)
        assert "What is photosynthesis?" in prompt
        assert "Context:" in prompt
        assert "[1]" in prompt

    def test_custom_template_is_respected(self, fake_embedder):
        pipeline = make_pipeline(fake_embedder)
        pipeline.index_directory(FIXTURES_DIR)
        prompt = pipeline.build_prompt(
            "test query",
            top_k=1,
            template="CTX={context}||Q={question}",
        )
        assert prompt.startswith("CTX=")
        assert "||Q=test query" in prompt


class TestInvalidConfig:
    def test_invalid_retrieval_mode_raises(self, fake_embedder):
        with pytest.raises(ValueError):
            make_pipeline(fake_embedder, retrieval_mode="bogus")
