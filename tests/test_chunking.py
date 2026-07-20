"""Tests for the chunking stage: size limits, overlap, edge cases."""

from __future__ import annotations

import pytest

from rag_kit.chunking import FixedSizeChunker, RecursiveChunker
from rag_kit.chunking.base import Chunk
from rag_kit.loaders.base import Document

LONG_TEXT = " ".join(f"word{i}" for i in range(500))  # ~3500 chars


class TestFixedSizeChunker:
    def test_respects_chunk_size_char_unit(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20, unit="char")
        chunks = chunker.chunk_text(LONG_TEXT)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 100

    def test_overlap_produces_shared_content(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=30, unit="char")
        chunks = chunker.chunk_text(LONG_TEXT)
        # Consecutive chunks should share a suffix/prefix due to overlap.
        assert chunks[0][-10:] in chunks[1] or chunks[1][:10] in chunks[0]

    def test_word_unit_respects_word_count(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2, unit="word")
        chunks = chunker.chunk_text(LONG_TEXT)
        for c in chunks:
            assert len(c.split()) <= 10

    def test_no_content_loss_word_unit(self):
        # Every original word should appear in at least one chunk.
        chunker = FixedSizeChunker(chunk_size=15, chunk_overlap=3, unit="word")
        chunks = chunker.chunk_text(LONG_TEXT)
        all_words_in_chunks = set(" ".join(chunks).split())
        original_words = set(LONG_TEXT.split())
        assert original_words.issubset(all_words_in_chunks)

    def test_empty_text_returns_no_chunks(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   \n  ") == []

    def test_short_text_returns_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_text("short text")
        assert chunks == ["short text"]

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=150)

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=0, chunk_overlap=0)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=10, unit="sentence")


class TestRecursiveChunker:
    def test_respects_chunk_size_roughly(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_text(LONG_TEXT)
        assert len(chunks) > 1
        # Recursive merge can slightly exceed target size for a single
        # unsplittable unit, but should never wildly blow past it.
        for c in chunks:
            assert len(c) <= 130

    def test_splits_on_paragraph_boundaries_when_possible(self):
        text = "Paragraph one has some words.\n\nParagraph two has other words.\n\nParagraph three is here."
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=5)
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2

    def test_empty_text_returns_no_chunks(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        assert chunker.chunk_text("") == []

    def test_short_text_returns_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_text("short text")
        assert chunks == ["short text"]

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)


class TestChunkerDocumentIntegration:
    def test_chunk_document_attaches_metadata(self):
        doc = Document(
            text=LONG_TEXT,
            metadata={"source": "fake.txt", "file_type": "txt"},
            doc_id="fake.txt",
        )
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(doc)

        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.metadata["doc_id"] == "fake.txt" for c in chunks)
        assert all(c.metadata["source"] == "fake.txt" for c in chunks)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_chunk_id_is_stable_and_unique(self):
        doc = Document(text=LONG_TEXT, metadata={"source": "fake.txt"}, doc_id="fake.txt")
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # all unique
        assert all(cid.startswith("fake.txt::chunk_") for cid in ids)

    def test_chunk_documents_flattens_multiple_docs(self):
        docs = [
            Document(text=LONG_TEXT, metadata={"source": "a.txt"}, doc_id="a.txt"),
            Document(text=LONG_TEXT, metadata={"source": "b.txt"}, doc_id="b.txt"),
        ]
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents(docs)
        doc_ids_seen = {c.metadata["doc_id"] for c in chunks}
        assert doc_ids_seen == {"a.txt", "b.txt"}
