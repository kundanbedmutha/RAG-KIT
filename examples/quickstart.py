"""Quickstart example for RAG-Kit.

Run with:
    pip install -e ".[sentence-transformers,faiss,bm25]"
    python examples/quickstart.py

This indexes the toy document set under tests/fixtures/ and runs a few
example queries in both dense and hybrid retrieval modes.
"""

from __future__ import annotations

from rag_kit import RAGPipeline

FIXTURES_DIR = "tests/fixtures"


def main() -> None:
    print("=== Dense retrieval ===")
    pipeline = RAGPipeline(
        embedder="sentence-transformers",
        embedder_kwargs={"model_name": "all-MiniLM-L6-v2"},
        vectorstore="faiss",
        chunker="recursive",
        chunker_kwargs={"chunk_size": 300, "chunk_overlap": 30},
    )
    n_chunks = pipeline.index_directory(FIXTURES_DIR)
    print(f"Indexed {n_chunks} chunks from {FIXTURES_DIR}\n")

    for query in [
        "Who created the Python programming language?",
        "How does photosynthesis work?",
    ]:
        print(f"Query: {query}")
        for r in pipeline.query(query, top_k=2):
            print(f"  ({r.score:.3f}) [{r.chunk.metadata.get('source')}] {r.chunk.text[:90]}...")
        print()

    print("=== Hybrid retrieval + reranking ===")
    hybrid_pipeline = RAGPipeline(
        embedder="sentence-transformers",
        embedder_kwargs={"model_name": "all-MiniLM-L6-v2"},
        vectorstore="faiss",
        retrieval_mode="hybrid",
        hybrid_kwargs={"method": "rrf", "fetch_k": 10},
        rerank=True,
        reranker_kwargs={"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    )
    hybrid_pipeline.index_directory(FIXTURES_DIR)

    query = "Calvin cycle stroma light reactions"
    print(f"Query: {query}")
    for r in hybrid_pipeline.query(query, top_k=2):
        print(f"  ({r.score:.3f}) [{r.chunk.metadata.get('source')}] {r.chunk.text[:90]}...")

    print("\n=== Prompt formatting for an LLM ===")
    prompt = pipeline.build_prompt("What is the Amazon rainforest?", top_k=2)
    print(prompt)


if __name__ == "__main__":
    main()
