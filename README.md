# RAG-Kit

A modular, swappable-backend Retrieval-Augmented Generation (RAG) pipeline for Python.
Drop in a folder of documents, get back ranked, relevant chunks for any query — without
rewriting the plumbing every time you start a new project.

## Problem statement

Most RAG prototypes end up as a single throwaway script: a loader function, a chunking
loop, an embedding call, and a vector store, all hardcoded together. The moment you want
to try a different embedding model, swap FAISS for Chroma, or add hybrid search, you're
rewriting the script from scratch.

RAG-Kit separates the RAG pipeline into five independent, pluggable stages —
**loading, chunking, embedding, vector storage, and retrieval (+ optional reranking)** —
each behind a small abstract interface. Swapping a backend is a one-line config change,
not a rewrite:

```python
# Swap embedder + vector store without touching anything else
pipeline = RAGPipeline(embedder="openai", vectorstore="chroma")
```

RAG-Kit deliberately stops at retrieval. It doesn't call an LLM for you — `build_prompt()`
formats retrieved context into a prompt string, and you generate with whichever provider
you already use. That keeps RAG-Kit useful regardless of which LLM you're building on top of.

## Install

```bash
# Core install (loaders, chunkers, pipeline scaffolding)
pip install rag-kit

# Pick the backends you actually need
pip install "rag-kit[sentence-transformers]"   # local embeddings
pip install "rag-kit[openai]"                  # OpenAI-style embeddings API
pip install "rag-kit[faiss]"                   # FAISS vector store
pip install "rag-kit[chroma]"                  # Chroma vector store
pip install "rag-kit[bm25]"                    # hybrid search (BM25)
pip install "rag-kit[rerank]"                  # cross-encoder reranking

# Or everything at once
pip install "rag-kit[all]"

# From source, editable, with test dependencies
git clone https://github.com/kundanbedmutha/RAG-KIT.git
cd rag-kit
pip install -e ".[all,dev]"
```

## Quickstart

```python
from rag_kit import RAGPipeline

# 1. Configure the pipeline: pick your embedder, vector store, chunker.
pipeline = RAGPipeline(
    embedder="sentence-transformers",
    embedder_kwargs={"model_name": "all-MiniLM-L6-v2"},
    vectorstore="faiss",
    chunker="recursive",
    chunker_kwargs={"chunk_size": 500, "chunk_overlap": 50},
)

# 2. Index a document set — .txt, .md, .pdf all supported, mixed freely.
pipeline.index_directory("./my_docs")

# 3. Query it.
results = pipeline.query("What are the main findings of the report?", top_k=5)
for r in results:
    print(round(r.score, 3), r.chunk.metadata["source"], "->", r.chunk.text[:100])

# 4. Or go straight to an LLM-ready prompt.
prompt = pipeline.build_prompt("What are the main findings of the report?", top_k=5)
# ... pass `prompt` to your LLM of choice (Anthropic, OpenAI, local, etc.)
```

### Turning on hybrid search + reranking

```python
pipeline = RAGPipeline(
    embedder="sentence-transformers",
    vectorstore="faiss",
    retrieval_mode="hybrid",              # dense + BM25, fused via Reciprocal Rank Fusion
    hybrid_kwargs={"method": "rrf", "fetch_k": 20},
    rerank=True,                          # cross-encoder reranking of the fused candidates
    reranker_kwargs={"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
)
pipeline.index_directory("./my_docs")
results = pipeline.query("exact phrase or keyword-heavy query", top_k=5)
```

### Swapping backends

Every stage is resolved from a string key at construction time — nothing else in your
code needs to change when you swap:

```python
# Local embeddings + FAISS (fully offline)
RAGPipeline(embedder="sentence-transformers", vectorstore="faiss")

# OpenAI-style embeddings API + Chroma (persistent, on disk)
RAGPipeline(
    embedder="openai",
    embedder_kwargs={"model": "text-embedding-3-small"},
    vectorstore="chroma",
    vectorstore_kwargs={"persist_directory": "./chroma_db"},
)
```

## Architecture

```
                     ┌──────────────────────────────────────────────────┐
                     │                   RAGPipeline                    │
                     │        (wires everything below via config)       │
                     └──────────────────────────────────────────────────┘
                                            │
        ┌────────────┬────────────┬────────┴────────┬────────────────┐
        ▼            ▼            ▼                 ▼                ▼
   ┌─────────┐  ┌──────────┐  ┌──────────┐   ┌──────────────┐  ┌────────────┐
   │ Loading │─▶│ Chunking │─▶│Embedding │──▶│ Vector Store │─▶│ Retrieval  │
   └─────────┘  └──────────┘  └──────────┘   └──────────────┘  └────────────┘
   .txt/.md/    fixed_size /  sentence-       FAISS / Chroma    dense / hybrid
   .pdf files   recursive    transformers /                     (dense+BM25),
                             OpenAI-style                        + optional
                                                                  reranking
```

**Data flow:** `Document` (raw text + metadata) → `Chunk` (segmented text + provenance) →
embedding vector (attached to the chunk in the vector store) → `ScoredChunk` (chunk +
relevance score, returned by retrieval).

**Stage contracts** (each is an ABC in `rag_kit/<stage>/base.py`):

| Stage       | Interface                                   | Built-in implementations                          |
|-------------|----------------------------------------------|----------------------------------------------------|
| Loading     | `BaseLoader.load(path) -> list[Document]`     | `TextLoader`, `MarkdownLoader`, `PDFLoader`         |
| Chunking    | `BaseChunker.chunk_text(text) -> list[str]`   | `FixedSizeChunker`, `RecursiveChunker`              |
| Embedding   | `BaseEmbedder.embed(texts) -> np.ndarray`     | `SentenceTransformersEmbedder`, `OpenAIEmbedder`    |
| Vector store| `BaseVectorStore.add/search`                  | `FAISSVectorStore`, `ChromaVectorStore`             |
| Retrieval   | `BaseRetriever.retrieve(query) -> list[ScoredChunk]` | `DenseRetriever`, `BM25Retriever`, `HybridRetriever` |
| Reranking   | `CrossEncoderReranker.rerank(query, candidates)` | cross-encoder (optional, composes with any retriever) |

Adding a new backend to any stage means writing one class against its ABC and adding one
line to that stage's registry (`<stage>/__init__.py`) — the pipeline and every other stage
are untouched.

## Extending RAG-Kit

```python
from rag_kit.chunking.base import BaseChunker

class SentenceWindowChunker(BaseChunker):
    def chunk_text(self, text: str) -> list[str]:
        # your custom splitting logic
        ...

pipeline = RAGPipeline(chunker=SentenceWindowChunker())  # pass an instance directly
```

Every `RAGPipeline` constructor argument accepts either a registry string
(`"faiss"`, `"recursive"`, ...) or an already-constructed instance of the corresponding
ABC, so custom backends compose with the rest of the pipeline exactly like the built-ins.

## Running tests

```bash
pip install -e ".[dev,bm25]"
pytest
```

Tests cover chunking correctness (size limits, overlap, no content loss), retrieval
relevance ordering (dense, BM25, hybrid fusion, reranking), and a full pipeline
end-to-end pass over a toy document set. The suite uses a tiny deterministic embedder
double so it runs fully offline — no model downloads required.

## License

MIT