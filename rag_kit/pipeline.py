"""The `RAGPipeline` class: wires loading, chunking, embedding, vector
storage, retrieval, and reranking into a single configurable object.

Typical usage:

    pipeline = RAGPipeline(
        embedder="sentence-transformers",
        embedder_kwargs={"model_name": "all-MiniLM-L6-v2"},
        vectorstore="faiss",
        chunker="recursive",
        chunker_kwargs={"chunk_size": 500, "chunk_overlap": 50},
        retrieval_mode="hybrid",
        rerank=True,
    )
    pipeline.index_directory("./my_docs")
    results = pipeline.query("What is the main topic of these documents?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_kit.chunking import BaseChunker, get_chunker
from rag_kit.chunking.base import Chunk
from rag_kit.embedding import BaseEmbedder, get_embedder
from rag_kit.loaders import Document, LoaderRegistry
from rag_kit.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRetriever
from rag_kit.vectorstore import BaseVectorStore, ScoredChunk, get_vectorstore

_VALID_RETRIEVAL_MODES = ("dense", "hybrid")


class RAGPipeline:
    """End-to-end, backend-swappable Retrieval-Augmented Generation pipeline.

    The pipeline is intentionally "just" the retrieval half of RAG — it
    indexes documents and returns relevant chunks for a query. Generation
    (calling an LLM with the retrieved context) is left to the caller via
    `build_prompt`, so RAG-Kit stays usable regardless of which LLM
    provider you generate with.

    Args:
        embedder: Registry key for the embedding backend
            ("sentence-transformers" or "openai").
        embedder_kwargs: Keyword args forwarded to the embedder constructor.
        vectorstore: Registry key for the vector store backend
            ("faiss" or "chroma").
        vectorstore_kwargs: Keyword args forwarded to the vector store
            constructor. `dimension` is filled in automatically from the
            embedder if not provided (FAISS requires it up front).
        chunker: Registry key for the chunking strategy
            ("fixed_size" or "recursive").
        chunker_kwargs: Keyword args forwarded to the chunker constructor.
        retrieval_mode: "dense" for pure embedding similarity, or
            "hybrid" to additionally fuse in BM25 keyword search.
        hybrid_kwargs: Keyword args forwarded to `HybridRetriever`
            (method, alpha, rrf_k, fetch_k), only used when
            retrieval_mode="hybrid".
        rerank: Whether to apply cross-encoder reranking to retrieved
            candidates before returning them.
        reranker_kwargs: Keyword args forwarded to `CrossEncoderReranker`.
        rerank_fetch_k: How many candidates to retrieve before reranking
            down to the requested `top_k`. Only relevant when rerank=True.
    """

    def __init__(
        self,
        embedder: str | BaseEmbedder = "sentence-transformers",
        embedder_kwargs: dict[str, Any] | None = None,
        vectorstore: str | BaseVectorStore = "faiss",
        vectorstore_kwargs: dict[str, Any] | None = None,
        chunker: str | BaseChunker = "recursive",
        chunker_kwargs: dict[str, Any] | None = None,
        retrieval_mode: str = "dense",
        hybrid_kwargs: dict[str, Any] | None = None,
        rerank: bool = False,
        reranker_kwargs: dict[str, Any] | None = None,
        rerank_fetch_k: int = 20,
    ) -> None:
        if retrieval_mode not in _VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of {_VALID_RETRIEVAL_MODES}, "
                f"got {retrieval_mode!r}"
            )

        # --- Embedder ---
        self.embedder: BaseEmbedder = (
            embedder if isinstance(embedder, BaseEmbedder) else get_embedder(embedder, **(embedder_kwargs or {}))
        )

        # --- Vector store (needs the embedder's dimension for FAISS) ---
        vs_kwargs = dict(vectorstore_kwargs or {})
        if isinstance(vectorstore, str):
            vs_kwargs.setdefault("dimension", self.embedder.dimension)
            self.vectorstore: BaseVectorStore = get_vectorstore(vectorstore, **vs_kwargs)
        else:
            self.vectorstore = vectorstore

        # --- Chunker ---
        self.chunker: BaseChunker = (
            chunker if isinstance(chunker, BaseChunker) else get_chunker(chunker, **(chunker_kwargs or {}))
        )

        # --- Loader registry (fixed: txt/md/pdf, not user-configurable
        # via string since there's no meaningful "swap" here — every
        # format needs its own loader regardless) ---
        self.loader_registry = LoaderRegistry()

        # --- Retrieval mode ---
        self.retrieval_mode = retrieval_mode
        self.hybrid_kwargs = hybrid_kwargs or {}
        self._bm25_retriever: BM25Retriever | None = None

        # --- Reranking ---
        self.rerank_enabled = rerank
        self.rerank_fetch_k = rerank_fetch_k
        self.reranker: CrossEncoderReranker | None = (
            CrossEncoderReranker(**(reranker_kwargs or {})) if rerank else None
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[Document]) -> int:
        """Chunk, embed, and store a list of already-loaded `Document`s.

        Returns:
            The number of chunks added to the vector store.
        """
        chunks = self.chunker.chunk_documents(documents)
        return self._index_chunks(chunks)

    def index_directory(self, directory: str | Path, recursive: bool = True) -> int:
        """Load every supported file under `directory`, then index it.

        Returns:
            The number of chunks added to the vector store.
        """
        documents = self.loader_registry.load_directory(directory, recursive=recursive)
        return self.index_documents(documents)

    def index_file(self, path: str | Path) -> int:
        """Load a single file, then index it.

        Returns:
            The number of chunks added to the vector store.
        """
        documents = self.loader_registry.load_file(path)
        return self.index_documents(documents)

    def index_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """Index raw strings directly, bypassing the loader stage.

        Useful for programmatically-generated content that doesn't live
        in files (e.g. scraped web pages already held in memory).
        """
        metadatas = metadatas or [{} for _ in texts]
        documents = [
            Document(text=t, metadata=m, doc_id=m.get("source", f"text_{i}"))
            for i, (t, m) in enumerate(zip(texts, metadatas))
        ]
        return self.index_documents(documents)

    def _index_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.vectorstore.add(chunks, embeddings)
        # Invalidate any cached BM25 index so it gets rebuilt against the
        # now-larger corpus the next time hybrid retrieval is used.
        self._bm25_retriever = None
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _build_retriever(self):
        dense = DenseRetriever(vectorstore=self.vectorstore, embedder=self.embedder)
        if self.retrieval_mode == "dense":
            return dense

        # hybrid: (re)build the BM25 index lazily, cached until the next
        # indexing call invalidates it.
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever(self.vectorstore.all_chunks())
        return HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=self._bm25_retriever,
            **self.hybrid_kwargs,
        )

    def query(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Retrieve the `top_k` most relevant chunks for `query`.

        If reranking is enabled, `rerank_fetch_k` candidates are first
        retrieved by the configured retrieval mode, then reranked down
        to `top_k` by the cross-encoder.
        """
        retriever = self._build_retriever()

        if self.rerank_enabled and self.reranker is not None:
            fetch_k = max(self.rerank_fetch_k, top_k)
            candidates = retriever.retrieve(query, top_k=fetch_k)
            return self.reranker.rerank(query, candidates, top_k=top_k)

        return retriever.retrieve(query, top_k=top_k)

    def build_prompt(
        self,
        query: str,
        top_k: int = 5,
        template: str | None = None,
    ) -> str:
        """Retrieve context for `query` and format it into an LLM prompt.

        This is a convenience formatting step, not a call to any LLM —
        RAG-Kit stops at "here is your retrieved context and question,"
        so it composes with whichever generation provider you use.

        Args:
            template: A format string with `{context}` and `{question}`
                placeholders. A sensible default is used if omitted.
        """
        results = self.query(query, top_k=top_k)
        context = "\n\n".join(
            f"[{i + 1}] {r.chunk.text}" for i, r in enumerate(results)
        )
        template = template or (
            "Answer the question using only the context below. "
            "If the answer isn't in the context, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return template.format(context=context, question=query)

    def __len__(self) -> int:
        return len(self.vectorstore)
