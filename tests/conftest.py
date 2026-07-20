"""Shared pytest fixtures.

Tests use `FakeEmbedder`, a tiny deterministic hashing-based embedder,
instead of `SentenceTransformersEmbedder`/`OpenAIEmbedder`. This keeps
the test suite fast and network-free while still exercising the exact
same `BaseEmbedder` interface the real pipeline relies on -- a fake that
implements the real ABC is a more faithful test double than a mock.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from rag_kit.embedding.base import BaseEmbedder

FIXTURES_DIR = "tests/fixtures"


class FakeEmbedder(BaseEmbedder):
    """Deterministic bag-of-words-ish embedder requiring no network/model.

    Produces vectors where similar word content yields similar vectors,
    by hashing each token into a fixed-size vector and summing +
    normalizing. This is NOT a real embedding model -- it exists purely
    so retrieval-ordering tests are meaningful without downloading
    weights in a sandboxed environment.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                idx = h % self._dim
                sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
                vectors[i, idx] += sign
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=64)


@pytest.fixture
def toy_documents():
    from rag_kit.loaders import LoaderRegistry

    return LoaderRegistry().load_directory(FIXTURES_DIR)
