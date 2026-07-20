"""Base abstraction for the embedding stage."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedding backends.

    Every embedder maps a list of strings to a 2D numpy array of shape
    `(len(texts), dim)`. `dim` must be reported via `.dimension` so vector
    stores can pre-allocate indexes without embedding first.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into vectors.

        Returns:
            A float32 numpy array of shape (len(texts), self.dimension).
        """
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Default: delegate to `embed`.

        Some backends (e.g. instruction-tuned embedding models) use a
        different prefix/instruction for queries vs. documents; those
        backends should override this method.
        """
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors this backend produces."""
        raise NotImplementedError
