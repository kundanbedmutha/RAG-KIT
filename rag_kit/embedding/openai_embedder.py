"""Embedding backend for any OpenAI-compatible `/embeddings` API.

Works against api.openai.com as well as OpenAI-protocol-compatible
endpoints (Azure OpenAI, local servers like LM Studio / vLLM / Ollama's
OpenAI shim, etc.) by letting the base URL be configured.
"""

from __future__ import annotations

import os

import numpy as np

from rag_kit.embedding.base import BaseEmbedder

_DEFAULT_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """Embeds text via an OpenAI-style `/v1/embeddings` HTTP endpoint.

    Args:
        model: Embedding model name, e.g. "text-embedding-3-small".
        api_key: API key. Falls back to the OPENAI_API_KEY env var.
        base_url: API base URL, defaults to OpenAI's own endpoint. Point
            this at a self-hosted OpenAI-compatible server to swap
            providers without touching the rest of the pipeline.
        dimension: Explicit output dimension. Inferred from `model` for
            known OpenAI models; required for unknown/custom models.
        batch_size: Number of texts sent per HTTP request.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        dimension: int | None = None,
        batch_size: int = 100,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

        self._dimension = dimension or _DEFAULT_DIMENSIONS.get(model)
        if self._dimension is None:
            raise ValueError(
                f"Unknown embedding dimension for model {model!r}. "
                "Pass `dimension=` explicitly for custom/non-OpenAI models."
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "OpenAIEmbedder requires 'requests'. "
                "Install with: pip install rag-kit[openai]"
            ) from exc

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": batch},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            # Preserve request order via the `index` field in the response.
            ordered = sorted(payload["data"], key=lambda item: item["index"])
            all_vectors.extend(item["embedding"] for item in ordered)

        return np.asarray(all_vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension
