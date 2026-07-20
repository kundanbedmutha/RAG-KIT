"""Loader for plain .txt files."""

from __future__ import annotations

from pathlib import Path

from rag_kit.loaders.base import BaseLoader, Document


class TextLoader(BaseLoader):
    """Loads plain-text files as a single `Document`."""

    extensions = (".txt",)

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        text = path.read_text(encoding=self.encoding, errors="replace")
        metadata = {
            "source": str(path),
            "file_type": "txt",
        }
        return [Document(text=text, metadata=metadata, doc_id=str(path))]
