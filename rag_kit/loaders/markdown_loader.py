"""Loader for Markdown (.md) files.

Kept deliberately simple: markdown is loaded as raw text rather than
rendered to HTML/plain text, because most embedding models handle markdown
syntax (headers, bullets) fine and stripping it loses useful structural
signal that the recursive chunker can split on.
"""

from __future__ import annotations

from pathlib import Path

from rag_kit.loaders.base import BaseLoader, Document


class MarkdownLoader(BaseLoader):
    """Loads Markdown files as a single `Document`."""

    extensions = (".md", ".markdown")

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        text = path.read_text(encoding=self.encoding, errors="replace")
        metadata = {
            "source": str(path),
            "file_type": "markdown",
        }
        return [Document(text=text, metadata=metadata, doc_id=str(path))]
