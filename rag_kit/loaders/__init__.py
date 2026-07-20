"""Document loading stage: turns files on disk into `Document` objects."""

from rag_kit.loaders.base import BaseLoader, Document
from rag_kit.loaders.markdown_loader import MarkdownLoader
from rag_kit.loaders.pdf_loader import PDFLoader
from rag_kit.loaders.registry import LoaderRegistry
from rag_kit.loaders.text_loader import TextLoader

__all__ = [
    "BaseLoader",
    "Document",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "LoaderRegistry",
]
