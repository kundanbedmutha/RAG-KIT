"""Loader for PDF files, backed by pypdf.

By default this returns ONE `Document` per PDF (all pages concatenated),
with per-page boundaries preserved via `\\f` (form-feed) markers so a
downstream chunker can optionally split on page boundaries. Set
`one_document_per_page=True` to instead get one `Document` per page,
which is useful when you want retrieval results traceable to an exact
page number.
"""

from __future__ import annotations

from pathlib import Path

from rag_kit.loaders.base import BaseLoader, Document


class PDFLoader(BaseLoader):
    """Loads PDF files, extracting text per page via pypdf."""

    extensions = (".pdf",)

    def __init__(self, one_document_per_page: bool = False) -> None:
        self.one_document_per_page = one_document_per_page

    def load(self, path: str | Path) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "PDFLoader requires 'pypdf'. Install with: pip install rag-kit"
            ) from exc

        path = Path(path)
        reader = PdfReader(str(path))
        page_texts = [page.extract_text() or "" for page in reader.pages]

        if self.one_document_per_page:
            docs = []
            for i, page_text in enumerate(page_texts):
                metadata = {
                    "source": str(path),
                    "file_type": "pdf",
                    "page": i + 1,
                    "num_pages": len(page_texts),
                }
                docs.append(
                    Document(text=page_text, metadata=metadata, doc_id=f"{path}#page={i + 1}")
                )
            return docs

        full_text = "\f".join(page_texts)
        metadata = {
            "source": str(path),
            "file_type": "pdf",
            "num_pages": len(page_texts),
        }
        return [Document(text=full_text, metadata=metadata, doc_id=str(path))]
