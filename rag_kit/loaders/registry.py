"""Extension -> loader dispatch, plus directory-level convenience loading.

This is the only place in the loading stage that knows about "all the
loaders at once" — individual loader modules stay independent of each
other, so adding a new format means adding one class here and nowhere else.
"""

from __future__ import annotations

from pathlib import Path

from rag_kit.loaders.base import BaseLoader, Document
from rag_kit.loaders.markdown_loader import MarkdownLoader
from rag_kit.loaders.pdf_loader import PDFLoader
from rag_kit.loaders.text_loader import TextLoader

#: Default set of loaders, tried in order for each file extension.
DEFAULT_LOADERS: list[BaseLoader] = [
    TextLoader(),
    MarkdownLoader(),
    PDFLoader(),
]


class LoaderRegistry:
    """Resolves file paths to the right loader and loads document sets.

    Example:
        >>> registry = LoaderRegistry()
        >>> docs = registry.load_directory("./my_docs")
    """

    def __init__(self, loaders: list[BaseLoader] | None = None) -> None:
        self.loaders = loaders if loaders is not None else list(DEFAULT_LOADERS)

    def loader_for(self, path: str | Path) -> BaseLoader | None:
        """Return the first loader that supports this file, or None."""
        for loader in self.loaders:
            if loader.supports(path):
                return loader
        return None

    def load_file(self, path: str | Path) -> list[Document]:
        """Load a single file, raising if no loader supports its extension."""
        loader = self.loader_for(path)
        if loader is None:
            raise ValueError(
                f"No loader registered for file extension: {Path(path).suffix!r} "
                f"(path={path})"
            )
        return loader.load(path)

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        skip_unsupported: bool = True,
    ) -> list[Document]:
        """Load every supported file under `directory` into `Document`s.

        Args:
            directory: Root folder to scan.
            recursive: Whether to descend into subdirectories.
            skip_unsupported: If True, files with no matching loader are
                silently skipped. If False, an unsupported file raises.
        """
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        documents: list[Document] = []

        for path in sorted(directory.glob(pattern)):
            if not path.is_file():
                continue
            loader = self.loader_for(path)
            if loader is None:
                if skip_unsupported:
                    continue
                raise ValueError(f"No loader registered for file: {path}")
            documents.extend(loader.load(path))

        return documents
