"""Doctest configuration."""

import importlib
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from pathlib import Path

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType

from sybil import Document, Region, Sybil, document as sybil_document
from sybil.evaluators.doctest import DocTestEvaluator
from sybil.parsers import rest
from sybil.parsers.abstract.doctest import DocTestStringParser
from sybil.parsers.myst import (
    CodeBlockParser,
    DocTestDirectiveParser,
    PythonCodeBlockParser,
    SkipParser as MystSkipParser,
)
from sybil.python import import_path as sybil_import_path

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE


CX_HYPOTHESIS_ROOT = (
    Path(__file__).parent
    / "packages"
    / "coordinax.hypothesis"
    / "src"
    / "coordinax"
    / "hypothesis"
)


def _import_path_with_namespace(path: Path) -> ModuleType:
    """Import coordinax.hypothesis files using their fully-qualified module path."""
    if path.is_relative_to(CX_HYPOTHESIS_ROOT):
        relative = path.relative_to(CX_HYPOTHESIS_ROOT)
        if relative.name == "__init__.py":
            parts = tuple(relative.parts[:-1])
        else:
            parts = (*tuple(relative.parts[:-1]), relative.stem)

        module = ".".join(("coordinax", "hypothesis", *parts))
        return importlib.import_module(module)

    return sybil_import_path(path)


sybil_document.import_path = _import_path_with_namespace


class PyconCodeBlockParser:
    """Parser for MyST pycon code blocks with doctest evaluation."""

    def __init__(self, doctest_optionflags: int = 0) -> None:
        """Initialize the pycon code block parser.

        Parameters
        ----------
        doctest_optionflags : int
            Doctest option flags (e.g., ELLIPSIS, NORMALIZE_WHITESPACE).

        """
        self.doctest_parser = DocTestStringParser(DocTestEvaluator(doctest_optionflags))
        self.codeblock_parser = CodeBlockParser(language="pycon")

    def __call__(self, document: Document) -> Iterable[Region]:
        """Parse pycon code blocks and yield doctest regions.

        Parameters
        ----------
        document : Document
            The Sybil document to parse.

        Yields
        ------
        Region
            Parsed regions from either pycon doctest blocks or regular code blocks.

        """
        for region in self.codeblock_parser(document):
            source = region.parsed
            # Treat pycon blocks as doctests (always start with >>>)
            if isinstance(source, str) and source.startswith(">>>"):
                for doctest_region in self.doctest_parser(source, document.path):
                    doctest_region.adjust(region, source)
                    yield doctest_region
            else:
                yield region


parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    DocTestDirectiveParser(optionflags=optionflags),
    PythonCodeBlockParser(doctest_optionflags=optionflags),
    PyconCodeBlockParser(doctest_optionflags=optionflags),
    MystSkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(
    parsers=[*parsers, rest.DocTestParser(optionflags=optionflags), rest.SkipParser()],
    patterns=["*.py"],
)


pytest_collect_file = (docs + python).pytest()
