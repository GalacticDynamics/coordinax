"""Doctest configuration."""

from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Document, Region, Sybil
from sybil.parsers import myst, rest

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    myst.SkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(  # TODO: use myst for doctests
    parsers=[*parsers, rest.DocTestParser(optionflags=optionflags), rest.SkipParser()],
    patterns=["*.py"],
)


pytest_collect_file = (docs + python).pytest()
