"""Smoke tests for the `coordinaxs.hypothesis` public API.

`coordinaxs.hypothesis` is a namespace package with no top-level ``__init__``,
so the surface is enumerated from its subpackages rather than one ``__all__``.
"""

__all__: tuple[str, ...] = ()

import importlib

import pytest

#: The subpackages that make up the `coordinaxs.hypothesis` surface.
SUBPACKAGES = (
    "angles",
    "charts",
    "distances",
    "main",
    "manifolds",
    "representations",
    "utils",
    "vectors",
)

EXPORT_CASES = [
    (subpackage, name)
    for subpackage in SUBPACKAGES
    for name in importlib.import_module(f"coordinaxs.hypothesis.{subpackage}").__all__
]


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_subpackage_importable(subpackage: str) -> None:
    """Each subpackage is importable."""
    importlib.import_module(f"coordinaxs.hypothesis.{subpackage}")


@pytest.mark.parametrize(
    ("subpackage", "name"), EXPORT_CASES, ids=[f"{s}.{n}" for s, n in EXPORT_CASES]
)
def test_all_symbols_present(subpackage: str, name: str) -> None:
    """Every name in a subpackage's ``__all__`` resolves on it."""
    module = importlib.import_module(f"coordinaxs.hypothesis.{subpackage}")
    assert hasattr(module, name), f"coordinaxs.hypothesis.{subpackage} missing: {name}"
