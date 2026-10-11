"""Smoke tests for the `coordinaxs.astro` public API."""

__all__: tuple[str, ...] = ()

import importlib.metadata

import pytest

import coordinaxs.astro as cxastro


@pytest.mark.parametrize("name", sorted(cxastro.__all__))
def test_all_symbols_present(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    assert hasattr(cxastro, name), f"coordinaxs.astro missing: {name}"


def test_version() -> None:
    """``__version__`` is the installed distribution's version."""
    assert isinstance(cxastro.__version__, str)
    assert cxastro.__version__
    assert cxastro.__version__ == importlib.metadata.version("coordinaxs.astro")
