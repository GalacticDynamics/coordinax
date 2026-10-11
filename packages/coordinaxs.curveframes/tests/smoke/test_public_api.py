"""Smoke tests for the `coordinaxs.curveframes` public API."""

__all__: tuple[str, ...] = ()

import importlib.metadata

import pytest

import coordinaxs.curveframes as cxfc


@pytest.mark.parametrize("name", sorted(cxfc.__all__))
def test_all_symbols_present(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    assert hasattr(cxfc, name), f"coordinaxs.curveframes missing: {name}"


def test_version() -> None:
    """``__version__`` is the installed distribution's version."""
    assert isinstance(cxfc.__version__, str)
    assert cxfc.__version__
    assert cxfc.__version__ == importlib.metadata.version("coordinaxs.curveframes")
