"""Smoke tests for the `coordinax` public API."""

__all__: tuple[str, ...] = ()

import importlib.metadata

import pytest

import coordinax as cx


@pytest.mark.parametrize("name", sorted(cx.__all__))
def test_all_symbols_present(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    assert hasattr(cx, name), f"coordinax missing: {name}"


def test_version() -> None:
    """``__version__`` is the installed distribution's version."""
    assert isinstance(cx.__version__, str)
    assert cx.__version__
    assert cx.__version__ == importlib.metadata.version("coordinax")
