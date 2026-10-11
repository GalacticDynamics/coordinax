"""Smoke tests for the `coordinaxs.interop.astropy` public API."""

__all__: tuple[str, ...] = ()

import importlib.metadata

import pytest

import coordinaxs.interop.astropy as cxia


@pytest.mark.parametrize("name", sorted(cxia.__all__))
def test_all_symbols_present(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    assert hasattr(cxia, name), f"coordinaxs.interop.astropy missing: {name}"


def test_version() -> None:
    """``__version__`` is the installed distribution's version."""
    assert isinstance(cxia.__version__, str)
    assert cxia.__version__
    assert cxia.__version__ == importlib.metadata.version("coordinaxs.interop.astropy")
