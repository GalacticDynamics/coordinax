"""Smoke tests for the `coordinaxs.interop.astropy` public API."""

__all__: tuple[str, ...] = ()

import pytest

import coordinaxs.interop.astropy as cxia


@pytest.mark.parametrize("name", sorted(cxia.__all__))
def test_all_symbols_present(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    assert hasattr(cxia, name), f"coordinaxs.interop.astropy missing: {name}"
