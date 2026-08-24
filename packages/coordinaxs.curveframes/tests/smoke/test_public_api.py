"""Smoke tests for the `coordinaxs.curveframes` public API."""

__all__: tuple[str, ...] = ()

import pytest

import coordinaxs.curveframes as cxfc


@pytest.mark.parametrize("name", sorted(cxfc.__all__))
def test_all_symbols_present(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    assert hasattr(cxfc, name), f"coordinaxs.curveframes missing: {name}"
