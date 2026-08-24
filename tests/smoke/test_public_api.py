"""Smoke tests for the `coordinax` public API."""

__all__: tuple[str, ...] = ()

import pytest

import coordinax as cx


@pytest.mark.parametrize("name", sorted(cx.__all__))
def test_all_symbols_present(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    assert hasattr(cx, name), f"coordinax missing: {name}"
