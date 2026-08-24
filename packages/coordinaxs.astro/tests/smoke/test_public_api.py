"""Smoke tests for the `coordinaxs.astro` public API."""

__all__: tuple[str, ...] = ()

import pytest

import coordinaxs.astro as cxastro


@pytest.mark.parametrize("name", sorted(cxastro.__all__))
def test_all_symbols_present(name: str) -> None:
    """Every name in ``__all__`` resolves on the package."""
    assert hasattr(cxastro, name), f"coordinaxs.astro missing: {name}"
