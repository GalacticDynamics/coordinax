"""Tests for coordinax_astro module."""

import coordinax_astro as cxastro


def test_module_exports() -> None:
    """Test that the module exports the expected public API."""
    expected_exports = {"AbstractSpaceFrame", "ICRS", "Galactocentric"}

    assert set(cxastro.__all__) == expected_exports

    # All items in __all__ should be accessible
    for name in cxastro.__all__:
        assert hasattr(cxastro, name)


def test_module_docstring() -> None:
    """Test that the module has a docstring."""
    assert cxastro.__doc__ is not None
    assert "astronomy" in cxastro.__doc__.lower()
