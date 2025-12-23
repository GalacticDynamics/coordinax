"""Tests for coordinax_api module."""

import coordinax_api as cxapi


def test_module_has_version() -> None:
    """Test that the module has a __version__ attribute."""
    assert hasattr(cxapi, "__version__")
    assert isinstance(cxapi.__version__, str)


def test_module_exports() -> None:
    """Test that the module exports the expected public API."""
    expected_exports = {"__version__", "vconvert"}

    assert set(cxapi.__all__) == expected_exports

    # All items in __all__ should be accessible
    for name in cxapi.__all__:
        assert hasattr(cxapi, name)


def test_module_docstring() -> None:
    """Test that the module has a docstring."""
    assert cxapi.__doc__ is not None
    assert "coordinax" in cxapi.__doc__.lower()
