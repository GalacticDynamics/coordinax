"""Tests for vconvert API."""

import inspect

import plum
import pytest

import coordinax_api as cxapi
from coordinax_api import vconvert


def test_vconvert_is_abstract() -> None:
    """Test that vconvert is an abstract dispatch function."""
    # vconvert should raise NotFoundLookupError when called with unregistered types
    with pytest.raises(plum.NotFoundLookupError):
        vconvert(int, 1)


def test_vconvert_signature() -> None:
    """Test that vconvert has the expected signature."""
    sig = inspect.signature(vconvert)
    params = list(sig.parameters.keys())

    # Check that it has the expected parameters
    assert params[0] == "target"
    assert params[1] == "args"
    assert params[2] == "kwargs"


def test_vconvert_is_dispatchable() -> None:
    """Test that vconvert can be dispatched on."""

    # Should be able to add dispatches to vconvert
    @plum.dispatch
    def vconvert(target: str, value: int, **kwargs) -> str:
        return f"int:{value}"

    # Now this dispatch should work
    result = vconvert("str", 42)
    assert result == "int:42"


def test_vconvert_module_exports() -> None:
    """Test that vconvert is properly exported from the module."""
    # Should be in __all__
    assert "vconvert" in cxapi.__all__

    # Should be accessible from module
    assert hasattr(cxapi, "vconvert")
    assert cxapi.vconvert is vconvert
