"""Test the package itself."""

import importlib.metadata

import coordinax as cx


def test_version():
    """Test version."""
    assert importlib.metadata.version("coordinax") == cx.__version__
