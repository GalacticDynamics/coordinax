"""Test the package itself."""

import importlib.metadata

import coordinax as pkg


def test_version():
    """Test version."""
    assert importlib.metadata.version("coordinax") == pkg.__version__
