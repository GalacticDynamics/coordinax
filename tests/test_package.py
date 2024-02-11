"""Test the package itself."""

import importlib.metadata

import vector as m


def test_version():
    """Test version."""
    assert importlib.metadata.version("vector") == m.__version__
