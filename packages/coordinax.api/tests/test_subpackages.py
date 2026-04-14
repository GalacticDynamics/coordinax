"""Tests for API."""


def test_has_subpackage_charts() -> None:
    """Test that the charts subpackage is importable."""
    import coordinax.api.charts as cxcapi  # noqa: F401


def test_has_subpackage_representations() -> None:
    """Test that the representations subpackage is importable."""
    import coordinax.api.representations as cxrapi  # noqa: F401


def test_has_subpackage_manifolds() -> None:
    """Test that the manifolds subpackage is importable."""
    import coordinax.api.manifolds as cxmapi  # noqa: F401
