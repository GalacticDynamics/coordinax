"""Tests for API."""


def test_has_subpackage_charts() -> None:
    """Test that the charts subpackage is importable."""
    import coordinax.api.charts as cxcapi  # noqa: F401
