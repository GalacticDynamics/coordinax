"""Tests for cconvert API."""

import coordinax.api.charts as cxcapi


def test_cartesian_chart() -> None:
    """Test that cartesian_chart can be dispatched on."""

    assert len(cxcapi.cartesian_chart.methods) > 0


def test_pt_map() -> None:
    """Test that pt_map can be dispatched on."""

    assert len(cxcapi.pt_map.methods) > 0


def test_realize_cartesian() -> None:
    """Test that realize_cartesian can be dispatched on."""

    assert len(cxcapi.realize_cartesian.methods) > 0


def test_unrealize_cartesian() -> None:
    """Test that unrealize_cartesian can be dispatched on."""

    assert len(cxcapi.unrealize_cartesian.methods) > 0


def test_guess_chart() -> None:
    """Test that guess_chart can be dispatched on."""

    assert len(cxcapi.guess_chart.methods) > 0


def test_cdict() -> None:
    """Test that cdict can be dispatched on."""

    assert len(cxcapi.cdict.methods) > 0
