"""Tests for vconvert API."""

import coordinax.api.transforms as cxfmapi


def test_act() -> None:
    """Test that act can be dispatched on."""

    assert len(cxfmapi.act.methods) > 0


def test_compose() -> None:
    """Test that compose can be dispatched on."""

    assert len(cxfmapi.compose.methods) > 0


def test_simplify() -> None:
    """Test that simplify can be dispatched on."""

    assert len(cxfmapi.simplify.methods) > 0
