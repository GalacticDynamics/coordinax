"""Tests for vconvert API."""

import coordinax.api.manifolds as cxmapi


def test_guess_manifold() -> None:
    """Test that guess_manifold can be dispatched on."""

    assert len(cxmapi.guess_manifold.methods) > 0


def test_pt_embed() -> None:
    """Test that pt_embed can be dispatched on."""

    assert len(cxmapi.pt_embed.methods) > 0


def test_pt_project() -> None:
    """Test that pt_project can be dispatched on."""

    assert len(cxmapi.pt_project.methods) > 0


def test_pt_map() -> None:
    """Test that pt_map can be dispatched on."""

    assert len(cxmapi.pt_map.methods) > 0


def test_scale_factors() -> None:
    """Test that scale_factors can be dispatched on."""

    assert len(cxmapi.scale_factors.methods) > 0


def test_angle_between() -> None:
    """Test that angle_between can be dispatched on."""

    assert len(cxmapi.angle_between.methods) > 0
