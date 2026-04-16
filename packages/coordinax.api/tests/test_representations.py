"""Tests for cconvert API."""

import coordinax.api.representations as cxrapi


def test_cconvert() -> None:
    """Test that cconvert can be dispatched on."""

    assert len(cxrapi.cconvert.methods) > 0


def test_guess_geometry_kind() -> None:
    """Test that guess_geometry_kind can be dispatched on."""

    assert len(cxrapi.guess_geometry_kind.methods) > 0


def test_guess_rep() -> None:
    """Test that guess_rep can be dispatched on."""

    assert len(cxrapi.guess_rep.methods) > 0


def test_guess_semantic_kind() -> None:
    """Test that guess_semantic_kind can be dispatched on."""

    assert len(cxrapi.guess_semantic_kind.methods) > 0
