"""Tests for astronomical reference frames."""

import coordinax_astro as cxastro


def test_icrs_frame_creation() -> None:
    """Test that ICRS frame can be created."""
    frame = cxastro.ICRS()
    assert isinstance(frame, cxastro.AbstractSpaceFrame)
    assert isinstance(frame, cxastro.ICRS)


def test_galactocentric_frame_creation() -> None:
    """Test that Galactocentric frame can be created."""
    frame = cxastro.Galactocentric()
    assert isinstance(frame, cxastro.AbstractSpaceFrame)
    assert isinstance(frame, cxastro.Galactocentric)
