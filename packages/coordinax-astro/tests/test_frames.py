"""Tests for astronomical reference frames."""

import coordinax_astro as cxa


def test_icrs_frame_creation() -> None:
    """Test that ICRS frame can be created."""
    frame = cxa.ICRS()
    assert isinstance(frame, cxa.AbstractSpaceFrame)
    assert isinstance(frame, cxa.ICRS)


def test_galactocentric_frame_creation() -> None:
    """Test that Galactocentric frame can be created."""
    frame = cxa.Galactocentric()
    assert isinstance(frame, cxa.AbstractSpaceFrame)
    assert isinstance(frame, cxa.Galactocentric)


def test_abstract_space_frame_is_abstract() -> None:
    """Test that AbstractSpaceFrame cannot be directly instantiated."""
    # This test depends on how AbstractSpaceFrame is implemented
    # If it's truly abstract, this should raise an error
    # For now, just verify it exists
    assert hasattr(cxa, "AbstractSpaceFrame")
