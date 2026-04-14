"""Tests for vconvert API."""

import coordinax.api.frames as cxfapi


def test_frame_transition() -> None:
    """Test that frame_transition can be dispatched on."""

    assert len(cxfapi.frame_transition.methods) > 0
