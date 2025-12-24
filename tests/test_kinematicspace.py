"""Tests for `coordinax.KinematicSpace`."""

import coordinax as cx


def test_repr():
    """Test the repr of KinematicSpace."""
    # A simple KinematicSpace from a position
    space = cx.KinematicSpace.from_(cx.CartesianPos3D.from_([1, 2, 3], "kpc"))

    # Expected repr
    exp = "KinematicSpace({'length': CartesianPos3D(x=Q(1, 'kpc'), y=Q(2, 'kpc'), z=Q(3, 'kpc'))})"  # noqa: E501

    assert repr(space) == exp
