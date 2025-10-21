"""Tests for `coordinax.KinematicSpace`."""

from textwrap import dedent

import coordinax as cx


def test_repr():
    """Test the repr of KinematicSpace."""
    # A simple KinematicSpace from a position
    space = cx.KinematicSpace.from_(cx.CartesianPos3D.from_([1, 2, 3], "kpc"))

    # Expected repr
    exp = """
    KinematicSpace({
      'length':
      CartesianPos3D(
        x=Quantity(1, unit='kpc'), y=Quantity(2, unit='kpc'), z=Quantity(3, unit='kpc')
      )
    })
    """[1:-1]  # remove first and last newline
    exp = dedent(exp).strip()  # dedent & strip leading/trailing whitespace

    assert repr(space) == exp
