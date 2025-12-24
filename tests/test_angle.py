"""Test :mod:`coordinax.angles`."""

import pytest
from plum import convert, promote

import unxt as u

import coordinax as cx


@pytest.mark.parametrize("a", [cx.Angle(90, "deg"), cx.angles.Parallax(26, "mas")])
def test_promotion_rule(a):
    """Test the promotion rule for angles."""
    # Quantities
    q = u.Q(1.0, "rad")

    # Explicit promotion test
    a_p, q_p = promote(a, q)
    assert isinstance(a_p, u.Q)
    assert isinstance(q_p, u.Q)

    # Implicit promotion test
    assert isinstance(a * q, u.Q)
    assert isinstance(q * a, u.Q)


@pytest.mark.parametrize("a", [cx.Angle(90, "deg"), cx.angles.Parallax(26, "mas")])
def test_convert_angle_to_quantity(a):
    """Test converting angle types to general quantity types.

    These conversions should be covered under rules defined in `unxt`.

    """
    q = convert(a, u.Q)

    assert isinstance(q, u.Q)
    assert q.unit is a.unit
    assert q.value is a.value
