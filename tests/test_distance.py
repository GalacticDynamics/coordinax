"""Test :mod:`coordinax.distance`."""

from plum import promote

import unxt as u

import coordinax as cx


def test_promotion_rule():
    """Test the promotion rule for distance."""
    # Quantities
    a = cx.distance.Distance(90.0, "pc")
    q = u.Quantity(1.0, "kpc")

    # Explicit promotion test
    a_p, q_p = promote(a, q)
    assert isinstance(a_p, u.Quantity)
    assert isinstance(q_p, u.Quantity)

    # Implicit promotion test
    assert isinstance(a * q, u.Quantity)
    assert isinstance(q * a, u.Quantity)
