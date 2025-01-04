"""Test :mod:`coordinax.distance`."""

import pytest
from plum import convert, promote

import unxt as u

import coordinax as cx


def test_promotion_rule():
    """Test the promotion rule for distance."""
    # Quantities
    d = cx.distance.Distance(90.0, "pc")
    q = u.Quantity(1.0, "kpc")

    # Explicit promotion test
    d_p, q_p = promote(d, q)
    assert isinstance(d_p, u.Quantity)
    assert isinstance(q_p, u.Quantity)

    # Implicit promotion test
    assert isinstance(d * q, u.Quantity)
    assert isinstance(q * d, u.Quantity)


@pytest.mark.parametrize(
    "d", [cx.distance.Distance(90, "pc"), cx.distance.DistanceModulus(26, "mag")]
)
def test_convert_distance_to_quantity(d):
    """Test converting distance types to general quantity types.

    These conversions should be covered under rules defined in `unxt`.

    """
    q = convert(d, u.Quantity)

    assert isinstance(q, u.Quantity)
    assert q.unit is d.unit
    assert q.value is d.value
