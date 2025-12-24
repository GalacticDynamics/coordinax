"""Test :mod:`coordinax.distance`."""

import plum
import pytest

import unxt as u

import coordinax as cx


def test_promote_distance():
    """Test the promotion rule for distance."""
    # Quantities
    d = cx.distances.Distance(90.0, "pc")
    q = u.Q(1.0, "kpc")

    # Explicit promotion test
    d_p, q_p = plum.promote(d, q)
    assert isinstance(d_p, u.Q)
    assert isinstance(q_p, u.Q)

    # Implicit promotion test
    assert isinstance(d * q, u.Q)
    assert isinstance(q * d, u.Q)


@pytest.mark.parametrize(
    "d", [cx.distances.Distance(90, "pc"), cx.distances.DistanceModulus(26, "mag")]
)
def test_convert_distance_to_quantity(d):
    """Test converting distance types to general quantity types.

    These conversions should be covered under rules defined in `unxt`.

    """
    q = plum.convert(d, u.Q)

    assert isinstance(q, u.Q)
    assert q.unit is d.unit
    assert q.value is d.value
