"""Test :mod:`coordinax.distance`."""

import hypothesis.strategies as st
import plum
from hypothesis import given

import unxt as u

import coordinax_hypothesis.core as cxst


@given(cxst.distances())
def test_promote_distance(d):
    """Test the promotion rule for distance."""
    # Quantities
    q = u.Q(1.0, "kpc")

    # Explicit promotion test
    d_p, q_p = plum.promote(d, q)
    assert isinstance(d_p, u.Q)
    assert isinstance(q_p, u.Q)

    # Implicit promotion test
    assert isinstance(d * q, u.Q)
    assert isinstance(q * d, u.Q)


@given(st.one_of(cxst.distances(), cxst.distance_moduli()))
def test_convert_distance_to_quantity(d):
    """Test converting distance types to general quantity types.

    These conversions should be covered under rules defined in `unxt`.

    """
    q = plum.convert(d, u.Q)

    assert isinstance(q, u.Q)
    assert q.unit is d.unit
    assert q.value is d.value
