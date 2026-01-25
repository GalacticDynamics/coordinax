"""Test :mod:`coordinax.angles`."""

import hypothesis.strategies as st
from hypothesis import given
from plum import convert, promote

import unxt as u

import coordinax_hypothesis as cxst


@given(st.one_of(cxst.angles(), cxst.parallaxes()))
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


@given(st.one_of(cxst.angles(), cxst.parallaxes()))
def test_convert_angle_to_quantity(a):
    """Test converting angle types to general quantity types.

    These conversions should be covered under rules defined in `unxt`.

    """
    q = convert(a, u.Q)

    assert isinstance(q, u.Q)
    assert q.unit is a.unit
    assert q.value is a.value
