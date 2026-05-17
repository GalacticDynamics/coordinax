"""Test {mod}`coordinax.angles`."""

from hypothesis import given
from plum import convert, promote

import unxt as u

import coordinax.hypothesis.main as cxst


@given(cxst.angles())
def test_promotion_rule(a):
    """Test the promotion rule for angles."""
    # Quantities
    q = u.Q(1, "rad")

    # Explicit promotion test
    a_p, q_p = promote(a, q)
    assert isinstance(a_p, u.Q)
    assert isinstance(q_p, u.Q)

    # Implicit promotion test
    assert isinstance(a * q, u.Q)
    assert isinstance(q * a, u.Q)


@given(cxst.angles())
def test_convert_angle_to_quantity(a):
    """Test converting angle types to general quantity types.

    These conversions should be covered under rules defined in `unxt`.

    """
    q = convert(a, u.Q)

    assert isinstance(q, u.Q)
    assert q.unit is a.unit
    assert q.value is a.value
