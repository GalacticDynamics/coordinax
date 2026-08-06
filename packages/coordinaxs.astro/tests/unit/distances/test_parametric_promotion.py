"""Tests for the optional `unxts.parametric` registrations."""

import sys

import plum
import pytest

import unxt as u

import coordinaxs.astro as cxastro
from coordinaxs.astro._src.optional_deps import OptDeps

if not OptDeps.UNXTS_PARAMETRIC.installed:
    # Module-level: the imports below would fail at collection time, so this
    # has to skip before them rather than as a `pytestmark` on each test.
    pytest.skip("unxts.parametric is not installed", allow_module_level=True)

from unxts.parametric import PQ, ParametricQuantity

DISTANCES = pytest.mark.parametrize(
    "distance",
    [cxastro.Parallax(1, "mas"), cxastro.DistanceModulus(1, "mag")],
    ids=["Parallax", "DistanceModulus"],
)


def test_registrations_ran() -> None:
    """The gated import in ``_src/__init__`` fired for this install."""
    assert "coordinaxs.astro._src.register_parametric" in sys.modules


@DISTANCES
def test_promotes_to_parametric_quantity(distance: u.AbstractQuantity) -> None:
    """A distance and a `ParametricQuantity` promote to the latter."""
    promoted = plum.promote(distance, PQ(1.0, "rad"))
    assert all(isinstance(x, ParametricQuantity) for x in promoted)


@DISTANCES
def test_arithmetic_is_order_independent(distance: u.AbstractQuantity) -> None:
    """Both operand orders give a `ParametricQuantity`.

    Without the promotion rule, ``distance * pq`` dispatched to the
    distance-returning multiply and raised on the dimension check, while
    ``pq * distance`` succeeded -- the operand order decided whether the
    expression worked at all.
    """
    pq = PQ(1.0, "rad")
    assert isinstance(distance * pq, ParametricQuantity)
    assert isinstance(pq * distance, ParametricQuantity)


@DISTANCES
def test_quantity_promotion_is_unchanged(distance: u.AbstractQuantity) -> None:
    """The pre-existing degrade-to-`Q` behaviour still holds."""
    assert isinstance(distance * u.Q(2.0, ""), u.Q)
