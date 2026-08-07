"""Tests for the optional `unxts.parametric` registrations.

Promotion rules and `from_` dispatch both live here so the whole module
skips as a unit when the dependency is absent.
"""

import sys

import jax.numpy as jnp
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
    """Both operand orders give a `ParametricQuantity`."""
    pq = PQ(1.0, "rad")
    assert isinstance(distance * pq, ParametricQuantity)
    assert isinstance(pq * distance, ParametricQuantity)


@DISTANCES
def test_quantity_promotion_is_unchanged(distance: u.AbstractQuantity) -> None:
    """The pre-existing degrade-to-`Q` behaviour still holds."""
    assert isinstance(distance * u.Q(2.0, ""), u.Q)


def _resolved_from_(cls, arg, /):
    """Return the `from_` implementation plum selects for *arg*.

    `from_` is a `classmethod` wrapping a plum dispatcher, and how many
    wrappers sit between it and the `Function` that can `resolve_method` is not
    stable across Python versions: 3.12 nests
    ``method -> _BoundFunction -> Function``, 3.13+ drops the outer one. Walk
    the chain until something can resolve, rather than hardcoding the hops.
    """
    f = cls.from_
    for _ in range(5):
        if hasattr(f, "resolve_method"):
            fn, _ = f.resolve_method((cls, arg))
            return fn
        f = getattr(f, "__func__", None) or getattr(f, "_f", None)
        if f is None:
            break
    msg = f"could not reach plum's dispatcher from {cls.__name__}.from_"
    raise AssertionError(msg)


FROM_CASES = [
    pytest.param(
        cxastro.Parallax, [(1, "mas"), (10, "pc"), (10, "mag")], id="Parallax"
    ),
    pytest.param(
        cxastro.DistanceModulus,
        [(1, "pc"), (1, "mas"), (1, "mag")],
        id="DistanceModulus",
    ),
]


class TestParametricFromDispatch:
    """`from_` routes a `ParametricQuantity` by type, not by runtime dimension."""

    @pytest.mark.parametrize(("cls", "cases"), FROM_CASES)
    def test_matches_quantity_path(self, cls, cases) -> None:
        """A ParametricQuantity gives the same result as a plain Quantity."""
        for value, unit in cases:
            got = cls.from_(PQ(value, unit), dtype=float)
            expected = cls.from_(u.Q(value, unit), dtype=float)
            assert got.unit == expected.unit
            assert jnp.allclose(got.value, expected.value)

    @pytest.mark.parametrize(("cls", "cases"), FROM_CASES)
    def test_dispatches_by_type_not_by_dimension(self, cls, cases) -> None:
        """The parametric input selects a *different* overload than a plain one.

        Behavioural equivalence alone cannot see this: drop the parametric
        registrations and a `PQ` falls through to the `AbstractQuantity`
        overload, branches on `u.dimension_of` and returns the very same
        answer -- the feature gone with every other assertion still green.

        Poisoning `u.dimension_of` is not a substitute: `__check_init__` calls
        it to validate the *constructed* object, so it fires on both paths.
        """
        for value, unit in cases:
            parametric = _resolved_from_(cls, PQ(value, unit))
            plain = _resolved_from_(cls, u.Q(value, unit))
            assert parametric is not plain
