"""Tests for the optional `unxts.parametric` registrations.

Separated from `test_distance` so the whole module skips as a unit when the
dependency is absent, rather than each test carrying its own `importorskip`.
"""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.distances as cxd
from coordinax._src.optional_deps import OptDeps

if not OptDeps.UNXTS_PARAMETRIC.installed:
    # Module-level: the import below would fail at collection time, so this has
    # to skip before it rather than as a `pytestmark` on each test.
    pytest.skip("unxts.parametric is not installed", allow_module_level=True)

from unxts.parametric import PQ

PARAMS = [(1, "kpc"), (1, "mas"), (10, "mag")]


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


class TestParametricFromDispatch:
    """`from_` routes a `ParametricQuantity` by type, not by runtime dimension."""

    @pytest.mark.parametrize(("value", "unit"), PARAMS)
    def test_matches_quantity_path(self, value: float, unit: str) -> None:
        """A ParametricQuantity gives the same result as a plain Quantity."""
        got = cxd.Distance.from_(PQ(value, unit), dtype=float)
        expected = cxd.Distance.from_(u.Q(value, unit), dtype=float)
        assert got.unit == expected.unit
        assert jnp.allclose(got.value, expected.value)

    @pytest.mark.parametrize(("value", "unit"), PARAMS)
    def test_dispatches_by_type_not_by_dimension(self, value: float, unit: str) -> None:
        """The parametric input selects a *different* overload than a plain one.

        Behavioural equivalence alone cannot see this: drop the parametric
        registrations and a `PQ` falls through to the `AbstractQuantity`
        overload, branches on `u.dimension_of` and returns the very same
        answer -- the feature gone with every other assertion still green.

        Poisoning `u.dimension_of` is not a substitute: `__check_init__` calls
        it to validate the *constructed* object, so it fires on both paths.
        """
        parametric = _resolved_from_(cxd.Distance, PQ(value, unit))
        plain = _resolved_from_(cxd.Distance, u.Q(value, unit))
        assert parametric is not plain
