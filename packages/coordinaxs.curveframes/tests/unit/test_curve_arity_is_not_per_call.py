"""A curve's arity cannot change, so it is not re-derived on every evaluation.

`_is_two_argument` runs inside `_resolve`, which `__call__`, `location`,
`tangent` and `rotation_matrix` all route through -- so an
`inspect.signature` call there was paid once per curve evaluation, including
inside ODE and root solves.

Correctness is identical either way, so these pin the *mechanism*: without
them a revert to inspecting every time passes every other test in the suite.
"""

import functools as ft

import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc
from coordinaxs.curveframes._src.arclength import (
    _arity_from_signature_cached,
    _is_two_argument,
)


def one_arg(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def two_arg(tau: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    del t
    return one_arg(tau)


def test_a_hashable_curve_is_inspected_once() -> None:
    """The second ask is a cache hit, not a second `inspect.signature`."""
    _arity_from_signature_cached.cache_clear()

    assert _is_two_argument(one_arg) is False
    after_first = _arity_from_signature_cached.cache_info()
    assert after_first.misses == 1

    for _ in range(50):
        assert _is_two_argument(one_arg) is False
    after_many = _arity_from_signature_cached.cache_info()
    assert after_many.misses == 1  # still one -- never inspected again
    assert after_many.hits == 50


def test_attime_states_its_arity_instead_of_being_inspected() -> None:
    """Binding the time is what `AtTime` does, so one-argument is structural."""
    at = cxfc.AtTime(two_arg, u.Q(1.0, "s"))
    assert at._two_argument is False

    _arity_from_signature_cached.cache_clear()
    assert _is_two_argument(at) is False
    assert _arity_from_signature_cached.cache_info().misses == 0  # never reached


def test_an_unhashable_curve_still_works() -> None:
    """`equinox.Module` curves hold arrays and are unhashable.

    They cannot be cache keys, so they must fall through to the uncached
    read rather than raising on the lookup.
    """
    arc = cxfc.ArcLength(two_arg, "s")
    with pytest.raises(TypeError):
        hash(arc)
    assert _is_two_argument(arc) is True


def test_the_readings_are_unchanged() -> None:
    """Memoising must not move any of the boundaries `_is_two_argument` draws."""
    assert _is_two_argument(one_arg) is False
    assert _is_two_argument(two_arg) is True
    assert _is_two_argument(lambda tau, *args: tau) is False
    assert _is_two_argument(lambda tau, smoothing=0.1: tau) is False
    assert _is_two_argument(ft.partial(lambda a, t: a, t=1)) is False


def test_a_raise_is_not_memoised() -> None:
    """A required keyword-only second parameter raises every time, not once."""

    def kwonly(tau, *, resolution):
        del resolution
        return tau

    for _ in range(3):
        with pytest.raises(TypeError, match="keyword-only"):
            _is_two_argument(kwonly)
