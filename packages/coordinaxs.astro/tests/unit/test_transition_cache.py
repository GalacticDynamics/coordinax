"""The Galactocentric operator is built once per distinct frame (#951)."""

__all__: tuple[str, ...] = ()


import jax
import numpy as np
import pytest

import unxt as u

import coordinax.frames as cxf
import coordinaxs.astro as cxastro
from coordinaxs.astro._src import frame_transforms as ft

Q = u.Q([1.0, 2.0, 3.0], "kpc")


@pytest.fixture(autouse=True)
def _empty_cache():
    """Each test starts from an empty cache and leaves one behind."""
    ft._TRANSITION_CACHE.clear()
    yield
    ft._TRANSITION_CACHE.clear()


def test_equal_but_distinct_frames_share_one_operator() -> None:
    """The build is a pure function of the parameters, so it runs once."""
    first = cxf.frame_transition(cxastro.icrs, cxastro.Galactocentric())
    second = cxf.frame_transition(cxastro.icrs, cxastro.Galactocentric())
    assert first is second


def test_a_different_frame_is_not_served_from_the_cache() -> None:
    default = cxf.frame_transition(cxastro.icrs, cxastro.Galactocentric())(Q)
    rolled = cxf.frame_transition(
        cxastro.icrs, cxastro.Galactocentric(roll=u.Q(10, "deg"))
    )(Q)
    assert not np.allclose(np.asarray(default.value), np.asarray(rolled.value))


def test_the_cached_operator_matches_a_fresh_build() -> None:
    """A hit must be indistinguishable from rebuilding."""
    frame = cxastro.Galactocentric()
    cached = cxf.frame_transition(cxastro.icrs, frame)(Q)
    ft._TRANSITION_CACHE.clear()
    fresh = cxf.frame_transition(cxastro.icrs, frame)(Q)
    np.testing.assert_array_equal(np.asarray(cached.value), np.asarray(fresh.value))


def test_the_inverse_direction_round_trips() -> None:
    frame = cxastro.Galactocentric()
    there = cxf.frame_transition(cxastro.icrs, frame)(Q)
    back = cxf.frame_transition(frame, cxastro.icrs)(there)
    np.testing.assert_allclose(
        np.asarray(back.value), np.asarray(Q.value), rtol=0, atol=1e-12
    )


def test_a_traced_frame_is_not_cached() -> None:
    """A tracer has no concrete bytes, so it can neither hit nor fill the cache."""
    traced = jax.eval_shape(lambda f: f, cxastro.Galactocentric())
    assert ft._frame_key("icrs->gcf", traced) is None
    assert len(ft._TRANSITION_CACHE) == 0


def test_the_cache_is_bounded() -> None:
    """A parameter sweep cannot grow the cache without limit."""
    for i in range(ft._TRANSITION_CACHE_MAX + 5):
        cxf.frame_transition(cxastro.icrs, cxastro.Galactocentric(roll=u.Q(i, "deg")))
    assert len(ft._TRANSITION_CACHE) <= ft._TRANSITION_CACHE_MAX
