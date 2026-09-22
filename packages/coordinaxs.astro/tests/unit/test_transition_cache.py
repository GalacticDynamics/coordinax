"""The Galactocentric operator is built once per distinct frame (#951)."""

__all__: tuple[str, ...] = ()


import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.frames as cxf
import coordinaxs.astro as cxastro
from coordinaxs.astro._src import frame_transforms as ft

Q = u.Q([1.0, 2.0, 3.0], "kpc")


@pytest.fixture(autouse=True)
def _empty_cache():
    """Each test starts and ends with an empty cache.

    Cleared on the way in because anything earlier in the session may have
    populated it, and on the way out so these tests -- which count entries --
    do not leave any for the rest of the suite.
    """
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


class _ArrayFrame(cxf.AbstractReferenceFrame):
    """A frame with an array parameter, for keying tests only.

    Defined at module scope: plum registers globally, so a class defined inside
    a test would leak methods keyed to a dead class into the session.
    """

    param: jnp.ndarray


def test_the_key_separates_leaves_that_share_bytes() -> None:
    """Shape and dtype are part of the key, not just the buffer.

    ``(2,) float32`` and ``(1,) float64`` zeros have identical `tobytes()` and
    an identical treedef, so a bytes-only key would serve one frame's operator
    for the other.
    """
    small = _ArrayFrame(jnp.zeros((2,), dtype=jnp.float32))
    wide = _ArrayFrame(jnp.zeros((1,), dtype=jnp.float64))

    assert small.param.tobytes() == wide.param.tobytes()  # the collision
    assert ft._frame_key("t", small) != ft._frame_key("t", wide)
