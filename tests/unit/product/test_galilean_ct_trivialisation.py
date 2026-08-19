"""`GalileanCT` asserts a product, so its spatial factor must not move.

Galilean spacetime is a fibre bundle over the time axis: the time function is
canonical, the simultaneity slices are unambiguous, and nothing canonically
identifies a point on one slice with a point on another. Writing a chart as
``time1d x spatial_chart`` fixes such an identification -- a rest frame -- which
is fine for a spatial chart that does not move and false for one that does.

The array check inherited from `AbstractStaticChart` does not answer this. It is
about JAX safety: an array inside a static node is invisible to `jit`. A
time-dependent chart can pass it, which is what these tests pin.
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinaxs.curveframes as cxfc


def _static_curve(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """An ordinary curve: its shape does not depend on any time."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


def _moving_curve(s: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """A curve that stretches and bends with ``t``."""
    sv, tv = s.ustrip("km"), t.ustrip("s")
    x = sv * (1.0 + 0.5 * tv)
    y = 0.1 * tv * sv**2
    return u.Q(jnp.stack([x, y, jnp.zeros_like(x)]), "km")


def _tube(builder) -> cxfc.TubularChart:
    """A chart with *no array leaves*, so the static guard cannot object."""
    return cxfc.TubularChart(
        builder, tau_bounds=(u.StaticQuantity(0.0, "s"), u.StaticQuantity(2.0, "s"))
    )


def test_a_time_dependent_spatial_chart_is_not_a_factor() -> None:
    """The geometric guard, on an object the array guard lets through.

    Fails if `GalileanCT` goes back to relying on the array check alone.
    """
    moving = _tube(
        cxfc.FrenetSerretBuilder(
            _moving_curve, "km", station=u.StaticQuantity(1.3, "km")
        )
    )
    assert moving.is_time_dependent

    # the array guard has no objection: the only leaf is the curve itself
    assert not any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(moving)), (
        "this test is vacuous if the array guard would already reject it"
    )

    with pytest.raises(TypeError, match="fibre bundle"):
        cxc.GalileanCT(moving)


def test_a_static_spatial_chart_is_still_a_factor() -> None:
    """A curve that does not move gives a genuine product; nothing changes."""
    still = _tube(cxfc.FrenetSerretBuilder(_static_curve, "s"))
    assert not still.is_time_dependent
    cxc.GalileanCT(still)


def test_the_ordinary_charts_are_untouched() -> None:
    """The guard must not cost anything on the common path."""
    assert cxc.GalileanCT(cxc.cart3d).factors[1] is cxc.cart3d
    cxc.GalileanCT(cxc.sph3d)


def test_the_message_names_the_missing_datum() -> None:
    """A refusal that does not say what is missing sends the reader guessing.

    What a product cannot carry is the connection -- the frame velocity, i.e.
    which point at ``t`` counts as the same point at ``t'``.
    """
    moving = _tube(
        cxfc.FrenetSerretBuilder(
            _moving_curve, "km", station=u.StaticQuantity(1.3, "km")
        )
    )
    with pytest.raises(TypeError, match="connection"):
        cxc.GalileanCT(moving)
    with pytest.raises(TypeError, match="AtTime"):
        cxc.GalileanCT(moving)
