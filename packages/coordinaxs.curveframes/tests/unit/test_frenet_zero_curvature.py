r"""Frenet--Serret must refuse where its frame is undefined, not return NaN.

The normal is the Gram--Schmidt rejection of $\gamma''$ from the tangent,
normalised. Where the curvature vanishes that rejection is zero, so the
normalisation divides by zero and the whole triad comes back NaN -- silently,
because `_normalize` had no zero-norm guard where Bishop's `_orthonormalize`
has one.

This is not an edge case: it is every straight segment and every inflection.
`check_data` catches the NaN further downstream, but the builder accessors --
`normal`, `binormal`, `rotation_matrix`, `__call__` -- did not.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def cubic(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Inflection at ``tau = 0``: curvature vanishes at one point."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, t**3, jnp.zeros_like(t)]), "km")


def straight(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Curvature vanishes everywhere."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Unit circle: curvature never vanishes."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


@pytest.mark.parametrize("accessor", ["normal", "binormal", "rotation_matrix"])
def test_an_inflection_is_refused_not_silently_nan(accessor: str) -> None:
    """Every accessor must refuse, not just the one that divides by zero."""
    builder = cxfc.FrenetSerretBuilder(cubic, "s")
    with pytest.raises(Exception, match="curvature vanishes"):
        getattr(builder, accessor)(u.Q(0.0, "s"))


def test_a_straight_curve_is_refused_everywhere() -> None:
    """Not a measure-zero case here -- the whole curve is degenerate."""
    builder = cxfc.FrenetSerretBuilder(straight, "s")
    for tau in (0.0, 1.0, -2.5):
        with pytest.raises(Exception, match="curvature vanishes"):
            builder.normal(u.Q(tau, "s"))


def test_the_message_points_at_the_frame_that_does_work() -> None:
    """Bishop exists precisely for this, so the refusal should say so."""
    builder = cxfc.FrenetSerretBuilder(straight, "s")
    with pytest.raises(Exception, match="BishopBuilder"):
        builder.normal(u.Q(0.0, "s"))
    # ...and it genuinely works there.
    got = cxfc.BishopBuilder(straight, "s", normal_0="auto").normal1(u.Q(0.0, "s"))
    assert np.all(np.isfinite(np.asarray(getattr(got, "value", got))))


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.7])
def test_a_curved_curve_is_untouched(tau: float) -> None:
    """The guard must not disturb the ordinary case."""
    got = cxfc.FrenetSerretBuilder(circle, "s").normal(u.Q(tau, "s"))
    arr = np.asarray(getattr(got, "value", got))
    assert np.all(np.isfinite(arr))
    assert np.linalg.norm(arr) == pytest.approx(1.0, rel=1e-9)


def test_just_off_the_inflection_still_works() -> None:
    """The guard is relative, so it fires at the locus and not beside it."""
    builder = cxfc.FrenetSerretBuilder(cubic, "s")
    for tau in (-1e-9, 1e-9):
        got = builder.normal(u.Q(tau, "s"))
        arr = np.asarray(getattr(got, "value", got))
        assert np.all(np.isfinite(arr)), (tau, arr)
