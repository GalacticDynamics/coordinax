"""Tests for the built-in TimeDep builders."""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .conftest import X_M as X, ZHAT


@pytest.mark.parametrize(
    ("omega", "phase"),
    [(u.Q(90, "deg/s"), u.Q(0, "deg")), (u.Q(0, "deg/s"), u.Q(90, "deg"))],
)
def test_rotation_about_axis_matches_euler(omega, phase):
    """Theta = omega*tau + phase: both terms reach the same 90 deg rotation."""
    b = cxfm.builders.RotationAboutAxis(omega, axis=ZHAT, phase=phase)
    want = cxfm.Rotate.from_euler("z", u.Q(90, "deg")).R
    assert jnp.allclose(b(u.Q(1.0, "s")).R, want, atol=1e-12)


@pytest.mark.parametrize("field", ["omega", "phase"])
def test_rotation_about_axis_differentiable_in_theta_terms(field):
    """d/dtheta sin(theta) at theta=0 is 1, via either theta term."""

    def y(val):
        kw = {"omega": u.Q(0.0, "rad/s"), "phase": u.Q(0.0, "rad")}
        kw[field] = u.Q(val, "rad/s" if field == "omega" else "rad")
        op = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(axis=ZHAT, **kw))
        out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
        return out["y"].ustrip("m")

    assert jnp.allclose(jax.grad(y)(0.0), 1.0, atol=1e-12)


def test_rotation_about_axis_differentiable_in_axis():
    """Gradient w.r.t. the axis *direction*, not just its scale.

    Varying only ``axis = [0, 0, axis_z]`` (the old form of this test) varies
    the axis's scale, which `RotationAboutAxis.__call__` normalizes away, so
    that gradient is analytically and empirically exactly 0.0 -- an
    `assert jnp.isfinite(grad)` would pass trivially even with a
    `stop_gradient` on the axis path. Varying ``ax_x`` instead tilts the axis
    direction, so the resulting rotation -- and hence the gradient -- actually
    depends on it.
    """
    omega = u.Q(90.0, "deg/s")

    def y(ax_x):
        axis = jnp.array([ax_x, 0.0, 1.0])
        op = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(omega, axis=axis))
        out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
        return out["y"].ustrip("m")

    ax_x0 = 0.4  # away from 0.0: the gradient there vanishes by symmetry
    grad = jax.grad(y)(ax_x0)
    eps = 1e-5
    finite_diff = (y(ax_x0 + eps) - y(ax_x0 - eps)) / (2 * eps)
    assert not jnp.allclose(grad, 0.0, atol=1e-8)
    assert jnp.allclose(grad, finite_diff, atol=1e-4)


def test_uniform_translation():
    rate = {"x": u.Q(3.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
    b = cxfm.builders.UniformTranslation(rate, chart=cxc.cart3d)
    op = b(u.Q(2.0, "s"))
    assert isinstance(op, cxfm.Translate)
    assert jnp.allclose(op.delta["x"].ustrip("km"), 6.0)


def test_uniform_translation_differentiable_in_rate():
    def y(rate_x):
        rate = {"x": u.Q(rate_x, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
        op = cxfm.TimeDep(cxfm.builders.UniformTranslation(rate, chart=cxc.cart3d))
        out = cxfm.act(op, u.Q(2.0, "s"), X, cxc.cart3d, cxr.point)
        return out["x"].ustrip("m")

    # x(tau) = 1 (initial) + rate_x * tau; d/drate_x = tau * (km->m factor)
    assert jnp.allclose(jax.grad(y)(3.0), 2000.0, atol=1e-6)


def test_rotation_about_axis_zero_axis_raises():
    """A zero-length axis must fail loudly, not normalize to a NaN `R`."""
    b = cxfm.builders.RotationAboutAxis(u.Q(1, "rad/s"), axis=jnp.zeros(3))
    with pytest.raises(Exception, match="must be non-zero"):
        b(u.Q(1.0, "s"))


class TestAxisAcceptsQuantity:
    """`axis` is normalised on use, so a unit on it cancels exactly.

    Regression: the field is annotated `Shaped[Array, "3"]` but had no
    converter, and `jnp` in this module is `quaxed.numpy`, whose `asarray`
    returns a `Quantity` unchanged. A `Quantity` axis was therefore stored as
    one, and `__call__` failed with `TypeError: Error interpreting argument to
    unvmap_any as a JAX value` -- naming nothing the caller had written.
    """

    OMEGA = u.Q(90, "deg/s")
    ZHAT = jnp.asarray([0.0, 0.0, 1.0])

    def _rotation(self, axis):
        return cxfm.builders.RotationAboutAxis(self.OMEGA, axis=axis)(u.Q(1.0, "s"))

    def test_bare_array_is_stored_bare(self):
        b = cxfm.builders.RotationAboutAxis(self.OMEGA, axis=self.ZHAT)
        assert not isinstance(b.axis, u.AbstractQuantity)

    @pytest.mark.parametrize("unit", ["", "m", "km"])
    def test_quantity_axis_gives_the_same_rotation(self, unit):
        """Scale and unit both cancel: only the direction survives."""
        ref = self._rotation(self.ZHAT)
        got = self._rotation(u.Q(2.0 * self.ZHAT, unit))
        assert bool(jnp.allclose(got.R, ref.R, atol=1e-12))

    def test_quantity_axis_is_not_stored_as_a_quantity(self):
        b = cxfm.builders.RotationAboutAxis(self.OMEGA, axis=u.Q(self.ZHAT, "m"))
        assert not isinstance(b.axis, u.AbstractQuantity)
