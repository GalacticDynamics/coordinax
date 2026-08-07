"""Tests for the built-in TimeDep builders."""

import jax
import jax.numpy as jnp

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm

X = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}


def test_rotation_about_axis_matches_euler():
    b = cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=jnp.array([0.0, 0.0, 1.0]))
    R_t1 = b(u.Q(1.0, "s")).R
    want = cxfm.Rotate.from_euler("z", u.Q(90, "deg")).R
    assert jnp.allclose(R_t1, want, atol=1e-12)


def test_rotation_about_axis_phase():
    b = cxfm.RotationAboutAxis(
        u.Q(0, "deg/s"), axis=jnp.array([0.0, 0.0, 1.0]), phase=u.Q(90, "deg")
    )
    want = cxfm.Rotate.from_euler("z", u.Q(90, "deg")).R
    assert jnp.allclose(b(u.Q(3.0, "s")).R, want, atol=1e-12)


def test_rotation_about_axis_differentiable_in_omega():
    axis = jnp.array([0.0, 0.0, 1.0])

    def y(omega_val):
        op = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(omega_val, "rad/s"), axis=axis))
        out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
        return out["y"].ustrip("m")

    assert jnp.allclose(jax.grad(y)(0.0), 1.0, atol=1e-12)


def test_rotation_about_axis_differentiable_in_phase():
    axis = jnp.array([0.0, 0.0, 1.0])

    def y(phase_val):
        op = cxfm.TimeDep(
            cxfm.RotationAboutAxis(
                u.Q(0.0, "rad/s"), axis=axis, phase=u.Q(phase_val, "rad")
            )
        )
        out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
        return out["y"].ustrip("m")

    # d/dphase sin(phase) at phase=0 is cos(0) = 1
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
        op = cxfm.TimeDep(cxfm.RotationAboutAxis(omega, axis=axis))
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
    b = cxfm.UniformTranslation(rate, chart=cxc.cart3d)
    op = b(u.Q(2.0, "s"))
    assert isinstance(op, cxfm.Translate)
    assert jnp.allclose(op.delta["x"].ustrip("km"), 6.0)


def test_uniform_translation_differentiable_in_rate():
    def y(rate_x):
        rate = {
            "x": u.Q(rate_x, "km/s"),
            "y": u.Q(0.0, "km/s"),
            "z": u.Q(0.0, "km/s"),
        }
        op = cxfm.TimeDep(cxfm.UniformTranslation(rate, chart=cxc.cart3d))
        out = cxfm.act(op, u.Q(2.0, "s"), X, cxc.cart3d, cxr.point)
        return out["x"].ustrip("m")

    # x(tau) = 1 (initial) + rate_x * tau; d/drate_x = tau * (km->m factor)
    assert jnp.allclose(jax.grad(y)(3.0), 2000.0, atol=1e-6)
