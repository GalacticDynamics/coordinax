"""Tests for the built-in Parametric builders."""

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
        op = cxfm.Parametric(cxfm.RotationAboutAxis(u.Q(omega_val, "rad/s"), axis=axis))
        out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
        return out["y"].ustrip("m")

    assert jnp.allclose(jax.grad(y)(0.0), 1.0, atol=1e-12)


def test_rotation_about_axis_differentiable_in_phase():
    axis = jnp.array([0.0, 0.0, 1.0])

    def y(phase_val):
        op = cxfm.Parametric(
            cxfm.RotationAboutAxis(
                u.Q(0.0, "rad/s"), axis=axis, phase=u.Q(phase_val, "rad")
            )
        )
        out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
        return out["y"].ustrip("m")

    # d/dphase sin(phase) at phase=0 is cos(0) = 1
    assert jnp.allclose(jax.grad(y)(0.0), 1.0, atol=1e-12)


def test_rotation_about_axis_differentiable_in_axis():
    omega = u.Q(90.0, "deg/s")

    def y(axis_z):
        axis = jnp.array([0.0, 0.0, axis_z])
        op = cxfm.Parametric(cxfm.RotationAboutAxis(omega, axis=axis))
        out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
        return out["y"].ustrip("m")

    grad = jax.grad(y)(1.0)
    assert jnp.isfinite(grad)


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
        op = cxfm.Parametric(cxfm.UniformTranslation(rate, chart=cxc.cart3d))
        out = cxfm.act(op, u.Q(2.0, "s"), X, cxc.cart3d, cxr.point)
        return out["x"].ustrip("m")

    # x(tau) = 1 (initial) + rate_x * tau; d/drate_x = tau * (km->m factor)
    assert jnp.allclose(jax.grad(y)(3.0), 2000.0, atol=1e-6)
