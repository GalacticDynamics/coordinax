"""Capability tests for builder-based curve frames.

These exercise what the old closure-based design could not express:

(a) gradients with respect to a *curve parameter* — a field of an
    `equinox.Module` curve, hence a real pytree leaf;
(b) a fixed-``gamma`` frame *field* along the curve, and ``vmap`` over
    ``gamma``.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm
import unxt as u

import coordinaxs.curveframes as cxfc

TAU = u.Q(0.4, "s")
P = u.Q(jnp.array([2.0, 1.0, -0.5]), "km")


class Helix(eqx.Module):
    """A helix whose ``radius`` (in km) is a differentiable pytree leaf."""

    radius: Any

    def __call__(self, tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        return u.Q(
            jnp.stack([self.radius * jnp.cos(t), self.radius * jnp.sin(t), 0.3 * t]),
            "km",
        )


class StaticHelix(eqx.Module):
    """Same helix, but ``radius`` is *static* — the pre-refactor behaviour."""

    radius: Any = eqx.field(static=True)

    def __call__(self, tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        return u.Q(
            jnp.stack([self.radius * jnp.cos(t), self.radius * jnp.sin(t), 0.3 * t]),
            "km",
        )


# A generic weighting, so the readout mixes all three frame axes: a scalar
# that depends on the full rotation, not just one row of it.
_W = jnp.array([1.0, -2.0, 0.5])


def _readout(builder: cxfc.AbstractCurveFrameBuilder) -> jax.Array:
    """Scalar readout: a fixed linear functional of P in the curve frame."""
    return jnp.dot(_W, cxfm.act(cxfm.TimeDep(builder), TAU, P).ustrip("km"))


# ===================================================================
# (a) Gradient w.r.t. a differentiable curve parameter


class TestGradThroughCurveParameter:
    """The curve's own parameters are differentiable pytree leaves."""

    def test_grad_is_nonzero_and_matches_finite_differences(self):
        r0 = 1.5

        def loss(radius):
            return _readout(cxfc.FrenetSerretBuilder(Helix(radius)))

        g = jax.grad(loss)(r0)

        # It must actually depend on the radius.
        assert abs(float(g)) > 1e-3

        # ... and agree with a central difference.
        h = 1e-5
        fd = (loss(r0 + h) - loss(r0 - h)) / (2 * h)
        assert jnp.allclose(g, fd, rtol=1e-5, atol=1e-7)

    def test_grad_through_the_whole_builder_pytree(self):
        """`jax.grad` over the builder returns a gradient in ``curve.radius``."""
        builder = cxfc.FrenetSerretBuilder(Helix(1.5))
        gbuilder = jax.grad(_readout)(builder)

        assert isinstance(gbuilder, cxfc.FrenetSerretBuilder)
        assert abs(float(gbuilder.curve.radius)) > 1e-3

        # The scalar-argument route must agree.
        def loss(radius):
            return _readout(cxfc.FrenetSerretBuilder(Helix(radius)))

        assert jnp.allclose(gbuilder.curve.radius, jax.grad(loss)(1.5))

    def test_static_radius_is_not_differentiable(self):
        """Discriminator: a *static* radius field kills the gradient.

        This is the pre-refactor failure mode — a curve parameter that is a
        trace-time constant rather than a leaf.
        """

        def loss(radius):
            return _readout(cxfc.FrenetSerretBuilder(StaticHelix(radius)))

        # equinox refuses to stash a tracer in a static field (UserWarning,
        # promoted to an error by the project's filterwarnings config).
        with pytest.raises((UserWarning, TypeError, ValueError)):
            jax.grad(loss)(1.5)

    def test_grad_through_bishop_curve_parameter(self):
        """Bishop's ODE path is differentiable in the curve parameter too."""
        r0 = 1.5

        def loss(radius):
            return _readout(cxfc.BishopBuilder(Helix(radius)))

        g = jax.grad(loss)(r0)
        assert abs(float(g)) > 1e-3

        h = 1e-4
        fd = (loss(r0 + h) - loss(r0 - h)) / (2 * h)
        assert jnp.allclose(g, fd, rtol=1e-3, atol=1e-5)


# ===================================================================
# (b) Fixed gamma: a frame *field* along the curve


def _circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


class TestFixedGamma:
    """With ``gamma`` set, the frame sits at a fixed point of the curve."""

    def test_gamma_frame_is_tau_independent(self):
        gamma = u.Q(0.7, "s")
        op = cxfm.TimeDep(cxfc.FrenetSerretBuilder(_circle, "s", gamma))

        a = cxfm.act(op, u.Q(0.0, "s"), P).ustrip("km")
        b = cxfm.act(op, u.Q(3.1, "s"), P).ustrip("km")
        assert jnp.allclose(a, b, atol=1e-10)

    def test_gamma_frame_matches_the_moving_frame_at_gamma(self):
        gamma = u.Q(0.7, "s")
        fixed = cxfm.TimeDep(cxfc.FrenetSerretBuilder(_circle, "s", gamma))
        moving = cxfm.TimeDep(cxfc.FrenetSerretBuilder(_circle))

        assert jnp.allclose(
            cxfm.act(fixed, u.Q(0.0, "s"), P).ustrip("km"),
            cxfm.act(moving, gamma, P).ustrip("km"),
            atol=1e-10,
        )

    def test_vmap_over_gamma(self):
        """A frame field: vmap the fixed curve parameter, not tau."""
        gammas = u.Q(jnp.linspace(0.0, 1.5, 5), "s")

        def at_gamma(g):
            op = cxfm.TimeDep(cxfc.FrenetSerretBuilder(_circle, "s", g))
            return cxfm.act(op, u.Q(0.0, "s"), P)

        batched = jax.vmap(at_gamma)(gammas).ustrip("km")
        assert batched.shape == (5, 3)

        for i in range(5):
            expected = at_gamma(gammas[i]).ustrip("km")
            assert jnp.allclose(batched[i], expected, atol=1e-6)

    def test_grad_w_r_t_gamma(self):
        """``gamma`` is a leaf, so the frame field is differentiable in it."""

        def loss(g):
            builder = cxfc.FrenetSerretBuilder(_circle, "s", u.Q(g, "s"))
            return _readout(builder)

        g0 = 0.7
        grad = jax.grad(loss)(g0)
        assert abs(float(grad)) > 1e-3

        h = 1e-5
        fd = (loss(g0 + h) - loss(g0 - h)) / (2 * h)
        assert jnp.allclose(grad, fd, rtol=1e-5, atol=1e-7)

    def test_bishop_gamma_frame(self):
        gamma = u.Q(0.7, "s")
        fixed = cxfm.TimeDep(cxfc.BishopBuilder(_circle, "s", gamma))
        moving = cxfm.TimeDep(cxfc.BishopBuilder(_circle))

        assert jnp.allclose(
            cxfm.act(fixed, u.Q(2.0, "s"), P).ustrip("km"),
            cxfm.act(moving, gamma, P).ustrip("km"),
            atol=1e-6,
        )

    def test_frame_from_curve_accepts_gamma(self):
        gamma = u.Q(0.7, "s")
        frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), _circle, gamma=gamma)
        assert frame.xop.builder.gamma is gamma

        op = cxf.frame_transition(cxf.Alice(), frame)
        assert jnp.allclose(
            cxfm.act(op, u.Q(0.0, "s"), P).ustrip("km"),
            cxfm.act(op, u.Q(2.0, "s"), P).ustrip("km"),
            atol=1e-10,
        )


class TestJitWithArrayLeaves:
    """A builder holding array leaves needs `eqx.filter_jit`, not `jax.jit`."""

    def test_filter_jit_works_where_jax_jit_cannot_hash(self):
        builder = cxfc.FrenetSerretBuilder(Helix(jnp.asarray(1.5)))

        # Plain jax.jit hashes the bound method, hence the module, hence the
        # array leaf -- the ergonomic cliff of the differentiable-curve design.
        with pytest.raises(TypeError, match="unhashable"):
            jax.jit(builder.rotation_matrix)(TAU)

        R = eqx.filter_jit(builder.rotation_matrix)(TAU)
        assert jnp.allclose(R, builder.rotation_matrix(TAU), atol=1e-10)
