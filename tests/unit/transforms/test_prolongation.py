"""Tests for the jet-prolongation engine and kinematic `act` semantics.

The keystone property tested throughout: every hand-written fast path must
equal the generic autodiff prolongation of the operator's point action.
"""

from jaxtyping import Array, Real

import jax
import pytest
from hypothesis import given, settings, strategies as st

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.main as cx
import coordinax.representations as cxr
import coordinax.transforms as cxfm

# ============================================================================
# Helpers


def q3(x, y, z, unit):
    return {"x": u.Q(x, unit), "y": u.Q(y, unit), "z": u.Q(z, unit)}


def allclose_cdict(a, b, unit, atol=1e-10):
    return all(
        jnp.allclose(u.ustrip(unit, a[k]), u.ustrip(unit, b[k]), atol=atol) for k in a
    )


def rot_z(t) -> Real[Array, "3 3"]:
    """Uniform rotation about z at 1 rad/s."""
    th = t.ustrip("s")
    st_, ct = jnp.sin(th), jnp.cos(th)
    return jnp.array([[ct, -st_, 0.0], [st_, ct, 0.0], [0.0, 0.0, 1.0]])


# ============================================================================
# tau_derivative


class TestTauDerivative:
    """Unit tests for `tau_derivative`."""

    def test_linear(self):
        delta = lambda t: {"x": u.Q(3.0, "km/s") * t, "y": u.Q(0.0, "km")}
        out = cxfm.tau_derivative(delta, u.Q(5.0, "s"))
        assert jnp.allclose(u.ustrip("km/s", out["x"]), 3.0)
        assert jnp.allclose(u.ustrip("km/s", out["y"]), 0.0)

    def test_second_derivative(self):
        delta = lambda t: {"x": u.Q(0.5, "m/s2") * t**2}
        out = cxfm.tau_derivative(delta, u.Q(4.0, "s"), n=2)
        assert jnp.allclose(u.ustrip("m/s2", out["x"]), 1.0)

    def test_n_zero_is_evaluation(self):
        delta = lambda t: {"x": u.Q(2.0, "m/s") * t}
        out = cxfm.tau_derivative(delta, u.Q(3.0, "s"), n=0)
        assert jnp.allclose(u.ustrip("m", out["x"]), 6.0)

    def test_raw_array_output(self):
        f = lambda t: jnp.array([1.0, 2.0]) * t.ustrip("s")
        out = cxfm.tau_derivative(f, u.Q(7.0, "s"))
        assert jnp.allclose(out, jnp.array([1.0, 2.0]))

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="n >= 0"):
            cxfm.tau_derivative(lambda t: t, u.Q(1.0, "s"), n=-1)

    def test_nonsi_time_units(self):
        delta = lambda t: {"x": u.Q(2.0, "km") * t.ustrip("Myr")}
        out = cxfm.tau_derivative(delta, u.Q(3.0, "Myr"))
        assert jnp.allclose(u.ustrip("km/Myr", out["x"]), 2.0)


# ============================================================================
# is_time_dependent


class TestIsTimeDependent:
    """Unit tests for `is_time_dependent`."""

    def test_static(self):
        assert not cxfm.is_time_dependent(cxfm.Translate.from_([1, 2, 3], "km"))
        assert not cxfm.is_time_dependent(cxfm.Identity())
        assert not cxfm.is_time_dependent(cxfm.Scale.from_factors([1.0, 2.0, 3.0]))

    def test_callable_delta(self):
        op = cxfm.Translate(
            lambda t: q3(t.ustrip("s"), 0.0, 0.0, "km"), chart=cxc.cart3d
        )
        assert cxfm.is_time_dependent(op)

    def test_composed(self):
        static = cxfm.Translate.from_([1, 2, 3], "km")
        moving = cxfm.Translate(
            lambda t: q3(t.ustrip("s"), 0.0, 0.0, "km"), chart=cxc.cart3d
        )
        assert cxfm.is_time_dependent(static | moving)
        assert not cxfm.is_time_dependent(static | cxfm.Identity())

    def test_inverse_of_time_dependent(self):
        moving = cxfm.Translate(
            lambda t: q3(t.ustrip("s"), 0.0, 0.0, "km"), chart=cxc.cart3d
        )
        assert cxfm.is_time_dependent(moving.inverse)


# ============================================================================
# Physics acid tests


class TestPhysics:
    """Analytic closed forms the prolongation must reproduce exactly."""

    def test_falling_frame(self):
        """Delta = 1/2 g t^2 => vel += g t, acc += g."""
        g = u.Q(9.8, "m/s2")
        op = cxfm.Translate(
            lambda t: {"x": 0.5 * g * t**2, "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
            chart=cxc.cart3d,
        )
        tau = u.Q(2.0, "s")
        v = q3(1.0, 2.0, 3.0, "m/s")
        a = q3(0.0, 0.0, 0.0, "m/s2")

        out_v = cxfm.act(op, tau, v, cxc.cart3d, cxr.coord_vel)
        assert jnp.allclose(u.ustrip("m/s", out_v["x"]), 1.0 + 9.8 * 2.0)
        assert jnp.allclose(u.ustrip("m/s", out_v["y"]), 2.0)

        out_a = cxfm.act(op, tau, a, cxc.cart3d, cxr.coord_acc)
        assert jnp.allclose(u.ustrip("m/s2", out_a["x"]), 9.8)

    def test_rotating_frame_velocity(self):
        """V' = R v + dR/dt x; at t=0: v + omega x_perp."""
        op = cxfm.Rotate.from_(rot_z)
        tau = u.Q(0.0, "s")
        at = q3(1.0, 0.0, 0.0, "m")
        v = q3(0.0, 0.0, 0.0, "m/s")
        out = cxfm.act(op, tau, v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at)
        # omega z-hat cross x-hat = y-hat
        assert jnp.allclose(u.ustrip("m/s", out["y"]), 1.0, atol=1e-8)
        assert jnp.allclose(u.ustrip("m/s", out["x"]), 0.0, atol=1e-8)

    def test_rotating_frame_acceleration_coriolis_centrifugal(self):
        """A' = R a + 2 dR v + ddR x; at t=0: a + 2 omega z x v - omega^2 x_perp."""
        op = cxfm.Rotate.from_(rot_z)
        tau = u.Q(0.0, "s")
        at = q3(1.0, 0.0, 0.0, "m")
        at_vel = q3(0.0, 1.0, 0.0, "m/s")
        a = q3(0.0, 0.0, 0.0, "m/s2")
        out = cxfm.act(
            op,
            tau,
            a,
            cxc.cart3d,
            cxr.tangent_geom,
            cxr.coord_acc,
            at=at,
            at_vel=at_vel,
        )
        # 2*Omega x v = 2 * (z-hat x y-hat) = -2 x-hat; ddR x = -x-hat
        assert jnp.allclose(u.ustrip("m/s2", out["x"]), -3.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("m/s2", out["y"]), 0.0, atol=1e-6)

    def test_boost_equals_prolonged_translate(self):
        """Boost(dv) == prolongation of Translate(dpl, lambda t: dv*t)."""
        dv = q3(1.5, -0.5, 2.0, "km/s")
        boost = cxfm.Boost(dv, chart=cxc.cart3d)
        td = cxfm.Translate(
            lambda t: {k: c * t for k, c in dv.items()}, chart=cxc.cart3d
        )
        tau = u.Q(3.0, "s")
        jet = {
            0: q3(1.0, 2.0, 3.0, "km"),
            1: q3(0.1, 0.2, 0.3, "km/s"),
            2: q3(0.0, 0.0, 0.0, "km/s2"),
        }
        out_td = cxfm.prolong(td, tau, jet, cxc.cart3d)
        out_p = cxfm.act(boost, tau, jet[0], cxc.cart3d, cxr.point)
        out_v = cxfm.act(boost, tau, jet[1], cxc.cart3d, cxr.coord_vel)
        out_a = cxfm.act(boost, tau, jet[2], cxc.cart3d, cxr.coord_acc)
        assert allclose_cdict(out_p, out_td[0], "km")
        assert allclose_cdict(out_v, out_td[1], "km/s")
        assert allclose_cdict(out_a, out_td[2], "km/s2")


# ============================================================================
# Keystone: hand fast paths == generic autodiff prolongation


class TestFastPathEqualsGeneric:
    """Hand-written fast paths must equal the generic autodiff rule."""

    @given(
        c0=st.floats(-5, 5),
        c1=st.floats(-5, 5),
        c2=st.floats(-5, 5),
        tau=st.floats(0.1, 10),
    )
    @settings(max_examples=20, deadline=None)
    def test_translate_polynomial_delta(self, c0, c1, c2, tau):
        """Hand ladder rule == generic prolongation for polynomial delta."""

        def delta(t):
            ts = t.ustrip("s")
            val = c0 + c1 * ts + c2 * ts**2
            return {"x": u.Q(val, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}

        op = cxfm.Translate(delta, chart=cxc.cart3d)
        tq = u.Q(tau, "s")
        jet = {
            0: q3(1.0, 2.0, 3.0, "km"),
            1: q3(0.5, -0.5, 0.0, "km/s"),
            2: q3(0.1, 0.0, -0.1, "km/s2"),
        }
        out_gen = cxfm.prolong(op, tq, jet, cxc.cart3d)
        out_v = cxfm.act(op, tq, jet[1], cxc.cart3d, cxr.coord_vel)
        out_a = cxfm.act(op, tq, jet[2], cxc.cart3d, cxr.coord_acc)
        assert allclose_cdict(out_v, out_gen[1], "km/s", atol=1e-6)
        assert allclose_cdict(out_a, out_gen[2], "km/s2", atol=1e-6)

    @given(tau=st.floats(0.0, 6.0))
    @settings(max_examples=20, deadline=None)
    def test_rotate_closed_form_vs_generic(self, tau):
        """Rotate's Cartesian vel closed form == generic prolongation."""
        op = cxfm.Rotate.from_(rot_z)
        tq = u.Q(tau, "s")
        at = q3(1.0, -2.0, 0.5, "m")
        v = q3(0.3, 0.1, -0.2, "m/s")
        out_hand = cxfm.act(
            op, tq, v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at
        )
        out_gen = cxfm.prolong(op, tq, {0: at, 1: v}, cxc.cart3d)
        assert allclose_cdict(out_hand, out_gen[1], "m/s", atol=1e-6)

    def test_vel_kick_translate_vs_generic_fibre_law(self):
        """TD vel-kick Translate: acc gains delta-dot (hand rule)."""
        kick = cxfm.Translate(
            lambda t: {
                "x": u.Q(5.0, "km/s2") * t,
                "y": u.Q(0.0, "km/s"),
                "z": u.Q(0.0, "km/s"),
            },
            chart=cxc.cart3d,
            semantic_kind=cxr.vel,
        )
        tau = u.Q(2.0, "s")
        a = q3(1.0, 1.0, 1.0, "km/s2")
        out = cxfm.act(kick, tau, a, cxc.cart3d, cxr.coord_acc)
        assert jnp.allclose(u.ustrip("km/s2", out["x"]), 6.0)
        assert jnp.allclose(u.ustrip("km/s2", out["y"]), 1.0)


# ============================================================================
# Structural properties


class TestStructure:
    """Structural identities of the prolongation calculus."""

    def test_pushforward_equals_act_for_static(self):
        op = cxfm.Rotate.from_euler("z", u.Q(37.0, "deg"))
        v = q3(1.0, 2.0, 3.0, "m/s")
        out_act = cxfm.act(op, None, v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel)
        out_pf = cxfm.pushforward(op, None, v, cxc.cart3d, cxr.coord_vel)
        assert allclose_cdict(out_act, out_pf, "m/s")

    def test_dpl_invariant_under_translates(self):
        d = q3(1.0, 2.0, 3.0, "km")
        tau = u.Q(2.0, "s")
        ops = [
            cxfm.Translate.from_([1, 2, 3], "km"),
            cxfm.Translate(
                lambda t: q3(t.ustrip("s"), 0.0, 0.0, "km"), chart=cxc.cart3d
            ),
            cxfm.Boost(q3(1.0, 0.0, 0.0, "km/s"), chart=cxc.cart3d),
        ]
        for op in ops:
            out = cxfm.act(op, tau, d, cxc.cart3d, cxr.coord_disp)
            assert allclose_cdict(out, d, "km")

    def test_prolong_inverse_roundtrip(self):
        moving = cxfm.Translate(
            lambda t: q3(3.0 * t.ustrip("s"), 0.0, 0.0, "km"), chart=cxc.cart3d
        )
        tau = u.Q(2.0, "s")
        jet = {0: q3(1.0, 2.0, 3.0, "km"), 1: q3(0.5, -0.5, 0.0, "km/s")}
        fwd = cxfm.prolong(moving, tau, jet, cxc.cart3d)
        back = cxfm.prolong(moving.inverse, tau, fwd, cxc.cart3d)
        assert allclose_cdict(back[0], jet[0], "km", atol=1e-6)
        assert allclose_cdict(back[1], jet[1], "km/s", atol=1e-6)

    def test_prolong_composed_equals_sequential(self):
        opA = cxfm.Boost(q3(1.0, 0.0, 0.0, "km/s"), chart=cxc.cart3d)
        opB = cxfm.Translate(
            lambda t: q3(0.0, 2.0 * t.ustrip("s"), 0.0, "km"), chart=cxc.cart3d
        )
        tau = u.Q(2.0, "s")
        jet = {0: q3(1.0, 2.0, 3.0, "km"), 1: q3(0.5, -0.5, 0.0, "km/s")}
        out_pipe = cxfm.prolong(opA | opB, tau, jet, cxc.cart3d)
        out_seq = cxfm.prolong(
            opB, tau, cxfm.prolong(opA, tau, jet, cxc.cart3d), cxc.cart3d
        )
        assert allclose_cdict(out_pipe[0], out_seq[0], "km")
        assert allclose_cdict(out_pipe[1], out_seq[1], "km/s")


# ============================================================================
# Units


class TestUnits:
    """Unit-handling through the prolongation engine."""

    def test_spherical_chart_mixed_units(self):
        """Prolongation in a spherical chart handles mixed (m, rad) units."""
        moving = cxfm.Translate(
            lambda t: q3(3.0 * t.ustrip("s"), 0.0, 0.0, "m"), chart=cxc.cart3d
        )
        tau = u.Q(2.0, "s")
        jet = {
            0: {
                "r": u.Q(1.0, "m"),
                "theta": u.Q(jnp.pi / 2, "rad"),
                "phi": u.Q(0.0, "rad"),
            },
            1: {
                "r": u.Q(0.0, "m/s"),
                "theta": u.Q(0.0, "rad/s"),
                "phi": u.Q(0.0, "rad/s"),
            },
        }
        out = cxfm.prolong(moving, tau, jet, cxc.sph3d)
        # point at (1+6, 0, 0) cartesian -> r = 7
        assert jnp.allclose(u.ustrip("m", out[0]["r"]), 7.0, atol=1e-6)
        # velocity gains delta-dot = 3 m/s radially (point on +x axis)
        assert jnp.allclose(u.ustrip("m/s", out[1]["r"]), 3.0, atol=1e-6)
        assert u.dimension_of(out[1]["theta"]) == u.dimension_of(u.Q(1, "rad/s"))

    def test_tau_in_myr(self):
        moving = cxfm.Translate(
            lambda t: q3(2.0 * t.ustrip("Myr"), 0.0, 0.0, "kpc"), chart=cxc.cart3d
        )
        tau = u.Q(3.0, "Myr")
        v = q3(0.0, 0.0, 0.0, "kpc/Myr")
        out = cxfm.act(moving, tau, v, cxc.cart3d, cxr.coord_vel)
        assert jnp.allclose(u.ustrip("kpc/Myr", out["x"]), 2.0)


# ============================================================================
# Batching / JAX transforms


class TestBatchingAndJit:
    """jit/vmap/batching compatibility."""

    def test_jit_prolong(self):
        moving = cxfm.Translate(
            lambda t: q3(3.0 * t.ustrip("s"), 0.0, 0.0, "km"), chart=cxc.cart3d
        )
        jet = {0: q3(0.0, 0.0, 0.0, "km"), 1: q3(0.0, 0.0, 0.0, "km/s")}
        f = jax.jit(lambda tau, jet: cxfm.prolong(moving, tau, jet, cxc.cart3d))
        out = f(u.Q(2.0, "s"), jet)
        assert jnp.allclose(u.ustrip("km/s", out[1]["x"]), 3.0)

    def test_vmap_over_tau(self):
        g = u.Q(2.0, "m/s2")
        moving = cxfm.Translate(
            lambda t: {"x": 0.5 * g * t**2, "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
            chart=cxc.cart3d,
        )
        jet = {0: q3(0.0, 0.0, 0.0, "m"), 1: q3(0.0, 0.0, 0.0, "m/s")}
        f = jax.jit(lambda tau: cxfm.prolong(moving, tau, jet, cxc.cart3d)[1]["x"])
        taus = u.Q(jnp.array([1.0, 2.0, 3.0]), "s")
        out = jax.vmap(f)(taus)
        assert jnp.allclose(u.ustrip("m/s", out), jnp.array([2.0, 4.0, 6.0]))

    def test_batched_data(self):
        moving = cxfm.Translate(
            lambda t: q3(3.0 * t.ustrip("s"), 0.0, 0.0, "km"), chart=cxc.cart3d
        )
        v = {
            "x": u.Q(jnp.zeros(4), "km/s"),
            "y": u.Q(jnp.ones(4), "km/s"),
            "z": u.Q(jnp.zeros(4), "km/s"),
        }
        out = cxfm.act(moving, u.Q(2.0, "s"), v, cxc.cart3d, cxr.coord_vel)
        assert out["x"].shape == (4,)
        assert jnp.allclose(u.ustrip("km/s", out["x"]), 3.0)


# ============================================================================
# Error paths


class TestErrors:
    """Informative errors when required jet slots are missing."""

    def test_td_rotate_lone_vel_requires_at(self):
        op = cxfm.Rotate.from_(rot_z)
        v = q3(1.0, 0.0, 0.0, "m/s")
        with pytest.raises(TypeError, match="requires the base point"):
            cxfm.act(op, u.Q(1.0, "s"), v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel)

    def test_td_rotate_acc_requires_at_vel(self):
        op = cxfm.Rotate.from_(rot_z)
        a = q3(1.0, 0.0, 0.0, "m/s2")
        at = q3(1.0, 0.0, 0.0, "m")
        with pytest.raises(TypeError, match="at_vel"):
            cxfm.act(
                op, u.Q(1.0, "s"), a, cxc.cart3d, cxr.tangent_geom, cxr.coord_acc, at=at
            )

    def test_td_translate_point_requires_tau(self):
        """Materializing a callable delta without tau raises informatively."""
        moving = cxfm.Translate(
            lambda t: q3(t.ustrip("s"), 0.0, 0.0, "km"), chart=cxc.cart3d
        )
        p = q3(0.0, 0.0, 0.0, "km")
        with pytest.raises(TypeError, match=r"time-dependent \(callable\) delta"):
            cxfm.act(moving, None, p, cxc.cart3d, cxr.point)

    def test_td_vel_kick_matching_order_requires_tau(self):
        """A callable vel-kick on velocity data (n==0) also needs tau."""
        kick = cxfm.Translate(
            lambda t: q3(t.ustrip("s"), 0.0, 0.0, "km/s"),
            chart=cxc.cart3d,
            semantic_kind=cxr.vel,
        )
        v = q3(0.0, 0.0, 0.0, "km/s")
        with pytest.raises(TypeError, match=r"time-dependent \(callable\) delta"):
            cxfm.act(kick, None, v, cxc.cart3d, cxr.coord_vel)

    def test_td_translate_tangent_requires_tau(self):
        moving = cxfm.Translate(
            lambda t: q3(t.ustrip("s"), 0.0, 0.0, "km"), chart=cxc.cart3d
        )
        v = q3(1.0, 0.0, 0.0, "km/s")
        with pytest.raises(TypeError, match="tau=None"):
            cxfm.act(moving, None, v, cxc.cart3d, cxr.coord_vel)

    def test_prolong_missing_slot(self):
        # A non-additive op: the generic chain needs every lower slot.
        op = cxfm.Rotate.from_(rot_z)
        jet = {0: q3(0.0, 0.0, 0.0, "m"), 2: q3(0.0, 0.0, 0.0, "m/s2")}
        with pytest.raises(TypeError, match="slot 1 is missing"):
            cxfm.prolong(op, u.Q(1.0, "s"), jet, cxc.cart3d)

    def test_prolong_additive_skips_intermediate_slots(self):
        # Additive ops prolong slot-wise: no intermediate slots required.
        moving = cxfm.Translate(
            lambda t: {
                "x": 0.5 * u.Q(2.0, "km/s2") * t**2,
                "y": u.Q(0.0, "km"),
                "z": u.Q(0.0, "km"),
            },
            chart=cxc.cart3d,
        )
        jet = {0: q3(0.0, 0.0, 0.0, "km"), 2: q3(0.0, 0.0, 0.0, "km/s2")}
        out = cxfm.prolong(moving, u.Q(1.0, "s"), jet, cxc.cart3d)
        assert jnp.allclose(u.ustrip("km/s2", out[2]["x"]), 2.0)

    def test_static_scale_vel_requires_at(self):
        op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
        v = q3(1.0, 1.0, 1.0, "m/s")
        with pytest.raises(TypeError, match="base point"):
            cxfm.act(op, None, v, cxc.cart3d, cxr.coord_vel)


# ============================================================================
# Coordinate bundles


class TestCoordinateBundle:
    """Joint prolongation of Coordinate bundles."""

    def test_td_translate_bundle(self):
        point = cx.Point.from_([1.0, 0.0, 0.0], "m")
        vel = cx.Tangent(q3(1.0, 0.0, 0.0, "m/s"), cxc.cart3d, cxr.coord_basis, cxr.vel)
        pv = cx.Coordinate(point=point, velocity=vel)
        op = cx.Translate(
            lambda t: q3(3.0 * t.ustrip("s"), 0.0, 0.0, "m"), chart=cxc.cart3d
        )
        out = cx.act(op, u.Q(2.0, "s"), pv)
        assert jnp.allclose(u.ustrip("m", out.point.data["x"]), 7.0)
        assert jnp.allclose(u.ustrip("m/s", out["velocity"].data["x"]), 4.0)

    def test_td_rotate_bundle(self):
        point = cx.Point.from_([1.0, 0.0, 0.0], "m")
        vel = cx.Tangent(q3(0.0, 0.0, 0.0, "m/s"), cxc.cart3d, cxr.coord_basis, cxr.vel)
        pv = cx.Coordinate(point=point, velocity=vel)
        op = cx.Rotate.from_(rot_z)
        out = cx.act(op, u.Q(0.0, "s"), pv)
        # v' = Rv + dR x = omega z-hat cross x-hat = y-hat
        assert jnp.allclose(u.ustrip("m/s", out["velocity"].data["y"]), 1.0, atol=1e-8)

    def test_static_bundle_unchanged_behavior(self):
        point = cx.Point.from_([1.0, 0.0, 0.0], "m")
        vel = cx.Tangent(q3(1.0, 0.0, 0.0, "m/s"), cxc.cart3d, cxr.coord_basis, cxr.vel)
        pv = cx.Coordinate(point=point, velocity=vel)
        op = cx.Translate.from_([1, 0, 0], "m")
        out = cx.act(op, None, pv)
        assert jnp.allclose(u.ustrip("m", out.point.data["x"]), 2.0)
        assert jnp.allclose(u.ustrip("m/s", out["velocity"].data["x"]), 1.0)


# ============================================================================
# Fibre-only offsets through the jet path


class TestFibreKickProlong:
    """Fibre-only offsets must survive the joint (jet) prolongation path.

    A `Translate(semantic_kind=vel)` has identity point action, so a
    point-action-only prolongation would drop it; the slot-wise `prolong`
    registered for additive operators keeps it.
    """

    def test_vel_kick_prolong_slotwise(self):
        kick = cxfm.Translate(
            q3(100.0, 0.0, 0.0, "m/s"), chart=cxc.cart3d, semantic_kind=cxr.vel
        )
        jet = {0: q3(1.0, 0.0, 0.0, "m"), 1: q3(1.0, 0.0, 0.0, "m/s")}
        out = cxfm.prolong(kick, None, jet, cxc.cart3d)
        assert jnp.allclose(u.ustrip("m", out[0]["x"]), 1.0)
        assert jnp.allclose(u.ustrip("m/s", out[1]["x"]), 101.0)

    def test_td_translate_composed_with_vel_kick_on_coordinate(self):
        """Coordinate jet path == bare-tangent path for TD op | vel-kick."""
        moving = cxfm.Translate(
            lambda t: {
                "x": u.Q(3.0, "m/s") * t,
                "y": u.Q(0.0, "m"),
                "z": u.Q(0.0, "m"),
            },
            chart=cxc.cart3d,
        )
        kick = cxfm.Translate(
            q3(100.0, 0.0, 0.0, "m/s"), chart=cxc.cart3d, semantic_kind=cxr.vel
        )
        op = moving | kick
        tau = u.Q(2.0, "s")

        pv = cx.Coordinate(
            point=cx.Point.from_([1.0, 0.0, 0.0], "m"),
            velocity=cx.Tangent.from_([1.0, 0.0, 0.0], "m/s"),
        )
        out = cx.act(op, tau, pv)
        # v' = v + delta-dot + kick = 1 + 3 + 100
        assert jnp.allclose(u.ustrip("m/s", out["velocity"].data["x"]), 104.0)

        # and it matches the bare-tangent path
        v = q3(1.0, 0.0, 0.0, "m/s")
        at = q3(1.0, 0.0, 0.0, "m")
        bare = cxfm.act(op, tau, v, cxc.cart3d, cxr.coord_vel, at=at)
        assert jnp.allclose(
            u.ustrip("m/s", out["velocity"].data["x"]), u.ustrip("m/s", bare["x"])
        )

    def test_galilean_boost_prolong_slotwise(self):
        """Boost's prolong (via AbstractAdd) matches its act closed forms."""
        boost = cxfm.Boost(q3(1.0, 0.0, 0.0, "km/s"), chart=cxc.cart3d)
        tau = u.Q(3.0, "s")
        jet = {
            0: q3(1.0, 2.0, 3.0, "km"),
            1: q3(0.5, 0.0, 0.0, "km/s"),
            2: q3(0.1, 0.0, 0.0, "km/s2"),
        }
        out = cxfm.prolong(boost, tau, jet, cxc.cart3d)
        assert jnp.allclose(u.ustrip("km", out[0]["x"]), 4.0)  # x + dv*tau
        assert jnp.allclose(u.ustrip("km/s", out[1]["x"]), 1.5)  # v + dv
        assert jnp.allclose(u.ustrip("km/s2", out[2]["x"]), 0.1)  # a unchanged
