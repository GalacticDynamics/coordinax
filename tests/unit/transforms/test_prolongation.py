"""Tests for the jet-prolongation engine and kinematic `act` semantics.

The keystone property tested throughout: every hand-written fast path must
equal the generic autodiff prolongation of the operator's point action.
"""

from typing import ClassVar

import jax
import pytest
from hypothesis import given, settings, strategies as st

import quaxed.numpy as jnp
import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import coordinax.vectors as cxv
from .conftest import rot_z
from coordinax.transforms._src.actions.prolong import prolong_jet
from coordinax.transforms._src.actions.utils import is_flat_chart

# ============================================================================
# Helpers


def q3(x, y, z, unit):
    return {"x": u.Q(x, unit), "y": u.Q(y, unit), "z": u.Q(z, unit)}


def allclose_cdict(a, b, unit, atol=1e-10):
    return all(
        jnp.allclose(u.ustrip(unit, a[k]), u.ustrip(unit, b[k]), atol=atol) for k in a
    )


SPH_AT = {"r": u.Q(5.0, "km"), "theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
SPH_V = {"r": u.Q(0.3, "km/s"), "theta": u.Q(0.01, "rad/s"), "phi": u.Q(0.02, "rad/s")}


def rot_z_op() -> cxfm.TimeDep:
    """Uniform rotation about z at 1 rad/s, as a `TimeDep` family."""
    return cxfm.TimeDep(rot_z(u.Q(1.0, "rad/s")))


def _rot_z_raw_op() -> cxfm.TimeDep:
    """Rotation about z at 1 rad per unit of a raw (unitless) tau."""

    def build(t):
        st_, ct = jnp.sin(t), jnp.cos(t)
        return cxfm.Rotate(
            jnp.array([[ct, -st_, 0.0], [st_, ct, 0.0], [0.0, 0.0, 1.0]])
        )

    return cxfm.TimeDep.from_(build)


def uniform_translate(vx, unit="km/s", chart=cxc.cart3d) -> cxfm.TimeDep:
    """A `TimeDep` uniform translation along +x at rate ``vx``."""
    return cxfm.TimeDep(
        cxfm.builders.UniformTranslation(q3(vx, 0.0, 0.0, unit), chart=chart)
    )


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
    """Unit tests for `is_time_dependent`.

    `is_time_dependent` is a declared trait (`AbstractTransform.is_time_dependent`),
    not a leaf-callable scan: only `TimeDep` (and anything composed from it,
    or `Boost`, whose point action is intrinsically tau-dependent) reports
    `True`.
    """

    def test_static(self):
        assert not cxfm.is_time_dependent(cxfm.Translate.from_([1, 2, 3], "km"))
        assert not cxfm.is_time_dependent(cxfm.Identity())
        assert not cxfm.is_time_dependent(cxfm.Scale.from_factors([1.0, 2.0, 3.0]))

    def test_timedep(self):
        assert cxfm.is_time_dependent(uniform_translate(1.0))

    def test_boost(self):
        # Boost's point action is delta*tau even for a constant delta.
        assert cxfm.is_time_dependent(cxfm.Boost.from_([1.0, 0, 0], "km/s"))

    def test_composed(self):
        static = cxfm.Translate.from_([1, 2, 3], "km")
        moving = uniform_translate(1.0)
        assert cxfm.is_time_dependent(static | moving)
        assert not cxfm.is_time_dependent(static | cxfm.Identity())

    def test_inverse_of_time_dependent(self):
        moving = uniform_translate(1.0)
        assert cxfm.is_time_dependent(moving.inverse)


# ============================================================================
# Physics acid tests


class TestPhysics:
    """Analytic closed forms the prolongation must reproduce exactly."""

    def test_falling_frame(self):
        """Delta = 1/2 g t^2 => vel += g t, acc += g."""
        g = u.Q(9.8, "m/s2")
        op = cxfm.TimeDep.from_(
            lambda t: cxfm.Translate(
                {"x": 0.5 * g * t**2, "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
                chart=cxc.cart3d,
            )
        )
        tau = u.Q(2.0, "s")
        at = q3(0.0, 0.0, 0.0, "m")
        v = q3(1.0, 2.0, 3.0, "m/s")
        a = q3(0.0, 0.0, 0.0, "m/s2")

        out_v = cxfm.act(op, tau, v, cxc.cart3d, cxr.coord_vel, at=at)
        assert jnp.allclose(u.ustrip("m/s", out_v["x"]), 1.0 + 9.8 * 2.0)
        assert jnp.allclose(u.ustrip("m/s", out_v["y"]), 2.0)

        out_a = cxfm.act(op, tau, a, cxc.cart3d, cxr.coord_acc, at_jet={0: at, 1: v})
        assert jnp.allclose(u.ustrip("m/s2", out_a["x"]), 9.8)

    def test_rotating_frame_velocity(self):
        """V' = R v + dR/dt x; at t=0: v + omega x_perp."""
        op = rot_z_op()
        tau = u.Q(0.0, "s")
        at = q3(1.0, 0.0, 0.0, "m")
        v = q3(0.0, 0.0, 0.0, "m/s")
        out = cxfm.act(op, tau, v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at)
        # omega z-hat cross x-hat = y-hat
        assert jnp.allclose(u.ustrip("m/s", out["y"]), 1.0, atol=1e-8)
        assert jnp.allclose(u.ustrip("m/s", out["x"]), 0.0, atol=1e-8)

    def test_rotating_frame_acceleration_coriolis_centrifugal(self):
        """A' = R a + 2 dR v + ddR x; at t=0: a + 2 omega z x v - omega^2 x_perp."""
        op = rot_z_op()
        tau = u.Q(0.0, "s")
        at = q3(1.0, 0.0, 0.0, "m")
        vel = q3(0.0, 1.0, 0.0, "m/s")
        a = q3(0.0, 0.0, 0.0, "m/s2")
        out = cxfm.act(
            op,
            tau,
            a,
            cxc.cart3d,
            cxr.tangent_geom,
            cxr.coord_acc,
            at_jet={0: at, 1: vel},
        )
        # 2*Omega x v = 2 * (z-hat x y-hat) = -2 x-hat; ddR x = -x-hat
        assert jnp.allclose(u.ustrip("m/s2", out["x"]), -3.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("m/s2", out["y"]), 0.0, atol=1e-6)

    def test_boost_equals_prolonged_translate(self):
        """Boost(dv) == prolongation of TimeDep(UniformTranslation(dv))."""
        dv = q3(1.5, -0.5, 2.0, "km/s")
        boost = cxfm.Boost(dv, chart=cxc.cart3d)
        td = cxfm.TimeDep(cxfm.builders.UniformTranslation(dv, chart=cxc.cart3d))
        tau = u.Q(3.0, "s")
        jet = {
            0: q3(1.0, 2.0, 3.0, "km"),
            1: q3(0.1, 0.2, 0.3, "km/s"),
            2: q3(0.0, 0.0, 0.0, "km/s2"),
        }
        out_td = cxfm.act_jet(td, tau, jet, cxc.cart3d)
        out_p = cxfm.act(boost, tau, jet[0], cxc.cart3d, cxr.point)
        out_v = cxfm.act(boost, tau, jet[1], cxc.cart3d, cxr.coord_vel)
        out_a = cxfm.act(boost, tau, jet[2], cxc.cart3d, cxr.coord_acc)
        assert allclose_cdict(out_p, out_td[0], "km")
        assert allclose_cdict(out_v, out_td[1], "km/s")
        assert allclose_cdict(out_a, out_td[2], "km/s2")


# ============================================================================
# Keystone: hand fast paths == generic autodiff prolongation


class TestFastPathEqualsGeneric:
    r"""Hand-written fast paths must equal the generic autodiff rule.

    `Rotate`'s time-dependent closed form no longer exists (its matrix is
    always constant, and time dependence lives in `TimeDep`), so its case
    below pins the generic prolongation against the *hand-derived* closed
    form $v' = R v + \dot R x$ instead — same numeric oracle.
    """

    @given(
        c0=st.floats(-5, 5),
        c1=st.floats(-5, 5),
        c2=st.floats(-5, 5),
        tau=st.floats(0.1, 10),
    )
    @settings(max_examples=5)
    def test_translate_polynomial_delta(self, c0, c1, c2, tau):
        """Hand ladder rule == generic prolongation for polynomial delta."""

        def delta(t):
            ts = t.ustrip("s")
            val = c0 + c1 * ts + c2 * ts**2
            return {"x": u.Q(val, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}

        op = cxfm.TimeDep.from_(lambda t: cxfm.Translate(delta(t), chart=cxc.cart3d))
        tq = u.Q(tau, "s")
        jet = {
            0: q3(1.0, 2.0, 3.0, "km"),
            1: q3(0.5, -0.5, 0.0, "km/s"),
            2: q3(0.1, 0.0, -0.1, "km/s2"),
        }
        out_gen = cxfm.act_jet(op, tq, jet, cxc.cart3d)
        out_v = cxfm.act(op, tq, jet[1], cxc.cart3d, cxr.coord_vel, at=jet[0])
        out_a = cxfm.act(
            op, tq, jet[2], cxc.cart3d, cxr.coord_acc, at_jet={0: jet[0], 1: jet[1]}
        )
        assert allclose_cdict(out_v, out_gen[1], "km/s", atol=1e-6)
        assert allclose_cdict(out_a, out_gen[2], "km/s2", atol=1e-6)

    @given(tau=st.floats(0.0, 6.0))
    @settings(max_examples=5)
    def test_rotate_prolongation_vs_closed_form(self, tau):
        """TimeDep-rotate prolongation == the closed form v' = R v + Rdot x."""
        op = rot_z_op()
        tq = u.Q(tau, "s")
        at = q3(1.0, -2.0, 0.5, "m")
        v = q3(0.3, 0.1, -0.2, "m/s")
        out = cxfm.act(op, tq, v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at)
        out_gen = cxfm.act_jet(op, tq, {0: at, 1: v}, cxc.cart3d)
        assert allclose_cdict(out, out_gen[1], "m/s", atol=1e-6)

        # Hand-derived closed form for omega = 1 rad/s about z.
        st_, ct = jnp.sin(tau), jnp.cos(tau)
        rot = jnp.array([[ct, -st_, 0.0], [st_, ct, 0.0], [0.0, 0.0, 1.0]])
        rotdot = jnp.array([[-st_, -ct, 0.0], [ct, -st_, 0.0], [0.0, 0.0, 0.0]])
        v_arr = jnp.array([u.ustrip("m/s", v[k]) for k in "xyz"])
        at_arr = jnp.array([u.ustrip("m", at[k]) for k in "xyz"])
        expect = rot @ v_arr + rotdot @ at_arr
        got = jnp.array([u.ustrip("m/s", out[k]) for k in "xyz"])
        assert jnp.allclose(got, expect, atol=1e-6)

    def test_vel_kick_translate_vs_generic_fibre_law(self):
        """TD vel-kick Translate: acc gains delta-dot (hand rule)."""
        kick = cxfm.TimeDep.from_(
            lambda t: cxfm.Translate(
                {
                    "x": u.Q(5.0, "km/s2") * t,
                    "y": u.Q(0.0, "km/s"),
                    "z": u.Q(0.0, "km/s"),
                },
                chart=cxc.cart3d,
                semantic_kind=cxr.vel,
            )
        )
        tau = u.Q(2.0, "s")
        a = q3(1.0, 1.0, 1.0, "km/s2")
        out = cxfm.act(kick, tau, a, cxc.cart3d, cxr.coord_acc)
        assert jnp.allclose(u.ustrip("km/s2", out["x"]), 6.0)
        assert jnp.allclose(u.ustrip("km/s2", out["y"]), 1.0)

    def test_point_acting_timedeps_stay_on_the_generic_funnel(self):
        r"""Guard: the fibre-offset carve-out must not swallow point actions.

        The ladder rule for `TimeDep`-wrapped *fibre* offsets (ladder order
        $k \geq 1$, identity point action) bypasses the generic tangent
        funnel. If that predicate is ever widened to a transform with a real
        point action, the funnel's differentiation — and with it the $\dot R
        x$ / $\dot\gamma$ terms — is silently lost. These two pin that the
        point-acting families still get those terms.

        Both halves must DISCRIMINATE, i.e. fail under a widened predicate. In
        a flat, matching chart the ladder rule and the funnel agree
        numerically, so the order-0 half deliberately uses a non-flat (
        spherical) data chart, where the point-Jacobian coupling is in play and
        the two answers genuinely differ.
        """
        # A time-dependent ROTATION: the velocity must gain the Rdot x term.
        rot = rot_z_op()
        at = q3(1.0, 0.0, 0.0, "m")
        v = q3(0.0, 0.0, 0.0, "m/s")
        out = cxfm.act(rot, u.Q(0.0, "s"), v, cxc.cart3d, cxr.coord_vel, at=at)
        # omega z-hat x x-hat = y-hat: zero without the funnel's derivative.
        assert jnp.allclose(u.ustrip("m/s", out["y"]), 1.0, atol=1e-8)

        # A time-dependent order-0 TRANSLATE is NOT a fibre offset either: it
        # has a real point action, so it must reach the generic funnel. On a
        # spherical data chart the componentwise ladder rule would give a
        # materially different (wrong) answer, so this pins the k=0 boundary.
        moving = uniform_translate(3.0)
        tau = u.Q(2.0, "s")
        usys = u.unitsystems.si
        out_t = cxfm.act(
            moving, tau, SPH_V, cxc.sph3d, cxr.coord_vel, at=SPH_AT, usys=usys
        )
        gen = prolong_jet(moving, tau, {0: SPH_AT, 1: SPH_V}, cxc.sph3d, usys=usys)
        for k in out_t:
            unit = u.unit_of(gen[1][k])
            assert jnp.allclose(u.ustrip(unit, out_t[k]), gen[1][k].value, rtol=1e-6)


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
            uniform_translate(1.0),
            cxfm.Boost(q3(1.0, 0.0, 0.0, "km/s"), chart=cxc.cart3d),
        ]
        for op in ops:
            out = cxfm.act(op, tau, d, cxc.cart3d, cxr.coord_disp)
            assert allclose_cdict(out, d, "km")

    def test_prolong_inverse_roundtrip(self):
        moving = uniform_translate(3.0)
        tau = u.Q(2.0, "s")
        jet = {0: q3(1.0, 2.0, 3.0, "km"), 1: q3(0.5, -0.5, 0.0, "km/s")}
        fwd = cxfm.act_jet(moving, tau, jet, cxc.cart3d)
        back = cxfm.act_jet(moving.inverse, tau, fwd, cxc.cart3d)
        assert allclose_cdict(back[0], jet[0], "km", atol=1e-6)
        assert allclose_cdict(back[1], jet[1], "km/s", atol=1e-6)

    def test_prolong_composed_equals_sequential(self):
        opA = cxfm.Boost(q3(1.0, 0.0, 0.0, "km/s"), chart=cxc.cart3d)
        opB = cxfm.TimeDep(
            cxfm.builders.UniformTranslation(
                q3(0.0, 2.0, 0.0, "km/s"), chart=cxc.cart3d
            )
        )
        tau = u.Q(2.0, "s")
        jet = {0: q3(1.0, 2.0, 3.0, "km"), 1: q3(0.5, -0.5, 0.0, "km/s")}
        out_pipe = cxfm.act_jet(opA | opB, tau, jet, cxc.cart3d)
        out_seq = cxfm.act_jet(
            opB, tau, cxfm.act_jet(opA, tau, jet, cxc.cart3d), cxc.cart3d
        )
        assert allclose_cdict(out_pipe[0], out_seq[0], "km")
        assert allclose_cdict(out_pipe[1], out_seq[1], "km/s")


# ============================================================================
# Units


class TestUnits:
    """Unit-handling through the prolongation engine."""

    def test_spherical_chart_mixed_units(self):
        """Prolongation in a spherical chart handles mixed (m, rad) units."""
        moving = uniform_translate(3.0, "m/s")
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
        out = cxfm.act_jet(moving, tau, jet, cxc.sph3d)
        # point at (1+6, 0, 0) cartesian -> r = 7
        assert jnp.allclose(u.ustrip("m", out[0]["r"]), 7.0, atol=1e-6)
        # velocity gains delta-dot = 3 m/s radially (point on +x axis)
        assert jnp.allclose(u.ustrip("m/s", out[1]["r"]), 3.0, atol=1e-6)
        assert u.dimension_of(out[1]["theta"]) == u.dimension_of(u.Q(1, "rad/s"))

    def test_tau_in_myr(self):
        moving = uniform_translate(2.0, "kpc/Myr")
        tau = u.Q(3.0, "Myr")
        at = q3(0.0, 0.0, 0.0, "kpc")
        v = q3(0.0, 0.0, 0.0, "kpc/Myr")
        out = cxfm.act(moving, tau, v, cxc.cart3d, cxr.coord_vel, at=at)
        assert jnp.allclose(u.ustrip("kpc/Myr", out["x"]), 2.0)


# ============================================================================
# Batching / JAX transforms


class TestBatchingAndJit:
    """jit/vmap/batching compatibility."""

    def test_jit_prolong(self):
        moving = uniform_translate(3.0)
        jet = {0: q3(0.0, 0.0, 0.0, "km"), 1: q3(0.0, 0.0, 0.0, "km/s")}
        f = jax.jit(lambda tau, jet: cxfm.act_jet(moving, tau, jet, cxc.cart3d))
        out = f(u.Q(2.0, "s"), jet)
        assert jnp.allclose(u.ustrip("km/s", out[1]["x"]), 3.0)

    def test_vmap_over_tau(self):
        g = u.Q(2.0, "m/s2")
        moving = cxfm.TimeDep.from_(
            lambda t: cxfm.Translate(
                {"x": 0.5 * g * t**2, "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
                chart=cxc.cart3d,
            )
        )
        jet = {0: q3(0.0, 0.0, 0.0, "m"), 1: q3(0.0, 0.0, 0.0, "m/s")}
        f = jax.jit(lambda tau: cxfm.act_jet(moving, tau, jet, cxc.cart3d)[1]["x"])
        taus = u.Q(jnp.array([1.0, 2.0, 3.0]), "s")
        out = jax.vmap(f)(taus)
        assert jnp.allclose(u.ustrip("m/s", out), jnp.array([2.0, 4.0, 6.0]))

    def test_batched_data(self):
        moving = uniform_translate(3.0)
        v = {
            "x": u.Q(jnp.zeros(4), "km/s"),
            "y": u.Q(jnp.ones(4), "km/s"),
            "z": u.Q(jnp.zeros(4), "km/s"),
        }
        at = {k: u.Q(jnp.zeros(4), "km") for k in "xyz"}
        out = cxfm.act(moving, u.Q(2.0, "s"), v, cxc.cart3d, cxr.coord_vel, at=at)
        assert out["x"].shape == (4,)
        assert jnp.allclose(u.ustrip("km/s", out["x"]), 3.0)


# ============================================================================
# Error paths


class TestErrors:
    """Informative errors when required jet slots are missing."""

    def test_td_rotate_lone_vel_requires_at(self):
        op = rot_z_op()
        v = q3(1.0, 0.0, 0.0, "m/s")
        with pytest.raises(TypeError, match="requires the base point"):
            cxfm.act(op, u.Q(1.0, "s"), v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel)

    def test_td_rotate_acc_requires_slot_one(self):
        op = rot_z_op()
        a = q3(1.0, 0.0, 0.0, "m/s2")
        at = q3(1.0, 0.0, 0.0, "m")
        with pytest.raises(TypeError, match=r"slot\(s\) \[1\] are missing"):
            cxfm.act(
                op, u.Q(1.0, "s"), a, cxc.cart3d, cxr.tangent_geom, cxr.coord_acc, at=at
            )

    def test_td_rotate_pushforward_requires_tau(self):
        """Materializing a `TimeDep` without tau raises informatively."""
        op = rot_z_op()
        d = q3(1.0, 0.0, 0.0, "m")
        with pytest.raises(TypeError, match="requires a time parameter"):
            cxfm.pushforward(op, None, d, cxc.cart3d, cxr.coord_disp)
        with pytest.raises(TypeError, match="requires a time parameter"):
            cxfm.act(op, None, d, cxc.cart3d, cxr.point)

    def test_td_translate_point_requires_tau(self):
        """A `TimeDep` point action without tau raises informatively."""
        moving = uniform_translate(1.0)
        p = q3(0.0, 0.0, 0.0, "km")
        with pytest.raises(TypeError, match="requires a time parameter"):
            cxfm.act(moving, None, p, cxc.cart3d, cxr.point)

    def test_td_vel_kick_matching_order_requires_tau(self):
        """A `TimeDep` vel-kick on velocity data also needs tau."""
        kick = cxfm.TimeDep.from_(
            lambda t: cxfm.Translate(
                q3(t.ustrip("s"), 0.0, 0.0, "km/s"),
                chart=cxc.cart3d,
                semantic_kind=cxr.vel,
            )
        )
        v = q3(0.0, 0.0, 0.0, "km/s")
        with pytest.raises(TypeError, match="requires a time parameter"):
            cxfm.act(kick, None, v, cxc.cart3d, cxr.coord_vel)

    def test_td_translate_tangent_requires_tau(self):
        moving = uniform_translate(1.0)
        v = q3(1.0, 0.0, 0.0, "km/s")
        with pytest.raises(TypeError, match="tau=None"):
            cxfm.act(moving, None, v, cxc.cart3d, cxr.coord_vel)

    def test_prolong_missing_slot(self):
        # A non-additive op: the generic chain needs every lower slot.
        op = rot_z_op()
        jet = {0: q3(0.0, 0.0, 0.0, "m"), 2: q3(0.0, 0.0, 0.0, "m/s2")}
        with pytest.raises(TypeError, match="slot 1 is missing"):
            cxfm.act_jet(op, u.Q(1.0, "s"), jet, cxc.cart3d)

    def test_prolong_additive_missing_slot0(self):
        # The componentwise (additive) path indexes jet[0] for tangent slots;
        # a jet without slot 0 must raise the same TypeError as the generic
        # engine, not a bare KeyError.
        kick = cxfm.Translate(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
            chart=cxc.cart3d,
            semantic_kind=cxr.vel,
        )
        jet = {1: q3(0.0, 0.0, 0.0, "m/s")}
        with pytest.raises(TypeError, match="jet slot 0"):
            cxfm.act_jet(kick, None, jet, cxc.cart3d)

    def test_prolong_additive_skips_intermediate_slots(self):
        """Additive ops prolong slot-wise: no intermediate slots required.

        Slot 2 gains the operator's own order-2 offset (2 km/s2) with slot 1
        absent; the generic (jet-chain) engine would demand slot 1.
        """
        kick = cxfm.Translate(
            q3(2.0, 0.0, 0.0, "km/s2"), chart=cxc.cart3d, semantic_kind=cxr.acc
        )
        jet = {0: q3(0.0, 0.0, 0.0, "km"), 2: q3(0.0, 0.0, 0.0, "km/s2")}
        out = cxfm.act_jet(kick, None, jet, cxc.cart3d)
        assert jnp.allclose(u.ustrip("km/s2", out[2]["x"]), 2.0)

    def test_static_scale_vel_no_at_needed(self):
        # A static (time-independent) linear map has a constant Jacobian equal
        # to its matrix, so a Cartesian velocity transforms as v -> M v with no
        # base point required (matching Rotate).
        op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
        v = q3(1.0, 1.0, 1.0, "m/s")
        out = cxfm.act(op, None, v, cxc.cart3d, cxr.coord_vel)
        assert jnp.allclose(u.ustrip("m/s", out["x"]), 2.0)
        assert jnp.allclose(u.ustrip("m/s", out["y"]), 3.0)
        assert jnp.allclose(u.ustrip("m/s", out["z"]), 4.0)

    def test_static_scale_vel_noncartesian_still_requires_at(self):
        # On a non-Cartesian chart the Jacobian varies with position, so the
        # base point is still required.
        op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
        v = {"r": u.Q(1.0, "m/s"), "theta": u.Q(0.0, "rad/s"), "phi": u.Q(0.0, "rad/s")}
        with pytest.raises(TypeError, match="'at'"):
            cxfm.act(op, None, v, cxc.sph3d, cxr.coord_vel)


# ============================================================================
# Coordinate bundles


class TestCoordinateBundle:
    """Joint prolongation of Coordinate bundles."""

    def test_td_translate_bundle(self):
        point = cx.Point.from_([1.0, 0.0, 0.0], "m")
        vel = cx.Tangent(q3(1.0, 0.0, 0.0, "m/s"), cxc.cart3d, cxr.coord_basis, cxr.vel)
        pv = cx.Coordinate(point=point, velocity=vel)
        op = uniform_translate(3.0, "m/s")
        out = cx.act(op, u.Q(2.0, "s"), pv)
        assert jnp.allclose(u.ustrip("m", out.point.data["x"]), 7.0)
        assert jnp.allclose(u.ustrip("m/s", out["velocity"].data["x"]), 4.0)

    def test_td_rotate_bundle(self):
        point = cx.Point.from_([1.0, 0.0, 0.0], "m")
        vel = cx.Tangent(q3(0.0, 0.0, 0.0, "m/s"), cxc.cart3d, cxr.coord_basis, cxr.vel)
        pv = cx.Coordinate(point=point, velocity=vel)
        op = rot_z_op()
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
        out = cxfm.act_jet(kick, None, jet, cxc.cart3d)
        assert jnp.allclose(u.ustrip("m", out[0]["x"]), 1.0)
        assert jnp.allclose(u.ustrip("m/s", out[1]["x"]), 101.0)

    def test_td_translate_composed_with_vel_kick_on_coordinate(self):
        """Coordinate jet path == bare-tangent path for TD op | vel-kick."""
        moving = uniform_translate(3.0, "m/s")
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
        out = cxfm.act_jet(boost, tau, jet, cxc.cart3d)
        assert jnp.allclose(u.ustrip("km", out[0]["x"]), 4.0)  # x + dv*tau
        assert jnp.allclose(u.ustrip("km/s", out[1]["x"]), 1.5)  # v + dv
        assert jnp.allclose(u.ustrip("km/s2", out[2]["x"]), 0.1)  # a unchanged


# ============================================================================
# Jet slots above the named tangent semantic-kind ladder


class TestJetSlotsAboveLadder:
    """The additive routes prolong to any jet order, like the generic engine.

    The tangent semantic-kind ladder is only *named* up to `Acceleration`
    (order 2), but a jet has a slot at every order. The additive fast paths
    must not inherit that naming limit: the generic engine names no kinds and
    prolongs to any order, so they would otherwise have less capability than
    the very engine they exist to shortcut.
    """

    JET: ClassVar = {
        0: q3(1.0, 1.0, 1.0, "km"),
        1: q3(1.0, 1.0, 1.0, "km/s"),
        2: q3(1.0, 1.0, 1.0, "km/s2"),
        3: q3(1.0, 1.0, 1.0, "km/s3"),
        4: q3(1.0, 1.0, 1.0, "km/s4"),
    }

    @pytest.mark.parametrize("top", [3, 4])
    def test_translate_matches_generic(self, top):
        """A k=0 offset: the slot-wise route equals the generic prolongation."""
        op = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
        jet = {m: self.JET[m] for m in range(top + 1)}
        fast = cxfm.act_jet(op, None, jet, cxc.cart3d)
        generic = prolong_jet(op, None, jet, cxc.cart3d)
        for m in jet:
            assert allclose_cdict(fast[m], generic[m], self.JET[m]["x"].unit)

    @pytest.mark.parametrize("top", [3, 4])
    def test_static_fibre_kick_leaves_upper_slots(self, top):
        """A static offset only touches its own rung; upper slots pass through."""
        kick = cxfm.Translate(
            q3(100.0, 0.0, 0.0, "km/s"), chart=cxc.cart3d, semantic_kind=cxr.vel
        )
        jet = {m: self.JET[m] for m in range(top + 1)}
        out = cxfm.act_jet(kick, None, jet, cxc.cart3d)
        assert jnp.allclose(u.ustrip("km/s", out[1]["x"]), 101.0)
        for m in range(2, top + 1):
            assert allclose_cdict(out[m], jet[m], self.JET[m]["x"].unit)

    def test_timedep_fibre_kick_ladder_to_order_4(self):
        r"""A `TimeDep` fibre kick keeps climbing: slot $m$ gains $d^{m-1}\delta$.

        With $\delta(\tau) = A\tau^3$ ($A = 1$ km/s4) at $\tau = 2$ s the
        contributions are $\delta = 8$, $\dot\delta = 12$, $\ddot\delta = 12$
        and $\dddot\delta = 6$ -- the last two live on slots the ladder has no
        name for.
        """
        kick = cxfm.TimeDep.from_(
            lambda t: cxfm.Translate(
                {
                    "x": u.Q(1.0, "km/s4") * t**3,
                    "y": u.Q(0.0, "km/s"),
                    "z": u.Q(0.0, "km/s"),
                },
                chart=cxc.cart3d,
                semantic_kind=cxr.vel,
            )
        )
        out = cxfm.act_jet(kick, u.Q(2.0, "s"), self.JET, cxc.cart3d)
        assert jnp.allclose(u.ustrip("km", out[0]["x"]), 1.0)
        for m, gain in enumerate([8.0, 12.0, 12.0, 6.0], start=1):
            unit = self.JET[m]["x"].unit
            assert out[m]["x"].unit == unit
            assert jnp.allclose(u.ustrip(unit, out[m]["x"]), 1.0 + gain)
            # untouched components stay untouched
            assert jnp.allclose(u.ustrip(unit, out[m]["y"]), 1.0)


# ============================================================================
# Unit preservation and tau=None semantics (PR review)


class TestUnitPreservation:
    """Outputs preserve the data's own units; tau=None is passed through."""

    def test_pushforward_preserves_time_units(self):
        """A kpc/Myr velocity pushes forward to kpc/Myr, not kpc/s."""
        op = cxfm.Scale.from_factors([2.0, 2.0, 2.0])
        v = q3(1.0, 0.0, 0.0, "kpc/Myr")
        at = q3(1.0, 0.0, 0.0, "kpc")
        out = cxfm.pushforward(op, None, v, cxc.cart3d, cxr.coord_vel, at=at)
        assert out["x"].unit == u.unit("kpc/Myr")
        assert jnp.allclose(out["x"].value, 2.0)

    def test_prolong_preserves_time_units(self):
        """Jet slots come back in the data's own time base."""
        op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        jet = {0: q3(1.0, 0.0, 0.0, "kpc"), 1: q3(1.0, 0.0, 0.0, "kpc/Myr")}
        out = cxfm.act_jet(op, None, jet, cxc.cart3d)
        assert out[1]["y"].unit == u.unit("kpc/Myr")
        assert jnp.allclose(out[1]["y"].value, 1.0)

    def test_prolong_chain_consistent_across_tau_units(self):
        """Chain rule stays unit-consistent across differing tau units.

        The result must not depend on whether tau is given in seconds or
        Myr when the data's time base is Myr.
        """
        g = u.Q(1.0, "kpc/Myr2")
        moving = cxfm.TimeDep.from_(
            lambda t: cxfm.Translate(
                {"x": 0.5 * g * t**2, "y": u.Q(0.0, "kpc"), "z": u.Q(0.0, "kpc")},
                chart=cxc.cart3d,
            )
        )
        jet = {0: {k: u.Q(0.0, "kpc") for k in "xyz"}, 1: q3(1.0, 0.0, 0.0, "kpc/Myr")}
        out_myr = cxfm.act_jet(moving, u.Q(2.0, "Myr"), jet, cxc.cart3d)
        out_s = cxfm.act_jet(moving, u.Q(2.0, "Myr").uconvert("s"), jet, cxc.cart3d)
        # v' = v + g*tau = 1 + 2 = 3 kpc/Myr, regardless of tau's unit
        assert jnp.allclose(u.ustrip("kpc/Myr", out_myr[1]["x"]), 3.0)
        assert jnp.allclose(u.ustrip("kpc/Myr", out_s[1]["x"]), 3.0, atol=1e-6)

    def test_prolong_tau_none_passes_through_to_point_action(self):
        """tau=None is not replaced by a dummy time.

        A point action that genuinely requires tau (Boost) raises its own
        informative error even on the generic autodiff path.
        """
        boost = cxfm.Boost(q3(1.0, 0.0, 0.0, "km/s"), chart=cxc.cart3d)
        jet = {0: q3(1.0, 0.0, 0.0, "km"), 1: q3(0.0, 0.0, 0.0, "km/s")}
        generic = cxfm.act_jet.invoke(
            cxfm.AbstractTransform, object, dict, cxc.AbstractChart
        )
        with pytest.raises(TypeError, match="requires a time parameter"):
            generic(boost, None, jet, cxc.cart3d)


# ============================================================================
# Non-Cartesian operator charts (PR review): fast paths must defer to the
# generic engine when delta lives in a chart where the point action is
# base-point dependent.


class TestNonCartesianOpChart:
    """k=0 Translate with delta in a non-Cartesian chart."""

    @staticmethod
    def _td_op():
        def delta(t):
            s = t.ustrip("s")
            return {
                "r": u.Q(0.1, "km/s") * t,
                "theta": u.Q(0.0, "rad"),
                "phi": u.Q(0.02 * s, "rad"),
            }

        return cxfm.TimeDep.from_(lambda t: cxfm.Translate(delta(t), chart=cxc.sph3d))

    def test_td_velocity_matches_generic(self):
        """Act on velocity equals the generic prolongation of the point action."""
        op = self._td_op()
        tau = u.Q(2.0, "s")
        usys = u.unitsystems.si
        fast = cxfm.act(op, tau, SPH_V, cxc.sph3d, cxr.coord_vel, at=SPH_AT, usys=usys)
        gen = prolong_jet(op, tau, {0: SPH_AT, 1: SPH_V}, cxc.sph3d, usys=usys)
        for k in fast:
            unit = u.unit_of(gen[1][k])
            assert jnp.allclose(u.ustrip(unit, fast[k]), gen[1][k].value, rtol=1e-6)

    def test_static_velocity_not_identity(self):
        """A static spherical-chart delta is not identity on velocities.

        Its pushforward is base-point dependent, so velocities must NOT
        pass through unchanged.
        """
        op = cxfm.Translate(
            {"r": u.Q(0.2, "km"), "theta": u.Q(0.0, "rad"), "phi": u.Q(0.04, "rad")},
            chart=cxc.sph3d,
        )
        usys = u.unitsystems.si
        out = cxfm.act(op, None, SPH_V, cxc.sph3d, cxr.coord_vel, at=SPH_AT, usys=usys)
        # the phi-offset rotates the frame axes at the point: r-vel changes
        assert not jnp.allclose(u.ustrip("km/s", out["r"]), 0.3)

    def test_td_velocity_requires_at(self):
        """The generic fallback demands the base point."""
        op = self._td_op()
        with pytest.raises(TypeError, match="requires the base point"):
            cxfm.act(
                op,
                u.Q(2.0, "s"),
                SPH_V,
                cxc.sph3d,
                cxr.coord_vel,
                usys=u.unitsystems.si,
            )

    def test_cartesian_ladder_unaffected(self):
        """A Cartesian-chart TD delta still adds its rate to velocities.

        Time dependence now lives in `TimeDep`, so this goes through the
        generic tangent funnel (which needs the base point); the physics
        oracle is unchanged: v + ddelta/dtau = 1 + 3 = 4 km/s.
        """
        op = uniform_translate(3.0)
        v = {"x": u.Q(1.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
        at = q3(0.0, 0.0, 0.0, "km")
        out = cxfm.act(op, u.Q(2.0, "s"), v, cxc.cart3d, cxr.coord_vel, at=at)
        assert jnp.allclose(u.ustrip("km/s", out["x"]), 4.0)

    def test_is_flat_chart_no_global_cartesian(self):
        """Charts with no global Cartesian chart are non-flat, not an error."""
        assert is_flat_chart(cxc.cart3d)
        assert not is_flat_chart(cxc.sph3d)
        assert not is_flat_chart(cxc.PoincarePolar6D())

    def test_flat_delta_nonflat_data_chart_matches_generic(self):
        """Fast path equals generic when the data's chart is non-flat.

        A Cartesian delta acting on spherical-chart tangent data is
        nonlinear in the data's coordinates.
        """
        usys = u.unitsystems.si
        v = {
            "r": u.Q(0.3, "km/s"),
            "theta": u.Q(0.0, "rad/s"),
            "phi": u.Q(0.0, "rad/s"),
        }
        # static: NOT identity in spherical components
        op = cxfm.Translate.from_([100.0, 0.0, 0.0], "km")
        out = cxfm.act(op, None, v, cxc.sph3d, cxr.coord_vel, at=SPH_AT, usys=usys)
        assert not jnp.allclose(u.ustrip("km/s", out["r"]), 0.3)
        # TD: computes (previously raised ValueError) and equals generic
        op_td = uniform_translate(3.0)
        tau = u.Q(2.0, "s")
        fast = cxfm.act(op_td, tau, v, cxc.sph3d, cxr.coord_vel, at=SPH_AT, usys=usys)
        gen = prolong_jet(op_td, tau, {0: SPH_AT, 1: v}, cxc.sph3d, usys=usys)
        for k in fast:
            unit = u.unit_of(gen[1][k])
            assert jnp.allclose(u.ustrip(unit, fast[k]), gen[1][k].value, rtol=1e-6)

    def test_order2_unit_preservation(self):
        """Acceleration units survive the exact rational time-unit root."""
        op = cxfm.Scale.from_factors([2.0, 2.0, 2.0])
        a = q3(1.0, 0.0, 0.0, "kpc/Myr2")
        at = q3(1.0, 0.0, 0.0, "kpc")
        out = cxfm.pushforward(op, None, a, cxc.cart3d, cxr.coord_acc, at=at)
        assert out["x"].unit == u.unit("kpc/Myr2")
        assert jnp.allclose(out["x"].value, 2.0)

    def test_is_time_dependent_non_transform_raises(self):
        """Non-`AbstractTransform` inputs get a clear, informative `TypeError`."""
        with pytest.raises(TypeError, match="expects an AbstractTransform"):
            cxfm.is_time_dependent(lambda t: t)

    def test_prolong_jet_mismatched_slot_keys_raises(self):
        """Jet slots with different components than slot 0 raise informatively."""
        op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        jet = {
            0: q3(1.0, 0.0, 0.0, "km"),
            1: {"x": u.Q(0.0, "km/s"), "y": u.Q(0.0, "km/s")},  # missing z
        }
        with pytest.raises(TypeError, match=r"slot 1 .*missing \['z'\]"):
            cxfm.act_jet(op, None, jet, cxc.cart3d)

    def test_boost_nonflat_chart_acceleration_not_identity(self):
        """Static Boost on spherical-chart accelerations is not identity.

        It defers to the generic engine (previously silently identity) and
        forwards the anchors.
        """
        usys = u.unitsystems.si
        dv = q3(1.0, 0.0, 0.0, "km/s")
        boost = cxfm.Boost(dv, chart=cxc.cart3d)
        a = {
            "r": u.Q(0.0, "km/s2"),
            "theta": u.Q(0.0, "rad/s2"),
            "phi": u.Q(0.0, "rad/s2"),
        }
        v = {
            "r": u.Q(0.3, "km/s"),
            "theta": u.Q(0.0, "rad/s"),
            "phi": u.Q(0.0, "rad/s"),
        }
        tau = u.Q(2.0, "s")
        with pytest.raises(TypeError, match="requires the base point"):
            cxfm.act(boost, tau, a, cxc.sph3d, cxr.coord_acc, usys=usys)
        fast = cxfm.act(
            boost, tau, a, cxc.sph3d, cxr.coord_acc, at_jet={0: SPH_AT, 1: v}, usys=usys
        )
        td = cxfm.TimeDep(cxfm.builders.UniformTranslation(dv, chart=cxc.cart3d))
        gen = prolong_jet(td, tau, {0: SPH_AT, 1: v, 2: a}, cxc.sph3d, usys=usys)
        for k in fast:
            unit = u.unit_of(gen[2][k])
            assert jnp.allclose(
                u.ustrip(unit, fast[k]), gen[2][k].value, rtol=1e-5, atol=1e-9
            )
        # the true result is nonzero: it was previously silently identity
        assert not jnp.allclose(u.ustrip("km/s2", fast["r"]), 0.0)


# ============================================================================
# Final-audit regressions: fibre kicks, bundles, linear ops under new verbs


class TestFibreKickCrossChart:
    """A fibre kick is a tangent vector: cross-chart action via the Jacobian."""

    kick = None  # built in tests to avoid import-time work

    def test_kick_on_spherical_velocity_matches_tangent_map(self):
        """Cartesian vel-kick on spherical velocity == Jacobian-mapped add."""
        usys = u.unitsystems.si
        dv = q3(1.0, 0.0, 0.0, "km/s")
        kick = cxfm.Translate(dv, chart=cxc.cart3d, semantic_kind=cxr.vel)
        at = {"r": u.Q(5.0, "km"), "theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
        v = {
            "r": u.Q(0.3, "km/s"),
            "theta": u.Q(0.01, "rad/s"),
            "phi": u.Q(0.02, "rad/s"),
        }
        out = cxfm.act(kick, None, v, cxc.sph3d, cxr.coord_vel, at=at, usys=usys)
        # reference: map delta into the spherical chart at the point, add
        at_cart = cxc.pt_map(at, cxc.sph3d, cxc.cart3d, usys=usys)
        vel_rep = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.vel)
        dv_sph = cxr.tangent_map(
            dv, cxc.cart3d, vel_rep, cxc.sph3d, at=at_cart, usys=usys
        )
        for k, vk in v.items():
            unit = u.unit_of(dv_sph[k])
            expect = u.ustrip(unit, vk) + dv_sph[k].value
            assert jnp.allclose(u.ustrip(unit, out[k]), expect, rtol=1e-6)

    def test_kick_cross_chart_requires_at(self):
        """Without the base point the cross-chart kick raises informatively."""
        dv = q3(1.0, 0.0, 0.0, "km/s")
        kick = cxfm.Translate(dv, chart=cxc.cart3d, semantic_kind=cxr.vel)
        v = {
            "r": u.Q(0.3, "km/s"),
            "theta": u.Q(0.01, "rad/s"),
            "phi": u.Q(0.02, "rad/s"),
        }
        with pytest.raises(TypeError, match="requires the base point"):
            cxfm.act(kick, None, v, cxc.sph3d, cxr.coord_vel, usys=u.unitsystems.si)

    def test_kick_rejects_bare_arrays(self):
        """Unitless arrays are ambiguous under a kick: rejected, not no-op'd."""
        dv = q3(1.0, 0.0, 0.0, "km/s")
        kick = cxfm.Translate(dv, chart=cxc.cart3d, semantic_kind=cxr.vel)
        arr = jnp.asarray([1.0, 0.0, 0.0])
        with pytest.raises(TypeError, match="ambiguous"):
            cxfm.act(kick, None, arr, cxc.cart3d, cxr.point, usys=u.unitsystems.si)


class TestCoordinateBundleEdges:
    """Bundle-layer seams from the final audit."""

    @staticmethod
    def _bundle(**fields):
        pt = cxv.Point.from_([1.0, 2.0, 3.0], "km")
        return cxv.Coordinate(pt, **fields)

    def test_td_bundle_rejects_anchor_overrides(self):
        """TD Coordinate act raises on at= instead of silently ignoring it."""
        moving = uniform_translate(3.0)
        vel = cxv.Tangent(
            q3(0.1, 0.0, 0.0, "km/s"), cxc.cart3d, cxr.coord_basis, cxr.vel
        )
        coord = self._bundle(vel=vel)
        with pytest.raises(TypeError, match="does not accept keyword overrides"):
            cx.act(moving, u.Q(1.0, "s"), coord, at=q3(9.0, 0.0, 0.0, "km"))

    def test_static_boost_bundle_with_nonflat_fibre(self):
        """Static Boost on a bundle with a cylindrical fibre works.

        Boost's point action is intrinsically tau-dependent, so the bundle
        takes the joint-jet path even with a constant delta-v.
        """
        boost = cxfm.Boost(q3(1.0, 0.0, 0.0, "km/s"), chart=cxc.cart3d)
        vel = cxv.Tangent(
            q3(0.1, 0.0, 0.0, "km/s"), cxc.cart3d, cxr.coord_basis, cxr.vel
        )
        acc = cxv.Tangent(
            {
                "rho": u.Q(0.1, "km/s2"),
                "phi": u.Q(0.0, "rad/s2"),
                "z": u.Q(0.0, "km/s2"),
            },
            cxc.cyl3d,
            cxr.coord_basis,
            cxr.acc,
        )
        coord = self._bundle(vel=vel, acc=acc)
        out = cx.act(boost, u.Q(1.0, "s"), coord, usys=u.unitsystems.si)
        assert jnp.allclose(u.ustrip("km", out.point.data["x"]), 2.0)  # x + dv*tau
        assert jnp.allclose(u.ustrip("km/s", out._data["vel"].data["x"]), 1.1)

    def test_td_bundle_duplicate_ladder_order_raises(self):
        """Two fibres at the same ladder order are ambiguous for the jet."""
        moving = uniform_translate(3.0)
        v1 = cxv.Tangent(
            q3(0.1, 0.0, 0.0, "km/s"), cxc.cart3d, cxr.coord_basis, cxr.vel
        )
        v2 = cxv.Tangent(
            q3(0.2, 0.0, 0.0, "km/s"), cxc.cart3d, cxr.coord_basis, cxr.vel
        )
        coord = self._bundle(vel=v1, vel2=v2)
        with pytest.raises(ValueError, match="multiple fibres at ladder order"):
            cx.act(moving, u.Q(1.0, "s"), coord)

    def test_td_bundle_cross_chart_fibre_matches_cartesian(self):
        """Cross-chart fibre round trip matches the Cartesian-fibre result.

        A cylindrical velocity fibre under a TD op equals the same physics
        computed with a Cartesian fibre.
        """
        moving = uniform_translate(3.0)
        tau = u.Q(2.0, "s")
        usys = u.unitsystems.si
        vel_cart = cxv.Tangent(
            q3(0.1, 0.2, 0.0, "km/s"), cxc.cart3d, cxr.coord_basis, cxr.vel
        )
        coord_cart = self._bundle(vel=vel_cart)
        out_cart = cx.act(moving, tau, coord_cart, usys=usys)

        vel_cyl = cxr.cconvert(vel_cart, cxc.cyl3d, at=coord_cart.point.data, usys=usys)
        coord_cyl = self._bundle(vel=vel_cyl)
        out_cyl = cx.act(moving, tau, coord_cyl, usys=usys)
        # convert the cylindrical output fibre back to cartesian at the new point
        back = cxr.cconvert(
            out_cyl._data["vel"],
            cxc.cart3d,
            at=cxr.cconvert(out_cyl.point, cxc.cyl3d).data,
            usys=usys,
        )
        for k in "xyz":
            assert jnp.allclose(
                u.ustrip("km/s", back.data[k]),
                u.ustrip("km/s", out_cart._data["vel"].data[k]),
                atol=1e-6,
            )

    def test_td_bundle_displacement_fibre_pushforward(self):
        """Displacement fibres in a TD bundle take the frozen-tau pushforward.

        They are invariant under a flat translation.
        """
        moving = uniform_translate(3.0)
        d = cxv.Tangent(q3(0.5, 0.0, 0.0, "km"), cxc.cart3d, cxr.coord_basis, cxr.dpl)
        coord = self._bundle(disp=d)
        out = cx.act(moving, u.Q(2.0, "s"), coord)
        assert jnp.allclose(u.ustrip("km", out._data["disp"].data["x"]), 0.5)


class TestLinearOpsUnderNewVerbs:
    """Shear/Reflect coverage via the generic engine (previously untested)."""

    @pytest.mark.parametrize(
        "op",
        [
            cxfm.Shear(
                jnp.asarray([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            ),
            cxfm.Reflect.from_normal([1.0, 0.0, 0.0]),
        ],
        ids=["shear", "reflect"],
    )
    def test_pushforward_matches_act_on_velocity(self, op):
        """Static linear ops: act on velocity == frozen-tau pushforward."""
        at = q3(1.0, -2.0, 0.5, "m")
        v = q3(0.3, 0.1, -0.2, "m/s")
        a1 = cxfm.act(op, None, v, cxc.cart3d, cxr.coord_vel, at=at)
        a2 = cxfm.pushforward(op, None, v, cxc.cart3d, cxr.coord_vel, at=at)
        assert allclose_cdict(a1, a2, "m/s", atol=1e-8)

    @pytest.mark.parametrize(
        "op",
        [
            cxfm.Shear(
                jnp.asarray([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            ),
            cxfm.Reflect.from_normal([1.0, 0.0, 0.0]),
        ],
        ids=["shear", "reflect"],
    )
    def test_prolong_jet_matches_per_slot(self, op):
        """Prolong on a 1-jet gives the same slots as point-act + vel-act."""
        at = q3(1.0, -2.0, 0.5, "m")
        v = q3(0.3, 0.1, -0.2, "m/s")
        jet = cxfm.act_jet(op, None, {0: at, 1: v}, cxc.cart3d)
        p_ref = cxfm.act(op, None, at, cxc.cart3d, cxr.point)
        v_ref = cxfm.act(op, None, v, cxc.cart3d, cxr.coord_vel, at=at)
        assert allclose_cdict(jet[0], p_ref, "m", atol=1e-8)
        assert allclose_cdict(jet[1], v_ref, "m/s", atol=1e-8)

    def test_kick_cross_chart_in_jet(self):
        """Prolong supplies at=jet[0], so cross-chart kicks work in jets."""
        usys = u.unitsystems.si
        dv = q3(1.0, 0.0, 0.0, "km/s")
        kick = cxfm.Translate(dv, chart=cxc.cart3d, semantic_kind=cxr.vel)
        jet = {
            0: {"r": u.Q(5.0, "km"), "theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")},
            1: {
                "r": u.Q(0.3, "km/s"),
                "theta": u.Q(0.01, "rad/s"),
                "phi": u.Q(0.02, "rad/s"),
            },
        }
        out = cxfm.act_jet(kick, None, jet, cxc.sph3d, usys=usys)
        ref = cxfm.act(
            kick, None, jet[1], cxc.sph3d, cxr.coord_vel, at=jet[0], usys=usys
        )
        for k, refk in ref.items():
            unit = u.unit_of(refk)
            assert jnp.allclose(u.ustrip(unit, out[1][k]), refk.value, rtol=1e-6)
        assert jnp.allclose(u.ustrip("km", out[0]["r"]), 5.0)  # point untouched

    def test_pushforward_mismatched_components_raises(self):
        """Tangent components must match the base point's components."""
        op = cxfm.Scale.from_factors([2.0, 2.0, 2.0])
        at = q3(1.0, 0.0, 0.0, "m")
        v_missing = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}  # no z
        with pytest.raises(TypeError, match=r"missing \['z'\]"):
            cxfm.pushforward(op, None, v_missing, cxc.cart3d, cxr.coord_vel, at=at)
        v_extra = q3(1.0, 0.0, 0.0, "m/s") | {"w": u.Q(0.0, "m/s")}
        with pytest.raises(TypeError, match=r"unexpected \['w'\]"):
            cxfm.pushforward(op, None, v_extra, cxc.cart3d, cxr.coord_vel, at=at)

    def test_rotate_raw_tau_unitful_data(self):
        """Raw (unitless) tau with unitful data works and is consistent.

        d/dtau is interpreted in the data's own time base, per the generic
        engine's raw-tau convention.
        """
        op = _rot_z_raw_op()
        at = q3(1.0, 0.0, 0.0, "m")
        v = q3(0.0, 0.0, 0.0, "m/s")
        tau = jnp.asarray(0.0)
        out = cxfm.act(op, tau, v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at)
        gen = prolong_jet(op, tau, {0: at, 1: v}, cxc.cart3d)
        for k in out:
            unit = u.unit_of(gen[1][k])
            assert jnp.allclose(u.ustrip(unit, out[k]), gen[1][k].value, atol=1e-7)
        # omega = 1 per data-time-base; z-hat x x-hat = y-hat
        assert jnp.allclose(u.ustrip("m/s", out["y"]), 1.0, atol=1e-7)

    def test_rotate_closed_form_fully_raw_data(self):
        """Fully unitless data stays raw through the m=1 prolongation.

        Mirrors the generic engine's None-unit "stay raw" policy.
        """
        op = _rot_z_raw_op()
        at = {"x": jnp.asarray(1.0), "y": jnp.asarray(0.0), "z": jnp.asarray(0.0)}
        v = {"x": jnp.asarray(0.0), "y": jnp.asarray(0.0), "z": jnp.asarray(0.0)}
        tau = jnp.asarray(0.0)
        out = cxfm.act(op, tau, v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at)
        gen = prolong_jet(op, tau, {0: at, 1: v}, cxc.cart3d)
        for k in out:
            assert not u.quantity.is_any_quantity(out[k])  # stays raw
            assert jnp.allclose(out[k], gen[1][k], atol=1e-7)
        assert jnp.allclose(out["y"], 1.0, atol=1e-7)


# ============================================================================
# Robustness: double inversion, integer dtypes


class TestRobustness:
    """Robustness fixes: double inversion and integer-dtype promotion."""

    def test_time_dependent_inverse_roundtrip(self):
        """inverse.inverse of a time-dependent op is usable (and unwraps)."""
        moving = uniform_translate(3.0)
        inv2 = moving.inverse.inverse  # the pointwise inverse is an involution
        tau = u.Q(2.0, "s")
        p = q3(1.0, 0.0, 0.0, "km")
        out = cxfm.act(inv2, tau, p, cxc.cart3d, cxr.point)
        expected = cxfm.act(moving, tau, p, cxc.cart3d, cxr.point)
        assert allclose_cdict(out, expected, "km")

    def test_integer_inputs_through_prolongation(self):
        """Integer-valued Quantities are promoted at the jvp boundary."""
        moving = uniform_translate(3.0)
        tau = u.Q(2, "s")  # int
        jet = {
            0: {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")},  # ints
            1: q3(0.0, 0.0, 0.0, "km/s"),
        }
        out = cxfm.act_jet(moving, tau, jet, cxc.cart3d)
        assert jnp.allclose(u.ustrip("km/s", out[1]["x"]), 3.0)

    def test_integer_inputs_through_pushforward(self):
        """Integer-valued anchors are promoted in pushforward."""
        op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
        v = {"x": u.Q(1, "m/s"), "y": u.Q(1, "m/s"), "z": u.Q(1, "m/s")}
        at = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
        out = cxfm.pushforward(op, None, v, cxc.cart3d, cxr.coord_vel, at=at)
        assert jnp.allclose(u.ustrip("m/s", out["y"]), 3.0)


# ============================================================================
# gh#936: an anchor slot `act` cannot use must never be swallowed


# `act` on point CDict data resolves to one of exactly four methods; each is
# a separate registration, so each needs its own guard. Keep this in step with
# the dispatch table -- a new transform that registers its own CDict `act`
# rule leaves the generic funnel and belongs here.
POINT_ACTION_PATHS = [
    pytest.param(
        cxfm.Rotate.from_euler("z", u.Q(37.0, "deg")), None, id="funnel"
    ),  # generic funnel
    pytest.param(
        cxfm.Translate.from_([1, 2, 3], "kpc"), None, id="translate"
    ),  # translate.py
    pytest.param(
        cxfm.Boost.from_([1.0, 0.0, 0.0], "kpc/Myr"), u.Q(1.0, "Myr"), id="boost"
    ),  # boost.py
    pytest.param(
        cxfm.Composed((cxfm.Translate.from_([1, 2, 3], "kpc"),)), None, id="composed"
    ),  # composed.py
]


class TestUnusableAnchorSlotsAreRefused:
    r"""`act` anchors on slot 0; a higher slot it cannot read must raise.

    `act` on a lone order-$m$ slot under a *static* transform is the
    frozen-$\tau$ pushforward $\partial_x\phi \cdot v$ — first order, slot 0
    only. That is the documented split from `act_jet` (the full prolongation),
    but it used to *accept* `at_jet={1: v}` and discard it, so a caller who
    assembled the jet correctly got a silently first-order answer: for
    acceleration in a curvilinear chart the term $\partial_{xx}\phi(v,v)$ is
    simply missing.
    """

    ROT: ClassVar = cxfm.Rotate.from_euler(
        "z", u.Q(37.0, "deg")
    ) | cxfm.Rotate.from_euler("x", u.Q(20.0, "deg"))
    Q0: ClassVar = {
        "r": u.Q(2.0, "kpc"),
        "theta": u.Q(50.0, "deg"),
        "phi": u.Q(25.0, "deg"),
    }
    V0: ClassVar = {
        "r": u.Q(1.0, "kpc/Myr"),
        "theta": u.Q(0.3, "rad/Myr"),
        "phi": u.Q(0.7, "rad/Myr"),
    }
    A0: ClassVar = {
        "r": u.Q(0.1, "kpc/Myr2"),
        "theta": u.Q(0.05, "rad/Myr2"),
        "phi": u.Q(-0.02, "rad/Myr2"),
    }

    def test_a_velocity_slot_on_an_acceleration_act_raises(self):
        """The reported case: a supplied slot 1 was ignored, not used."""
        with pytest.raises(TypeError, match=r"act_jet"):
            cxfm.act(
                self.ROT,
                None,
                self.A0,
                cxc.sph3d,
                cxr.coord_acc,
                at=self.Q0,
                at_jet={1: self.V0},
            )

    def test_a_nonsense_slot_raises_too(self):
        """A slot no order could ever read is the same defect, louder."""
        op = cxfm.Rotate.from_euler("z", u.Q(37.0, "deg"))
        with pytest.raises(TypeError, match=r"act_jet"):
            cxfm.act(
                op,
                None,
                self.A0,
                cxc.sph3d,
                cxr.coord_acc,
                at=self.Q0,
                at_jet={7: self.V0},
            )

    def test_act_jet_is_the_documented_way_and_differs_materially(self):
        """`act_jet` uses every slot; the pushforward answer is not close.

        The measured gap is what made the silent discard worth an error
        rather than a docs note: ~82% on theta, ~99.9% on phi.
        """
        pushed = cxfm.act(self.ROT, None, self.A0, cxc.sph3d, cxr.coord_acc, at=self.Q0)
        full = cxfm.act_jet(
            self.ROT, None, {0: self.Q0, 1: self.V0, 2: self.A0}, cxc.sph3d
        )[2]
        rel = {}
        for k in ("theta", "phi"):
            ref = float(u.ustrip("rad/Myr2", full[k]))
            # The gap is stated relative to `full`, so a `full` near zero would
            # make the ratio meaningless (0/0 -> nan, which fails the assert
            # below for the wrong reason). Fail on the premise instead.
            assert abs(ref) > 1e-12, f"{k}: reference acceleration is ~0"
            rel[k] = abs(float(u.ustrip("rad/Myr2", pushed[k] - full[k])) / ref)
        assert rel["theta"] > 0.5
        assert rel["phi"] > 0.5

    def test_slot_zero_alone_still_works(self):
        """Only slots >= 1 are refused; `at_jet={0: q}` is `at=q`."""
        via_at = cxfm.act(self.ROT, None, self.A0, cxc.sph3d, cxr.coord_acc, at=self.Q0)
        via_jet = cxfm.act(
            self.ROT, None, self.A0, cxc.sph3d, cxr.coord_acc, at_jet={0: self.Q0}
        )
        for k in via_at:
            unit = u.unit_of(via_at[k])
            assert jnp.allclose(u.ustrip(unit, via_at[k]), u.ustrip(unit, via_jet[k]))

    def test_flat_cartesian_acceleration_is_unchanged(self):
        """A linear op in a flat chart needs no anchor at all, and still does.

        This is the behaviour the numerically-complete fix would have broken:
        routing order >= 2 through `prolong_slot` turns these exact,
        anchor-free calls into missing-slot errors.
        """
        op = cxfm.Rotate.from_euler("z", u.Q(90.0, "deg"))
        a = q3(1.0, 2.0, 3.0, "m/s2")
        out = cxfm.act(op, None, a, cxc.cart3d, cxr.coord_acc)
        full = cxfm.act_jet(
            op,
            None,
            {0: q3(1.0, 0.0, 0.0, "m"), 1: q3(0.5, -0.5, 0.0, "m/s"), 2: a},
            cxc.cart3d,
        )[2]
        assert allclose_cdict(out, full, "m/s2")

    def test_act_jet_itself_takes_every_slot_untouched(self):
        """`act_jet` is not narrowed by the guard: it reads slots 0..m."""
        out = cxfm.act_jet(
            self.ROT, None, {0: self.Q0, 1: self.V0, 2: self.A0}, cxc.sph3d
        )
        assert set(out) == {0, 1, 2}
        scaled = cxfm.act_jet(
            self.ROT,
            None,
            {0: self.Q0, 1: {k: 100 * v for k, v in self.V0.items()}, 2: self.A0},
            cxc.sph3d,
        )
        # slot 1 genuinely feeds slot 2 there -- the discriminator that the
        # jet path is not itself a disguised pushforward.
        assert not jnp.allclose(
            u.ustrip("rad/Myr2", out[2]["phi"]), u.ustrip("rad/Myr2", scaled[2]["phi"])
        )

    def test_the_fibre_offset_ladder_still_accepts_a_redundant_slot(self):
        r"""The ladder is exact, so a spare slot there is redundant, not lossy.

        Deliberately *not* guarded: `act` on a fibre offset applies
        $d^{m-k}\delta/d\tau^{m-k}$, which has no $x$-dependence to curve.
        """
        kick = cxfm.TimeDep.from_(
            lambda t: cxfm.Translate(
                {
                    "x": u.Q(3.0, "km/s3") * t,
                    "y": u.Q(0.0, "km/s2"),
                    "z": u.Q(0.0, "km/s2"),
                },
                chart=cxc.cart3d,
                semantic_kind=cxr.acc,
            )
        )
        a = q3(1.0, 1.0, 1.0, "km/s2")
        out = cxfm.act(
            kick,
            u.Q(2.0, "s"),
            a,
            cxc.cart3d,
            cxr.coord_acc,
            at_jet={0: q3(0.0, 0.0, 0.0, "km"), 1: q3(1.0, 2.0, 3.0, "km/s")},
        )
        assert jnp.allclose(u.ustrip("km/s2", out["x"]), 7.0)

    @pytest.mark.parametrize(("op", "tau"), POINT_ACTION_PATHS)
    def test_a_point_action_refuses_the_slot_in_point_language(self, op, tau):
        """Every point action refuses, and not in the tangent language.

        Point geometry has no ladder order (`None`, not 0), so the rejection
        needs its own wording: this pins the point message rather than a
        message claiming "order-None tangent data".

        The parameters are the *dispatch* map, not a sample of operators.
        `act` on point CDict data resolves to one of exactly four methods --
        the generic geometry funnel, `Translate`, `Boost` and `Composed` --
        and a guard on one of them says nothing about the other three. That
        is how gh#936 survived in the first place: `Composed` checked, and
        every primitive beside it swallowed the slot.
        """
        with pytest.raises(TypeError, match=r"on point data reads jet slot 0 alone"):
            cxfm.act(op, tau, self.Q0, cxc.sph3d, cxr.point, at_jet={1: self.V0})

    @pytest.mark.parametrize(("op", "tau"), POINT_ACTION_PATHS)
    def test_a_point_action_still_takes_slot_zero(self, op, tau):
        """Only slots >= 1 are refused on the point path as well."""
        q = q3(1.0, 2.0, 3.0, "kpc")
        via_plain = cxfm.act(op, tau, q, cxc.cart3d, cxr.point)
        via_jet = cxfm.act(op, tau, q, cxc.cart3d, cxr.point, at_jet={0: q})
        assert allclose_cdict(via_plain, via_jet, "kpc")

    def test_the_guard_covers_the_whole_point_dispatch_table(self):
        """The guard must cover the dispatch table, not a sample of it.

        This is the shape of gh#936's recurrence: `Composed` checked and
        every primitive beside it did not, because a transform that
        registers its own CDict `act` rule leaves the generic funnel --
        and the guard with it -- without anything saying so. So pin the set
        of methods `act` resolves to for point data. Growing that set is
        exactly the moment to add a guard, and this fails *then*, rather
        than on a quietly first-order answer much later.

        `identity` is the one deliberate omission: its rule is a total
        catch-all over every input shape (arrays, `Point`s, CDicts, with or
        without a chart), so it has no `rep` to tell a point call from a
        tangent one -- and it returns its input, so no slot it ignores can
        make the answer wrong.
        """
        act = cxfm.act
        act._resolve_pending_registrations()

        def subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from subclasses(sub)

        modules = set()
        for cls in subclasses(cxfm.AbstractTransform):
            try:
                # Only the *type* reaches dispatch, so an uninitialized
                # instance resolves the same method a real one would -- and
                # spares this a constructor call per transform.
                impl, _ = act.resolve_method(
                    (object.__new__(cls), None, self.Q0, cxc.sph3d, cxr.point)
                )
            except TypeError:  # abstract class
                continue
            modules.add(impl.__module__.rsplit(".", maxsplit=1)[-1])

        assert modules == {"prolong", "translate", "boost", "composed", "identity"}

    def test_the_error_names_the_operator_the_caller_wrote(self):
        """`Boost` delegates its point action; the message must not leak that.

        `Boost` reaches the ladder through the equivalent
        ``TimeDep(Translate)``, which would refuse the slot under *its* name.
        The caller wrote `Boost`.
        """
        op = cxfm.Boost.from_([1.0, 0.0, 0.0], "kpc/Myr")
        with pytest.raises(TypeError, match=r"^act\(Boost, "):
            cxfm.act(
                op, u.Q(1.0, "Myr"), self.Q0, cxc.sph3d, cxr.point, at_jet={1: self.V0}
            )


# ============================================================================
# gh#936: a fibre kick couples the slots above its rung through the chart map


class TestFibreKickAboveItsRung:
    r"""A velocity kick does not leave the *coordinate* acceleration alone.

    The ladder rule says order-$m$ data gains $d^{m-k}\delta/d\tau^{m-k}$, so
    a constant velocity kick ($k=1$) leaves order 2 untouched. That is exact
    in the chart the offset's components are constant in, and wrong in any
    other: pushing the kick through a chart map $\psi$ gives

    $$\ddot q' = \ddot q + 2 D^2\psi(\dot x, \Delta v) + D^2\psi(\Delta v, \Delta v)$$

    and both new terms were missing. The reference throughout is that a chart
    change and the kick must commute -- an identity the code cannot satisfy by
    accident, and which owes nothing to either implementation.
    """

    CH: ClassVar = cxc.sph3d
    Q0: ClassVar = {
        "r": u.Q(2.0, "kpc"),
        "theta": u.Q(0.9, "rad"),
        "phi": u.Q(0.4, "rad"),
    }
    V0: ClassVar = {
        "r": u.Q(1.0, "kpc/Myr"),
        "theta": u.Q(0.3, "rad/Myr"),
        "phi": u.Q(0.7, "rad/Myr"),
    }
    A0: ClassVar = {
        "r": u.Q(0.1, "kpc/Myr2"),
        "theta": u.Q(0.05, "rad/Myr2"),
        "phi": u.Q(-0.02, "rad/Myr2"),
    }
    KICK: ClassVar = cxfm.Translate(
        {
            "x": u.Q(0.2, "kpc/Myr"),
            "y": u.Q(-0.1, "kpc/Myr"),
            "z": u.Q(0.05, "kpc/Myr"),
        },
        chart=cxc.cart3d,
        semantic_kind=cxr.vel,
    )

    def _jet(self):
        return {0: self.Q0, 1: self.V0, 2: self.A0}

    @staticmethod
    def _to_cart(data):
        return cxc.pt_map(data, cxc.sph3d, cxc.cart3d)

    def _both_routes(self, op, tau=None):
        """(act_jet in sph then convert, convert then act_jet in cart)."""
        from coordinax.transforms._src.actions.prolong import prolong_point_map

        here = prolong_point_map(
            self._to_cart, cxfm.act_jet(op, tau, self._jet(), self.CH)
        )
        there = cxfm.act_jet(
            op, tau, prolong_point_map(self._to_cart, self._jet()), cxc.cart3d
        )
        return here, there

    def test_the_kick_commutes_with_the_chart_change(self):
        """Was 29.9% adrift on the acceleration slot; velocity was always fine."""
        here, there = self._both_routes(self.KICK)
        for slot, unit in ((1, "kpc/Myr"), (2, "kpc/Myr2")):
            for k in ("x", "y", "z"):
                assert jnp.allclose(
                    u.ustrip(unit, here[slot][k]), u.ustrip(unit, there[slot][k])
                )

    def test_a_time_dependent_kick_commutes_too(self):
        """The `TimeDep` ladder took the same shortcut, so it needs the same check."""
        td = cxfm.TimeDep.from_(
            lambda t: cxfm.Translate(
                {
                    "x": u.Q(0.2, "kpc/Myr2") * t,
                    "y": u.Q(-0.1, "kpc/Myr2") * t,
                    "z": u.Q(0.05, "kpc/Myr2") * t,
                },
                chart=cxc.cart3d,
                semantic_kind=cxr.vel,
            )
        )
        here, there = self._both_routes(td, tau=u.Q(2.0, "Myr"))
        for k in ("x", "y", "z"):
            assert jnp.allclose(
                u.ustrip("kpc/Myr2", here[2][k]), u.ustrip("kpc/Myr2", there[2][k])
            )

    def test_the_acceleration_actually_moves(self):
        """Guard the guard: the old answer returned the input unchanged."""
        out = cxfm.act_jet(self.KICK, None, self._jet(), self.CH)[2]
        assert not jnp.allclose(
            u.ustrip("rad/Myr2", out["theta"]), u.ustrip("rad/Myr2", self.A0["theta"])
        )

    def test_in_the_offsets_own_chart_nothing_changes(self):
        """There the offset IS a constant field, so the cheap ladder is exact."""
        q = q3(1.0, 2.0, 3.0, "kpc")
        v = q3(0.3, -0.4, 0.2, "kpc/Myr")
        a = q3(0.1, 0.2, 0.3, "kpc/Myr2")
        out = cxfm.act_jet(self.KICK, None, {0: q, 1: v, 2: a}, cxc.cart3d)
        assert allclose_cdict(out[2], a, "kpc/Myr2")
        assert jnp.allclose(u.ustrip("kpc/Myr", out[1]["x"]), 0.3 + 0.2)

    def test_a_lone_acceleration_slot_needs_the_velocity_and_says_so(self):
        """It used to return the input, 350% adrift, and discard a given slot."""
        with pytest.raises(TypeError, match=r"requires jet slots"):
            cxfm.act(self.KICK, None, self.A0, self.CH, cxr.coord_acc, at=self.Q0)

    def test_a_lone_acceleration_slot_is_exact_once_given_the_velocity(self):
        """With `at_jet={1: v}` it matches the full jet exactly."""
        got = cxfm.act(
            self.KICK,
            None,
            self.A0,
            self.CH,
            cxr.coord_acc,
            at=self.Q0,
            at_jet={1: self.V0},
        )
        ref = cxfm.act_jet(self.KICK, None, self._jet(), self.CH)[2]
        for k in ref:
            unit = u.unit_of(ref[k])
            assert jnp.allclose(u.ustrip(unit, got[k]), u.ustrip(unit, ref[k]))


class TestProlongPointMapValidatesItsJet:
    """The point-map engine must refuse a bad jet as clearly as `prolong_jet`.

    Both current callers validate before calling, so this is the guarantee for
    anyone reaching the engine directly: a hole used to surface as a bare
    `KeyError` from the first missing index rather than saying what was wrong.
    """

    Q0: ClassVar = {"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")}
    A0: ClassVar = {
        "x": u.Q(0.1, "kpc/Myr2"),
        "y": u.Q(0.2, "kpc/Myr2"),
        "z": u.Q(0.3, "kpc/Myr2"),
    }

    @staticmethod
    def _psi(data):
        return cxc.pt_map(data, cxc.cart3d, cxc.sph3d)

    def test_a_hole_raises_rather_than_keyerror(self):
        from coordinax.transforms._src.actions.prolong import prolong_point_map

        with pytest.raises(TypeError, match=r"requires all jet slots 1\.\.2"):
            prolong_point_map(self._psi, {0: self.Q0, 2: self.A0})

    def test_a_missing_base_point_raises(self):
        from coordinax.transforms._src.actions.prolong import prolong_point_map

        with pytest.raises(TypeError, match=r"base point at jet slot 0"):
            prolong_point_map(self._psi, {1: self.A0})

    def test_mismatched_components_raise(self):
        from coordinax.transforms._src.actions.prolong import prolong_point_map

        with pytest.raises(TypeError, match=r"do not match slot 0"):
            prolong_point_map(self._psi, {0: self.Q0, 1: {"x": u.Q(1.0, "kpc/Myr")}})

    def test_the_message_names_the_point_map_not_act_jet(self):
        """It is not `act_jet`, and saying so sends the reader to the wrong verb."""
        from coordinax.transforms._src.actions.prolong import prolong_point_map

        with pytest.raises(TypeError, match=r"^prolong_point_map"):
            prolong_point_map(self._psi, {0: self.Q0, 2: self.A0})


def test_prolong_point_map_on_a_base_point_alone():
    """A jet of just slot 0 is the plain point map, with no chain to build."""
    from coordinax.transforms._src.actions.prolong import prolong_point_map

    q = {"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")}
    out = prolong_point_map(lambda d: cxc.pt_map(d, cxc.cart3d, cxc.sph3d), {0: q})
    assert set(out) == {0}
    direct = cxc.pt_map(q, cxc.cart3d, cxc.sph3d)
    for k in direct:
        unit = u.unit_of(direct[k])
        assert jnp.allclose(u.ustrip(unit, out[0][k]), u.ustrip(unit, direct[k]))


def test_a_time_dependent_kick_on_a_lone_slot_uses_the_supplied_jet():
    """The `TimeDep` twin of the static lone-slot path.

    Above its rung the ladder is exact only where the offset is parallel, so
    a cross-chart `TimeDep` kick on a lone acceleration needs the jet -- and
    must use `at_jet` rather than discard it.
    """
    sph = cxc.sph3d
    q0 = {"r": u.Q(2.0, "kpc"), "theta": u.Q(0.9, "rad"), "phi": u.Q(0.4, "rad")}
    v0 = {
        "r": u.Q(1.0, "kpc/Myr"),
        "theta": u.Q(0.3, "rad/Myr"),
        "phi": u.Q(0.7, "rad/Myr"),
    }
    a0 = {
        "r": u.Q(0.1, "kpc/Myr2"),
        "theta": u.Q(0.05, "rad/Myr2"),
        "phi": u.Q(-0.02, "rad/Myr2"),
    }
    tau = u.Q(2.0, "Myr")
    kick = cxfm.TimeDep.from_(
        lambda t: cxfm.Translate(
            {
                "x": u.Q(0.2, "kpc/Myr2") * t,
                "y": u.Q(-0.1, "kpc/Myr2") * t,
                "z": u.Q(0.05, "kpc/Myr2") * t,
            },
            chart=cxc.cart3d,
            semantic_kind=cxr.vel,
        )
    )
    got = cxfm.act(kick, tau, a0, sph, cxr.coord_acc, at=q0, at_jet={1: v0})
    ref = cxfm.act_jet(kick, tau, {0: q0, 1: v0, 2: a0}, sph)[2]
    for k in ref:
        unit = u.unit_of(ref[k])
        assert jnp.allclose(u.ustrip(unit, got[k]), u.ustrip(unit, ref[k]))
    # and without the velocity it asks rather than answers
    with pytest.raises(TypeError, match=r"requires jet slots"):
        cxfm.act(kick, tau, a0, sph, cxr.coord_acc, at=q0)


def test_act_jet_on_a_displacement_translate_in_a_curved_chart():
    """A ladder-order-0 offset outside the flat matching case is the point action.

    Its point action is a real translation, so the generic prolongation
    captures it entirely -- unlike a fibre offset, which that prolongation
    cannot see at all.
    """
    sph = cxc.sph3d
    q0 = {"r": u.Q(2.0, "kpc"), "theta": u.Q(0.9, "rad"), "phi": u.Q(0.4, "rad")}
    v0 = {
        "r": u.Q(1.0, "kpc/Myr"),
        "theta": u.Q(0.3, "rad/Myr"),
        "phi": u.Q(0.7, "rad/Myr"),
    }
    shift = cxfm.Translate.from_([0.5, -0.3, 0.2], "kpc")  # cart3d, k=0
    out = cxfm.act_jet(shift, None, {0: q0, 1: v0}, sph)
    assert set(out) == {0, 1}
    # the point genuinely moved, so this is not an identity dressed up
    assert not jnp.allclose(u.ustrip("kpc", out[0]["r"]), u.ustrip("kpc", q0["r"]))


def test_the_lone_slot_kick_materializes_the_timedep_only_as_often_as_it_must():
    r"""Reaching the engine through `act_jet` would re-materialize the operator.

    That dispatch hop lands back in ``add.py`` and recovers ``op0`` and ``k``
    by calling ``evaluate_at`` again -- a whole ODE solve for a curve-frame
    builder, which the sibling branch in the same function already takes care
    to avoid. Two calls are inherent here: one materialization, and one
    derivative probe for $d^{m-k}\delta/d\tau^{m-k}$. A third means the hop is
    back.
    """
    sph = cxc.sph3d
    q0 = {"r": u.Q(2.0, "kpc"), "theta": u.Q(0.9, "rad"), "phi": u.Q(0.4, "rad")}
    v0 = {
        "r": u.Q(1.0, "kpc/Myr"),
        "theta": u.Q(0.3, "rad/Myr"),
        "phi": u.Q(0.7, "rad/Myr"),
    }
    a0 = {
        "r": u.Q(0.1, "kpc/Myr2"),
        "theta": u.Q(0.05, "rad/Myr2"),
        "phi": u.Q(-0.02, "rad/Myr2"),
    }
    kick = cxfm.TimeDep.from_(
        lambda t: cxfm.Translate(
            {
                "x": u.Q(0.2, "kpc/Myr2") * t,
                "y": u.Q(-0.1, "kpc/Myr2") * t,
                "z": u.Q(0.05, "kpc/Myr2") * t,
            },
            chart=cxc.cart3d,
            semantic_kind=cxr.vel,
        )
    )

    cls = type(kick)
    original = cls.evaluate_at
    calls = []

    def counting(self, t, *a, **kw):
        calls.append(t)
        return original(self, t, *a, **kw)

    cls.evaluate_at = counting
    try:
        out = cxfm.act(
            kick, u.Q(2.0, "Myr"), a0, sph, cxr.coord_acc, at=q0, at_jet={1: v0}
        )
    finally:
        cls.evaluate_at = original

    assert len(calls) <= 2, f"materialized {len(calls)}x; the act_jet hop is back"
    # and the shortcut did not change the answer
    ref = cxfm.act_jet(kick, u.Q(2.0, "Myr"), {0: q0, 1: v0, 2: a0}, sph)[2]
    for k in ref:
        unit = u.unit_of(ref[k])
        assert jnp.allclose(u.ustrip(unit, out[k]), u.ustrip(unit, ref[k]))
