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

        out_a = cxfm.act(op, tau, a, cxc.cart3d, cxr.coord_acc, at=at, at_vel=v)
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
    @settings(max_examples=5, deadline=None)
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
            op, tq, jet[2], cxc.cart3d, cxr.coord_acc, at=jet[0], at_vel=jet[1]
        )
        assert allclose_cdict(out_v, out_gen[1], "km/s", atol=1e-6)
        assert allclose_cdict(out_a, out_gen[2], "km/s2", atol=1e-6)

    @given(tau=st.floats(0.0, 6.0))
    @settings(max_examples=5, deadline=None)
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

    def test_td_rotate_acc_requires_at_vel(self):
        op = rot_z_op()
        a = q3(1.0, 0.0, 0.0, "m/s2")
        at = q3(1.0, 0.0, 0.0, "m")
        with pytest.raises(TypeError, match="at_vel"):
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
            boost, tau, a, cxc.sph3d, cxr.coord_acc, at=SPH_AT, at_vel=v, usys=usys
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
