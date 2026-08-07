"""Unit tests for cxfm.TimeDep."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .conftest import ORIGIN_KM, ORIGIN_M, X_M as X, rot_z

OMEGA = u.Q(jnp.pi / 2, "rad/s")  # 90 deg/s


def test_point_action_materializes():
    op = cxfm.TimeDep(rot_z(OMEGA))
    out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
    assert jnp.allclose(out["y"].ustrip("m"), 1.0, atol=1e-12)
    assert jnp.allclose(out["x"].ustrip("m"), 0.0, atol=1e-12)


def test_omega_is_a_leaf():
    op = cxfm.TimeDep(rot_z(OMEGA))
    leaves = jax.tree.leaves(op, is_leaf=u.quantity.is_any_quantity)
    assert any(leaf is op.builder.omega for leaf in leaves)


def test_vmap_over_omega():
    omegas = u.Q(jnp.array([0.0, jnp.pi / 2, jnp.pi]), "rad/s")

    def y_at_t1(omega):
        op = cxfm.TimeDep(rot_z(omega))
        return cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)["y"].ustrip("m")

    ys = jax.vmap(y_at_t1)(omegas)
    assert jnp.allclose(ys, jnp.sin(jnp.array([0.0, jnp.pi / 2, jnp.pi])), atol=1e-12)


def test_tangent_data_routes_through_prolongation():
    """CRITICAL: velocity under a rotating frame gains the Rdot x term.

    At tau=0 the rotation is the identity, so a naive materialize-and-
    delegate on tangent data would return v unchanged (0); the correct
    prolongation returns omega x r = (0, pi/2, 0) m/s.
    """
    op = cxfm.TimeDep(rot_z(OMEGA))
    at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    v = {"x": u.Q(0.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
    out = cxfm.act(
        op, u.Q(0.0, "s"), v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at
    )
    assert jnp.allclose(out["y"].ustrip("m/s"), jnp.pi / 2, atol=1e-12)


def test_tau_none_raises():
    op = cxfm.TimeDep(rot_z(OMEGA))
    with pytest.raises(TypeError, match="tau"):
        cxfm.act(op, None, X, cxc.cart3d, cxr.point)


def test_from_bare_function_is_static():
    def build(t) -> cxfm.Rotate:
        return cxfm.Rotate(jnp.eye(3))

    op = cxfm.TimeDep.from_(build)
    assert isinstance(op, cxfm.TimeDep)
    # static: the function is NOT a pytree leaf
    assert build not in jax.tree.leaves(op)
    out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
    assert jnp.allclose(out["x"].ustrip("m"), 1.0)


def test_jit_caches_on_structure():
    traces = []

    @jax.jit
    def f(op, tau_val):
        traces.append(1)
        out = cxfm.act(op, u.Q(tau_val, "s"), X, cxc.cart3d, cxr.point)
        return out["y"].ustrip("m")

    f(cxfm.TimeDep(rot_z(u.Q(1.0, "rad/s"))), 0.5)
    f(cxfm.TimeDep(rot_z(u.Q(2.0, "rad/s"))), 0.5)  # same structure
    assert len(traces) == 1


TAUS = [u.Q(0.3, "s"), u.Q(1.0, "s"), u.Q(2.5, "s")]


def _act_pt(op, tau):
    return cxfm.act(op, tau, X, cxc.cart3d, cxr.point)


def test_matmul_timedep_timedep_is_pointwise():
    a = cxfm.TimeDep(rot_z(u.Q(0.3, "rad/s")))
    b = cxfm.TimeDep(rot_z(u.Q(0.5, "rad/s")))
    ab = a @ b
    assert isinstance(ab, cxfm.TimeDep)
    for tau in TAUS:
        want = _act_pt(a.materialize(tau) @ b.materialize(tau), tau)
        got = _act_pt(ab, tau)
        assert jnp.allclose(got["y"].ustrip("m"), want["y"].ustrip("m"), atol=1e-12)


def test_matmul_timedep_constant_both_orders():
    a = cxfm.TimeDep(rot_z(u.Q(0.3, "rad/s")))
    c = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    for combo in (a @ c, c @ a):
        assert isinstance(combo, cxfm.TimeDep)
    tau = u.Q(1.0, "s")
    want = _act_pt(a.materialize(tau) @ c, tau)
    got = _act_pt(a @ c, tau)
    assert jnp.allclose(got["x"].ustrip("m"), want["x"].ustrip("m"), atol=1e-12)


def test_inverse_roundtrip_and_involution():
    a = cxfm.TimeDep(rot_z(u.Q(0.7, "rad/s")))
    inv = a.inverse
    assert isinstance(inv, cxfm.TimeDep)
    # involution unwraps to the original builder
    assert a.inverse.inverse.builder is a.builder
    tau = u.Q(1.3, "s")
    roundtrip = _act_pt(inv, tau)  # applied to X
    back = cxfm.act(a, tau, roundtrip, cxc.cart3d, cxr.point)
    assert jnp.allclose(back["x"].ustrip("m"), X["x"].ustrip("m"), atol=1e-12)


def test_simplify_timedep_is_identity_op():
    a = cxfm.TimeDep(rot_z(OMEGA))
    assert cxfm.simplify(a) is a


def test_is_time_dependent_trait():
    assert cxfm.is_time_dependent(cxfm.TimeDep(rot_z(OMEGA)))
    assert not cxfm.is_time_dependent(cxfm.Rotate(jnp.eye(3)))
    # Boost's point action is delta*tau even for constant delta: True now.
    boost = cxfm.Boost.from_([1.0, 0, 0], "km/s")
    assert cxfm.is_time_dependent(boost)
    # Composed: any child
    pipe = cxfm.Rotate(jnp.eye(3)) | cxfm.TimeDep(rot_z(OMEGA))
    assert cxfm.is_time_dependent(pipe)
    assert not cxfm.is_time_dependent(cxfm.Rotate(jnp.eye(3)) | cxfm.Identity())


def test_materialize_transform():
    op = cxfm.TimeDep(rot_z(OMEGA))
    tau = u.Q(1.0, "s")
    mat = cxfm.materialize_transform(op, tau)
    assert isinstance(mat, cxfm.Rotate)
    static = cxfm.Rotate(jnp.eye(3))
    assert cxfm.materialize_transform(static, tau) is static
    pipe = static | op
    mpipe = cxfm.materialize_transform(pipe, tau)
    assert isinstance(mpipe.transforms[1], cxfm.Rotate)


def test_materialize_transform_tau_none_raises():
    with pytest.raises(TypeError, match="tau"):
        cxfm.materialize_transform(cxfm.TimeDep(rot_z(OMEGA)), None)


# ============================================================================
# Non-commuting composition order (carried finding: same-axis compositions
# commute, so they cannot pin the operand order of `@`/`_merge`).


def test_matmul_noncommuting_axes_pins_apply_order():
    """`(a @ b).R == b.R @ a.R` (a applied first), pinned with axes that don't commute.

    `test_matmul_timedep_timedep_is_pointwise` and
    `test_simplify.py::test_time_dependent_rotations_merge_pointwise` compose
    only same-axis `rot_z` builders, which commute -- a reversed-operand bug in
    `_ComposedBuilder`/`_merge` would pass every test in this file anyway.
    Rotations about *different* axes do not commute, so this pins the actual
    order against `Rotate.__matmul__`'s documented convention. The final
    assertion is the sanity check: if the two orders agreed, the axes chosen
    here would still commute and this test would be exactly as blind as the
    ones it supplements.
    """
    a = cxfm.TimeDep(
        cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=jnp.array([0.0, 0.0, 1.0]))
    )
    b = cxfm.TimeDep(
        cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=jnp.array([1.0, 0.0, 0.0]))
    )
    tau = u.Q(1.0, "s")
    Ra, Rb = a.materialize(tau).R, b.materialize(tau).R

    R_ab = (a @ b).materialize(tau).R
    R_ba = (b @ a).materialize(tau).R

    assert jnp.allclose(R_ab, Rb @ Ra, atol=1e-12)
    assert jnp.allclose(R_ba, Ra @ Rb, atol=1e-12)
    # Sanity check: the two orders must actually disagree, or these axes
    # commute and neither assertion above has any signal.
    assert not jnp.allclose(R_ab, R_ba, atol=1e-6)


# ============================================================================
# Direct pushforward test (carried finding: `TimeDep.pushforward` was only
# ever reached transitively, never exercised directly).


def test_pushforward_frozen_tau():
    """`TimeDep.pushforward` freezes tau, materializes, then pushes forward."""
    op = cxfm.TimeDep(rot_z(OMEGA))
    tau = u.Q(1.0, "s")
    v = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
    out = cxfm.pushforward(op, tau, v, cxc.cart3d, cxr.coord_vel)
    want = cxfm.pushforward(op.materialize(tau), tau, v, cxc.cart3d, cxr.coord_vel)
    assert jnp.allclose(out["y"].ustrip("m/s"), want["y"].ustrip("m/s"), atol=1e-12)
    # omega=pi/2 rad/s, tau=1s: a 90 deg rotation sends (1,0,0) -> (0,1,0).
    assert jnp.allclose(out["y"].ustrip("m/s"), 1.0, atol=1e-12)
    assert jnp.allclose(out["x"].ustrip("m/s"), 0.0, atol=1e-12)


# ============================================================================
# gamma-as-leaf: a second (curvilinear) parameter living as a builder field,
# differentiable and vmappable -- something the OLD `Array | Callable` design
# could not express.


class CircleFrame(eqx.Module):
    """Frame translated to a point on a circle of radius r at angle gamma.

    gamma is a builder leaf: differentiate/vmap it directly.
    """

    r: u.AbstractQuantity
    gamma: jax.Array  # radians, raw

    def __call__(self, tau):
        del tau
        delta = {
            "x": self.r * jnp.cos(self.gamma),
            "y": self.r * jnp.sin(self.gamma),
            "z": self.r * 0,
        }
        return cxfm.Translate(delta, chart=cxc.cart3d)


class MovingAlongCircle(eqx.Module):
    """gamma = gamma(tau): time-dependent curvilinear frame."""

    r: u.AbstractQuantity
    gamma_rate: u.AbstractQuantity  # rad/s

    def __call__(self, tau):
        g = (self.gamma_rate * tau).ustrip("rad")
        return CircleFrame(self.r, g)(None)


def test_gamma_as_leaf_gradient():
    """d/dgamma of the frame origin's x is -r sin(gamma)."""

    def x_of(gamma):
        op = cxfm.TimeDep(CircleFrame(u.Q(2.0, "m"), gamma))
        out = cxfm.act(op, u.Q(0.0, "s"), ORIGIN_M, cxc.cart3d, cxr.point)
        return out["x"].ustrip("m")

    g0 = jnp.pi / 3
    assert jnp.allclose(jax.grad(x_of)(g0), -2.0 * jnp.sin(g0), atol=1e-12)


def test_gamma_of_tau_velocity_transport():
    """The gamma-dot term: comoving-point velocity is an r*gamma_rate tangent."""
    op = cxfm.TimeDep(MovingAlongCircle(u.Q(2.0, "m"), u.Q(0.5, "rad/s")))
    at = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    v = {"x": u.Q(0.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
    out = cxfm.act(
        op, u.Q(0.0, "s"), v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at
    )
    # frame origin moves along circle: at gamma=0, d/dt (r cos, r sin) = (0, r*rate)
    assert jnp.allclose(out["y"].ustrip("m/s"), 1.0, atol=1e-12)
    assert jnp.allclose(out["x"].ustrip("m/s"), 0.0, atol=1e-12)


# ============================================================================
# Regression: a builder returning a Composed that hides a fibre offset


def _fibre_kick(t):
    """A velocity kick growing at 5 km/s2 -- a ladder-order-1 fibre offset."""
    return cxfm.Translate(
        {"x": u.Q(5.0, "km/s2") * t, "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")},
        chart=cxc.cart3d,
        semantic_kind=cxr.vel,
    )


_ACC = {"x": u.Q(1.0, "km/s2"), "y": u.Q(1.0, "km/s2"), "z": u.Q(1.0, "km/s2")}
_AT_VEL = {"x": u.Q(0.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}


def _acc_x(op):
    out = cxfm.act(
        op, u.Q(2.0, "s"), _ACC, cxc.cart3d, cxr.coord_acc, at=ORIGIN_KM, at_vel=_AT_VEL
    )
    return out["x"].ustrip("km/s2")


def test_supported_fibre_kick_spellings():
    """The two supported spellings both pick up the kick's 5 km/s2 rate."""
    bare = cxfm.TimeDep.from_(_fibre_kick)
    assert jnp.allclose(_acc_x(bare), 6.0)

    shift = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
    assert jnp.allclose(_acc_x(shift | bare), 6.0)


def test_builder_returning_composed_fibre_offset_raises():
    """A fibre offset hidden inside a builder-returned `Composed` must be loud.

    The ladder carve-out only recognises an offset that is the *whole*
    materialized transform; inside a composite the generic funnel is blind to
    it (identity point action) and would silently return acc unchanged. The
    fix must raise, not quietly give 1.0.
    """
    shift = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
    op = cxfm.TimeDep.from_(lambda t: shift | _fibre_kick(t))
    with pytest.raises(TypeError, match="composite containing a fibre offset"):
        _acc_x(op)


def test_builder_returning_composed_fibre_offset_raises_on_jet():
    """Same guard on the jet path (the `Coordinate`-bundle route)."""
    shift = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
    op = cxfm.TimeDep.from_(lambda t: shift | _fibre_kick(t))
    jet = {0: ORIGIN_KM, 1: _AT_VEL, 2: _ACC}
    with pytest.raises(TypeError, match="composite containing a fibre offset"):
        cxfm.act_jet(op, u.Q(2.0, "s"), jet, cxc.cart3d)


def test_builder_returning_composed_without_fibre_offset_is_fine():
    """The guard fires only for fibre offsets: a point-acting Composed passes.

    The oracle is the same pipeline spelled the other way round: a builder
    returning ``shift | rot(t)`` must act exactly like ``shift |
    TimeDep(rot)``.
    """
    shift = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
    op = cxfm.TimeDep.from_(lambda t: shift | rot_z(OMEGA)(t))
    assert jnp.allclose(_acc_x(op), _acc_x(shift | cxfm.TimeDep(rot_z(OMEGA))))


# ============================================================================
# Regression: simplify must preserve semantics for non-`@` transforms


def _drift(vx):
    return cxfm.TimeDep(
        cxfm.UniformTranslation(
            {"x": u.Q(vx, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")},
            chart=cxc.cart3d,
        )
    )


def test_simplify_of_composed_translations_preserves_action():
    """`simplify` merges families with `@`; `Translate` has no `@`.

    Without the `|` fallback in the composed builder this raises
    ``TypeError: unsupported operand type(s) for @`` at materialize time.
    """
    pipe = _drift(1.0) | _drift(2.0)
    simplified = cxfm.simplify(pipe)
    for tau in (u.Q(0.0, "s"), u.Q(2.0, "s"), u.Q(-3.5, "s")):
        got = cxfm.act(simplified, tau, ORIGIN_KM, cxc.cart3d, cxr.point)["x"]
        want = cxfm.act(pipe, tau, ORIGIN_KM, cxc.cart3d, cxr.point)["x"]
        assert jnp.allclose(got.ustrip("km"), want.ustrip("km"))
        assert jnp.allclose(got.ustrip("km"), 3.0 * tau.ustrip("s"))


def test_simplify_does_not_fold_a_static_neighbour_into_a_timedep():
    """`simplify` must not turn a working pipeline into one that raises.

    Folding a static fibre offset into a `_ComposedBuilder` makes the family
    materialize to a `Composed` containing that offset -- exactly the spelling
    `add.py` rejects. `Composed` already represents the pair correctly.
    """
    kick = cxfm.Translate(
        {"x": u.Q(1.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")},
        chart=cxc.cart3d,
        semantic_kind=cxr.vel,
    )
    spin = cxfm.TimeDep(
        cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=jnp.array([0.0, 0.0, 1.0]))
    )
    pipe = cxfm.Composed((kick, spin))
    vel = {"x": u.Q(1.0, "km/s"), "y": u.Q(1.0, "km/s"), "z": u.Q(0.0, "km/s")}

    def vel_x(op):
        out = cxfm.act(op, u.Q(1.0, "s"), vel, cxc.cart3d, cxr.coord_vel, at=ORIGIN_KM)
        return out["x"].ustrip("km/s")

    assert jnp.allclose(vel_x(cxfm.simplify(pipe)), vel_x(pipe))


# ============================================================================
# Group membership and the `@` operand contract


def test_groups_is_diffeomorphism_only():
    """A `TimeDep` claims only the widest group: its builder can return anything.

    Narrower claims (affine, orthogonal, ...) would be unsound -- the value is
    unknown until tau -- so the set must be exactly the diffeomorphism group,
    not merely non-empty.
    """
    assert cxfm.TimeDep.groups() == frozenset({cxfm.DiffeomorphismGroup})


@pytest.mark.parametrize("other", [5, "rot", None])
def test_matmul_with_a_non_transform_raises_typeerror(other):
    """`TimeDep @ non-transform` must fall through to Python's `TypeError`.

    Returning `NotImplemented` (rather than raising, or wrapping the operand)
    is what lets the other operand get a turn and, failing that, produces the
    standard unsupported-operand error. Both orders are pinned.
    """
    op = cxfm.TimeDep(rot_z(OMEGA))
    with pytest.raises(TypeError, match="unsupported operand type"):
        op @ other
    with pytest.raises(TypeError, match="unsupported operand type"):
        other @ op


# ============================================================================
# The fibre-offset ladder rule: an order-k offset touches slot m only for
# m >= k, and then by d^(m-k)delta/dtau^(m-k).


_ACC_RATE = u.Q(3.0, "km/s3")  # the jerk-like rate of the acceleration kick
_TAU = u.Q(2.0, "s")
_VEL = {"x": u.Q(1.0, "km/s"), "y": u.Q(2.0, "km/s"), "z": u.Q(3.0, "km/s")}


def _acc_kick(t):
    """An acceleration kick growing at 3 km/s3 -- a ladder-order-2 fibre offset."""
    return cxfm.Translate(
        {"x": _ACC_RATE * t, "y": u.Q(0.0, "km/s2"), "z": u.Q(0.0, "km/s2")},
        chart=cxc.cart3d,
        semantic_kind=cxr.acc,
    )


def test_lower_order_fibre_is_untouched_by_a_higher_order_offset():
    """An acceleration offset (m < k) must leave velocity data exactly alone.

    The companion assertion is the discriminator: the SAME operator at the
    SAME tau does move the acceleration slot, so "unchanged" here is the
    ladder rule and not a dead operator.
    """
    op = cxfm.TimeDep.from_(_acc_kick)
    out = cxfm.act(op, _TAU, _VEL, cxc.cart3d, cxr.coord_vel, at=ORIGIN_KM)
    for k, v in _VEL.items():
        assert jnp.allclose(out[k].ustrip("km/s"), v.ustrip("km/s"), atol=1e-12)

    moved = cxfm.act(
        op, _TAU, _ACC, cxc.cart3d, cxr.coord_acc, at=ORIGIN_KM, at_vel=_VEL
    )
    assert jnp.allclose(moved["x"].ustrip("km/s2"), 1.0 + 6.0, atol=1e-12)


def test_matching_order_fibre_offset_adds_delta_at_tau():
    """At matching order (m == k) the offset added is delta(tau) itself.

    No derivative is taken: ``_fibre_ladder_op``'s ``n == 0`` fast path reuses
    the already-materialized ``op0.delta``.
    """
    op = cxfm.TimeDep.from_(_acc_kick)
    got = cxfm.act(op, _TAU, _ACC, cxc.cart3d, cxr.coord_acc, at=ORIGIN_KM, at_vel=_VEL)
    # delta(2 s) = 3 km/s3 * 2 s = 6 km/s2, added to the 1 km/s2 datum; the
    # kick is x-only, so y and z keep theirs.
    assert jnp.allclose(got["x"].ustrip("km/s2"), 7.0, atol=1e-12)
    assert jnp.allclose(got["y"].ustrip("km/s2"), 1.0, atol=1e-12)
    assert jnp.allclose(got["z"].ustrip("km/s2"), 1.0, atol=1e-12)


def test_fibre_offset_jet_is_slotwise_by_ladder_order():
    """The jet path applies the same ladder rule slot by slot.

    Slot 0 must equal the plain point action (identity, for a fibre offset),
    slot 1 must be untouched (m < k), and slot 2 must gain delta(tau).
    """
    op = cxfm.TimeDep.from_(_acc_kick)
    jet = {0: ORIGIN_KM, 1: _VEL, 2: _ACC}
    out = cxfm.act_jet(op, _TAU, jet, cxc.cart3d)

    want0 = cxfm.act(op, _TAU, ORIGIN_KM, cxc.cart3d, cxr.point)
    for k in ORIGIN_KM:
        assert jnp.allclose(out[0][k].ustrip("km"), want0[k].ustrip("km"), atol=1e-12)
    for k, v in _VEL.items():
        assert jnp.allclose(out[1][k].ustrip("km/s"), v.ustrip("km/s"), atol=1e-12)
    assert jnp.allclose(out[2]["x"].ustrip("km/s2"), 7.0, atol=1e-12)
    assert jnp.allclose(out[2]["y"].ustrip("km/s2"), 1.0, atol=1e-12)


def test_timedep_boost_keeps_its_point_action():
    """A `TimeDep`-wrapped `Boost` must NOT be routed onto the fibre ladder.

    `Boost.delta` is a velocity, but its point action is ``delta * tau`` --
    real, and recovered by the generic funnel's differentiation. The ladder
    carve-out is only for offsets with an *identity* point action, so the
    wrapped boost must act exactly like the bare one.
    """
    boost = cxfm.Boost.from_([1.0, 0.0, 0.0], "km/s")
    op = cxfm.TimeDep.from_(lambda t: boost)
    got = cxfm.act(op, _TAU, _VEL, cxc.cart3d, cxr.coord_vel, at=ORIGIN_KM)
    want = cxfm.act(boost, _TAU, _VEL, cxc.cart3d, cxr.coord_vel, at=ORIGIN_KM)
    for k in _VEL:
        assert jnp.allclose(got[k].ustrip("km/s"), want[k].ustrip("km/s"), atol=1e-12)
    # d/dtau (delta * tau) = delta: the velocity gains the full 1 km/s.
    assert jnp.allclose(got["x"].ustrip("km/s"), 2.0, atol=1e-12)
