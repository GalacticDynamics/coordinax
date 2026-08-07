"""Unit tests for cxfm.Parametric."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm


class RotZ(eqx.Module):
    """Uniform rotation about z: a builder with omega as a leaf."""

    omega: u.AbstractQuantity  # angular frequency, e.g. rad/s

    def __call__(self, tau):
        th = (self.omega * tau).ustrip("rad")
        st, ct = jnp.sin(th), jnp.cos(th)
        R = jnp.stack(
            [
                jnp.stack([ct, -st, jnp.zeros_like(th)], axis=-1),
                jnp.stack([st, ct, jnp.zeros_like(th)], axis=-1),
                jnp.stack(
                    [jnp.zeros_like(th), jnp.zeros_like(th), jnp.ones_like(th)],
                    axis=-1,
                ),
            ],
            axis=-2,
        )
        return cxfm.Rotate(R)


OMEGA = u.Q(jnp.pi / 2, "rad/s")  # 90 deg/s
X = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}


def test_point_action_materializes():
    op = cxfm.Parametric(RotZ(OMEGA))
    out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
    assert jnp.allclose(out["y"].ustrip("m"), 1.0, atol=1e-12)
    assert jnp.allclose(out["x"].ustrip("m"), 0.0, atol=1e-12)


def test_omega_is_a_leaf():
    op = cxfm.Parametric(RotZ(OMEGA))
    leaves = jax.tree.leaves(op, is_leaf=u.quantity.is_any_quantity)
    assert any(leaf is op.builder.omega for leaf in leaves)


def test_grad_wrt_omega():
    def y_at_t1(omega_val):
        op = cxfm.Parametric(RotZ(u.Q(omega_val, "rad/s")))
        out = cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)
        return out["y"].ustrip("m")

    # d/domega [sin(omega * 1s)] at omega=0 is 1 (per rad/s)
    g = jax.grad(y_at_t1)(0.0)
    assert jnp.allclose(g, 1.0, atol=1e-12)


def test_vmap_over_omega():
    omegas = u.Q(jnp.array([0.0, jnp.pi / 2, jnp.pi]), "rad/s")

    def y_at_t1(omega):
        op = cxfm.Parametric(RotZ(omega))
        return cxfm.act(op, u.Q(1.0, "s"), X, cxc.cart3d, cxr.point)["y"].ustrip("m")

    ys = jax.vmap(y_at_t1)(omegas)
    assert jnp.allclose(ys, jnp.sin(jnp.array([0.0, jnp.pi / 2, jnp.pi])), atol=1e-12)


def test_tangent_data_routes_through_prolongation():
    """CRITICAL: velocity under a rotating frame gains the Rdot x term.

    At tau=0 the rotation is the identity, so a naive materialize-and-
    delegate on tangent data would return v unchanged (0); the correct
    prolongation returns omega x r = (0, pi/2, 0) m/s.
    """
    op = cxfm.Parametric(RotZ(OMEGA))
    at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    v = {"x": u.Q(0.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
    out = cxfm.act(
        op, u.Q(0.0, "s"), v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at
    )
    assert jnp.allclose(out["y"].ustrip("m/s"), jnp.pi / 2, atol=1e-12)


def test_tau_none_raises():
    op = cxfm.Parametric(RotZ(OMEGA))
    with pytest.raises(TypeError, match="tau"):
        cxfm.act(op, None, X, cxc.cart3d, cxr.point)


def test_from_bare_function_is_static():
    def build(t) -> cxfm.Rotate:
        return cxfm.Rotate(jnp.eye(3))

    op = cxfm.Parametric.from_(build)
    assert isinstance(op, cxfm.Parametric)
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

    f(cxfm.Parametric(RotZ(u.Q(1.0, "rad/s"))), 0.5)
    f(cxfm.Parametric(RotZ(u.Q(2.0, "rad/s"))), 0.5)  # same structure
    assert len(traces) == 1


TAUS = [u.Q(0.3, "s"), u.Q(1.0, "s"), u.Q(2.5, "s")]


def _act_pt(op, tau):
    return cxfm.act(op, tau, X, cxc.cart3d, cxr.point)


def test_matmul_parametric_parametric_is_pointwise():
    a = cxfm.Parametric(RotZ(u.Q(0.3, "rad/s")))
    b = cxfm.Parametric(RotZ(u.Q(0.5, "rad/s")))
    ab = a @ b
    assert isinstance(ab, cxfm.Parametric)
    for tau in TAUS:
        want = _act_pt(a.materialize(tau) @ b.materialize(tau), tau)
        got = _act_pt(ab, tau)
        assert jnp.allclose(got["y"].ustrip("m"), want["y"].ustrip("m"), atol=1e-12)


def test_matmul_parametric_constant_both_orders():
    a = cxfm.Parametric(RotZ(u.Q(0.3, "rad/s")))
    c = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    for combo in (a @ c, c @ a):
        assert isinstance(combo, cxfm.Parametric)
    tau = u.Q(1.0, "s")
    want = _act_pt(a.materialize(tau) @ c, tau)
    got = _act_pt(a @ c, tau)
    assert jnp.allclose(got["x"].ustrip("m"), want["x"].ustrip("m"), atol=1e-12)


def test_inverse_roundtrip_and_involution():
    a = cxfm.Parametric(RotZ(u.Q(0.7, "rad/s")))
    inv = a.inverse
    assert isinstance(inv, cxfm.Parametric)
    # involution unwraps to the original builder
    assert a.inverse.inverse.builder is a.builder
    tau = u.Q(1.3, "s")
    roundtrip = _act_pt(inv, tau)  # applied to X
    back = cxfm.act(a, tau, roundtrip, cxc.cart3d, cxr.point)
    assert jnp.allclose(back["x"].ustrip("m"), X["x"].ustrip("m"), atol=1e-12)


def test_merge_two_parametrics_in_composed():
    a = cxfm.Parametric(RotZ(u.Q(0.3, "rad/s")))
    b = cxfm.Parametric(RotZ(u.Q(0.5, "rad/s")))
    merged = cxfm.simplify(a | b)
    # Improvement over the old design: time-dependent rotations DO merge.
    assert isinstance(merged, cxfm.Parametric)
    tau = u.Q(0.9, "s")
    want = _act_pt(b.materialize(tau), tau)  # a then b == pipe order
    want = cxfm.act(a.materialize(tau), tau, X, cxc.cart3d, cxr.point)
    want = cxfm.act(b.materialize(tau), tau, want, cxc.cart3d, cxr.point)
    got = _act_pt(merged, tau)
    assert jnp.allclose(got["y"].ustrip("m"), want["y"].ustrip("m"), atol=1e-12)


def test_simplify_parametric_is_identity_op():
    a = cxfm.Parametric(RotZ(OMEGA))
    assert cxfm.simplify(a) is a


def test_is_time_dependent_trait():
    assert cxfm.is_time_dependent(cxfm.Parametric(RotZ(OMEGA)))
    assert not cxfm.is_time_dependent(cxfm.Rotate(jnp.eye(3)))
    # Boost's point action is delta*tau even for constant delta: True now.
    boost = cxfm.Boost.from_([1.0, 0, 0], "km/s")
    assert cxfm.is_time_dependent(boost)
    # Composed: any child
    pipe = cxfm.Rotate(jnp.eye(3)) | cxfm.Parametric(RotZ(OMEGA))
    assert cxfm.is_time_dependent(pipe)
    assert not cxfm.is_time_dependent(cxfm.Rotate(jnp.eye(3)) | cxfm.Identity())


def test_materialize_transform():
    op = cxfm.Parametric(RotZ(OMEGA))
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
        cxfm.materialize_transform(cxfm.Parametric(RotZ(OMEGA)), None)
