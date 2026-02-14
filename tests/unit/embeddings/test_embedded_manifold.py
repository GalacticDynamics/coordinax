"""Tests for embedded manifold support."""

import jax.numpy as jnp
from hypothesis import given, settings, strategies as st

import unxt as u
from unxt.quantity import AllowValue

import coordinax as cx
import coordinax.charts as cxc
import coordinax.embeddings as cxe
import coordinax.roles as cxr
import coordinax.transforms as cxt


def _embedded_twosphere(
    R: u.Q | None = None,
) -> cxe.EmbeddedManifold:
    params = {} if R is None else {"R": R}
    return cxe.EmbeddedManifold(
        intrinsic_chart=cxc.twosphere,
        ambient_chart=cxc.cart3d,
        params=params,
    )


def _to_rad(x: u.AbstractQuantity) -> jnp.ndarray:
    return u.ustrip(AllowValue, "rad", x)


def _wrap_angle_rad(dphi: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi))


# ===================================================================
# Deterministic tests


def test_twosphere_embed_pos_known_angles() -> None:
    R = u.Q(2.0, "km")
    rep = _embedded_twosphere(R)

    p0 = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(0.0, "rad")}
    p1 = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    p2 = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(jnp.pi / 2, "rad")}

    q0 = cxe.embed_point(rep, p0)
    q1 = cxe.embed_point(rep, p1)
    q2 = cxe.embed_point(rep, p2)

    unit = u.unit_of(R)

    x0 = u.uconvert(unit, q0["x"]).value
    y0 = u.uconvert(unit, q0["y"]).value
    z0 = u.uconvert(unit, q0["z"]).value

    x1 = u.uconvert(unit, q1["x"]).value
    y1 = u.uconvert(unit, q1["y"]).value
    z1 = u.uconvert(unit, q1["z"]).value

    x2 = u.uconvert(unit, q2["x"]).value
    y2 = u.uconvert(unit, q2["y"]).value
    z2 = u.uconvert(unit, q2["z"]).value

    assert jnp.allclose(x0, 0.0)
    assert jnp.allclose(y0, 0.0)
    assert jnp.allclose(z0, R.value)

    assert jnp.allclose(x1, R.value)
    assert jnp.allclose(y1, 0.0)
    assert jnp.allclose(z1, 0.0)

    assert jnp.allclose(x2, 0.0)
    assert jnp.allclose(y2, R.value)
    assert jnp.allclose(z2, 0.0)


def test_twosphere_project_pos_roundtrip_known() -> None:
    rep = _embedded_twosphere(u.Q(3.0, "km"))
    p = {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(-jnp.pi / 4, "rad")}

    q = cxe.embed_point(rep, p)
    p2 = cxe.project_point(rep, q)

    theta1 = _to_rad(p["theta"])
    phi1 = _to_rad(p["phi"])
    theta2 = _to_rad(p2["theta"])
    phi2 = _to_rad(p2["phi"])

    assert jnp.allclose(theta1, theta2, atol=1e-6)
    assert jnp.allclose(_wrap_angle_rad(phi2 - phi1), 0.0, atol=1e-6)


# ===================================================================
# Property-based tests


eps = 1e-3

theta_strategy = st.floats(
    min_value=eps,
    max_value=float(jnp.pi - eps),
    allow_nan=False,
    allow_infinity=False,
).map(lambda x: u.Angle(x, "rad"))

phi_strategy = st.floats(
    min_value=-float(jnp.pi),
    max_value=float(jnp.pi),
    allow_nan=False,
    allow_infinity=False,
).map(lambda x: u.Angle(x, "rad"))


@given(theta=theta_strategy, phi=phi_strategy)
@settings(max_examples=60, deadline=None)
def test_embed_project_roundtrip(theta: u.Angle, phi: u.Angle) -> None:
    rep = _embedded_twosphere(u.Q(1.0, "km"))
    p = {"theta": theta, "phi": phi}

    q = cxe.embed_point(rep, p)
    p2 = cxe.project_point(rep, q)

    theta1 = _to_rad(theta)
    phi1 = _to_rad(phi)
    theta2 = _to_rad(p2["theta"])
    phi2 = _to_rad(p2["phi"])

    assert jnp.allclose(theta2, theta1, atol=1e-6)
    assert jnp.allclose(_wrap_angle_rad(phi2 - phi1), 0.0, atol=1e-6)


@given(theta=theta_strategy, phi=phi_strategy)
@settings(max_examples=60, deadline=None)
def test_tangent_basis_orthonormal(theta: u.Angle, phi: u.Angle) -> None:
    rep = _embedded_twosphere(u.Q(2.0, "km"))
    p = {"theta": theta, "phi": phi}

    B = cxt.frame_cart(rep, at=p)
    BTB = jnp.matmul(B.T, B)

    assert jnp.allclose(BTB, jnp.eye(2), atol=1e-6)

    q = cxe.embed_point(rep, p)
    x = u.ustrip(AllowValue, q["x"])
    y = u.ustrip(AllowValue, q["y"])
    z = u.ustrip(AllowValue, q["z"])
    r = jnp.sqrt(x**2 + y**2 + z**2)
    r_hat = jnp.array([x, y, z]) / r
    dots = jnp.einsum("i,ij->j", r_hat, B)

    assert jnp.allclose(dots, jnp.zeros(2), atol=1e-6)


@given(
    theta=theta_strategy,
    phi=phi_strategy,
    v_theta=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    v_phi=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
)
@settings(max_examples=60, deadline=None)
def test_pushforward_pullback_consistency(
    theta: u.Angle,
    phi: u.Angle,
    v_theta: float,
    v_phi: float,
) -> None:
    rep = _embedded_twosphere(u.Q(1.0, "km"))
    p = {"theta": theta, "phi": phi}
    v = {
        "theta": u.Q(v_theta, "km/s"),
        "phi": u.Q(v_phi, "km/s"),
    }

    v_cart = cxe.embed_tangent(rep, v, at=p)
    v_back = cxe.project_tangent(rep, v_cart, at=p)

    vt1 = u.uconvert("km/s", v["theta"]).value
    vp1 = u.uconvert("km/s", v["phi"]).value
    vt2 = u.uconvert("km/s", v_back["theta"]).value
    vp2 = u.uconvert("km/s", v_back["phi"]).value

    assert jnp.allclose(vt2, vt1, atol=1e-6)
    assert jnp.allclose(vp2, vp1, atol=1e-6)


@given(
    theta=theta_strategy,
    phi=phi_strategy,
    v_theta=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    v_phi=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
)
@settings(max_examples=40, deadline=None)
def test_vconvert_roundtrip_with_embedded(
    theta: u.Angle,
    phi: u.Angle,
    v_theta: float,
    v_phi: float,
) -> None:
    rep = _embedded_twosphere(u.Q(1.0, "km"))
    p = {"theta": theta, "phi": phi}
    v = {
        "theta": u.Q(v_theta, "km/s"),
        "phi": u.Q(v_phi, "km/s"),
    }

    qvec = cx.Vector(data=p, chart=rep, role=cxr.phys_disp)
    vvec = cx.Vector(data=v, chart=rep, role=cxr.phys_vel)

    v_cart = vvec.vconvert(cxc.cart3d, qvec)
    v_back = v_cart.vconvert(rep, qvec)

    vt1 = u.uconvert("km/s", v["theta"]).value
    vp1 = u.uconvert("km/s", v["phi"]).value
    vt2 = u.uconvert("km/s", v_back.data["theta"]).value
    vp2 = u.uconvert("km/s", v_back.data["phi"]).value

    assert jnp.allclose(vt2, vt1, atol=1e-6)
    assert jnp.allclose(vp2, vp1, atol=1e-6)
