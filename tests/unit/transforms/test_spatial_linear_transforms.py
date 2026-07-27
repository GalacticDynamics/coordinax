"""Test spatial linear transforms."""

__all__: tuple[str, ...] = ()

import equinox as eqx
import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm


def _to_np(x: object, unit: str) -> np.ndarray:
    assert isinstance(x, u.AbstractQuantity)
    return np.asarray(u.ustrip(unit, x), dtype=float)


def test_scale_from_factors_singular_raises_under_jit() -> None:
    """A zero scale factor is rejected even under jit (no tracer bool)."""
    build = eqx.filter_jit(cxfm.Scale.from_factors)
    with pytest.raises(eqx.EquinoxRuntimeError, match="invertible"):
        jax.block_until_ready(build(jnp.asarray([2.0, 0.0, 4.0])).S)


def test_scale_from_factors_nonsingular_jits() -> None:
    """A valid Scale builds cleanly under jit."""
    op = eqx.filter_jit(cxfm.Scale.from_factors)(jnp.asarray([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(np.asarray(jnp.diag(op.S)), [2.0, 3.0, 4.0])


def test_public_surface_includes_scale_and_shear() -> None:
    """`coordinax.transforms` exports Scale and Shear."""
    assert hasattr(cxfm, "Scale")
    assert hasattr(cxfm, "Shear")


def test_scale_from_factors_applies_axiswise_scaling() -> None:
    """Scale.from_factors scales each Cartesian axis independently."""
    op = cxfm.Scale.from_factors([2, 3, 4])
    q = u.Q(jnp.asarray([1, 2, 3]), "m")

    out = cxfm.act(op, None, q)
    np.testing.assert_allclose(_to_np(out, "m"), np.asarray([2, 6, 12]))


def test_scale_inverse_roundtrip_is_identity() -> None:
    """Applying scale then inverse returns the original point."""
    op = cxfm.Scale.from_factors([2, 0.5, 4])
    q = u.Q(jnp.asarray([3, -2, 1.5]), "km")

    fwd = cxfm.act(op, None, q)
    back = cxfm.act(op.inverse, None, fwd)
    np.testing.assert_allclose(_to_np(back, "km"), _to_np(q, "km"), rtol=0, atol=1e-12)


def test_shear_matrix_applies_linear_shear() -> None:
    """Shear applies a standard linear shear matrix in Cartesian coordinates."""
    # x' = x + y, y' = y, z' = z
    op = cxfm.Shear(jnp.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]))
    q = u.Q(jnp.asarray([1, 2, 3]), "m")

    out = cxfm.act(op, None, q)
    np.testing.assert_allclose(_to_np(out, "m"), np.asarray([3, 2, 3]))


def test_simplify_identity_scale_and_shear_to_identity() -> None:
    """Identity matrices simplify to the shared identity transform."""
    s = cxfm.Scale.from_factors([1, 1, 1])
    h = cxfm.Shear(jnp.eye(3))

    assert cxfm.simplify(s) is cxfm.identity
    assert cxfm.simplify(h) is cxfm.identity


# ============================================================================
# Kinematic (velocity / acceleration) acts — constant linear maps need no `at`


def _vel(x, y, z):
    return {"x": u.Q(x, "m/s"), "y": u.Q(y, "m/s"), "z": u.Q(z, "m/s")}


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (cxfm.Scale.from_factors([2.0, 3.0, 4.0]), (2.0, 3.0, 4.0)),
        (cxfm.Reflect.from_normal([1.0, 0.0, 0.0]), (-1.0, 1.0, 1.0)),
        (
            cxfm.Shear(
                jnp.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            ),
            (2.0, 1.0, 1.0),
        ),
    ],
)
def test_linear_transform_acts_on_velocity_without_at(op, expected) -> None:
    """A constant linear map transforms a Cartesian velocity as v -> M v, no `at`."""
    out = cxfm.act(op, None, _vel(1.0, 1.0, 1.0), cxc.cart3d, cxr.coord_vel)
    got = tuple(round(float(u.ustrip("m/s", out[c])), 6) for c in ("x", "y", "z"))
    assert got == expected


def test_linear_transform_acts_on_acceleration_without_at() -> None:
    """Acceleration also transforms as a -> M a for a constant linear map."""
    op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
    acc = {"x": u.Q(1.0, "m/s2"), "y": u.Q(0.0, "m/s2"), "z": u.Q(2.0, "m/s2")}
    out = cxfm.act(op, None, acc, cxc.cart3d, cxr.coord_acc)
    got = tuple(round(float(u.ustrip("m/s2", out[c])), 6) for c in ("x", "y", "z"))
    assert got == (2.0, 0.0, 8.0)


def test_linear_velocity_matches_generic_prolongation_with_at() -> None:
    """Keystone: the no-`at` fast path equals the generic autodiff prolongation.

    Compare against `cxfm.act_jet` (which differentiates the point action) with
    `at` as jet slot 0 — not merely against the same pushforward path invoked
    with `at`, which for a static linear map is the identical code.
    """
    op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
    v = _vel(1.0, -2.0, 0.5)
    at = {"x": u.Q(2.0, "m"), "y": u.Q(-1.0, "m"), "z": u.Q(3.0, "m")}
    fast = cxfm.act(op, None, v, cxc.cart3d, cxr.coord_vel)
    generic = cxfm.act_jet(op, None, {0: at, 1: v}, cxc.cart3d)[1]
    for c in ("x", "y", "z"):
        np.testing.assert_allclose(
            _to_np(fast[c], "m/s"), _to_np(generic[c], "m/s"), atol=1e-12
        )


def test_linear_velocity_noncartesian_roundtrips_with_at() -> None:
    """Non-Cartesian pushforward (needs `at`): Scale then inverse recovers v."""
    op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
    at = {"r": u.Q(2.0, "kpc"), "theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
    v = {
        "r": u.Q(0.1, "kpc/Myr"),
        "theta": u.Q(0.2, "rad/Myr"),
        "phi": u.Q(0.3, "rad/Myr"),
    }
    fwd = cxfm.act(op, None, v, cxc.sph3d, cxr.tangent_geom, cxr.coord_vel, at=at)
    at2 = cxfm.act(op, None, at, cxc.sph3d, cxr.point)
    back = cxfm.act(
        op.inverse, None, fwd, cxc.sph3d, cxr.tangent_geom, cxr.coord_vel, at=at2
    )
    for k, vk in v.items():
        unit = u.unit_of(vk).to_string()
        np.testing.assert_allclose(_to_np(back[k], unit), _to_np(vk, unit), atol=1e-9)


def test_linear_velocity_on_product_chart_is_factorwise() -> None:
    """A 3x3 Scale acts factorwise on each Cart3D factor of a 6D velocity."""
    ps = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
    op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
    v = {f"{f}.{c}": u.Q(1.0, "m/s") for f in ("q", "p") for c in "xyz"}
    out = cxfm.act(op, None, v, ps, cxr.tangent_geom, cxr.coord_vel)
    got = [float(out[k].value) for k in ("q.x", "q.y", "q.z", "p.x", "p.y", "p.z")]
    assert got == [2.0, 3.0, 4.0, 2.0, 3.0, 4.0]


def test_linear_acceleration_on_product_chart_is_factorwise() -> None:
    """A 3x3 Scale acts factorwise on each Cart3D factor of a 6D acceleration."""
    ps = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
    op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
    a = {f"{f}.{c}": u.Q(1.0, "m/s2") for f in ("q", "p") for c in "xyz"}
    out = cxfm.act(op, None, a, ps, cxr.tangent_geom, cxr.coord_acc)
    got = [float(out[k].value) for k in ("q.x", "q.y", "q.z", "p.x", "p.y", "p.z")]
    assert got == [2.0, 3.0, 4.0, 2.0, 3.0, 4.0]
