"""Test spatial linear transforms."""

__all__: tuple[str, ...] = ()

import equinox as eqx
import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

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
