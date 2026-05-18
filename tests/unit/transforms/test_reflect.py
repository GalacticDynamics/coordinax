"""Tests for the Reflect frame transform."""

__all__: tuple[str, ...] = ()

from typing import Any, cast

import jax.numpy as jnp
import numpy as np

import unxt as u

import coordinax.main as cx
import coordinax.transforms as cxfm
from .conftest import EXPECTED_IDENTITY, EXPECTED_REFLECT


def _extract_xyz(result: Any) -> tuple[float, float, float]:
    if isinstance(result, cx.Point):
        result = result.data

    if isinstance(result, dict):
        x = float(cast("Any", u.ustrip("km", result["x"])))
        y = float(cast("Any", u.ustrip("km", result["y"])))
        z = float(cast("Any", u.ustrip("km", result["z"])))
        return (x, y, z)

    if isinstance(result, u.AbstractQuantity):
        arr = np.asarray(u.ustrip("km", result), dtype=float)
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    arr = np.asarray(jnp.asarray(result), dtype=float)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def test_reflect_from_normal_constructs_householder_matrix() -> None:
    """Normal-vector construction yields the expected Householder reflection."""
    op = cxfm.Reflect.from_normal([1, 0, 0])
    expected = jnp.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(op.H, expected, rtol=0, atol=1e-12)


def test_reflect_quantity_applies_hyperplane_reflection(reflect_op) -> None:
    """Reflect flips only the component along the chosen normal."""
    q = u.Q(jnp.asarray([1, 0, 0]), "km")
    result = cxfm.act(reflect_op, None, q)
    np.testing.assert_allclose(
        _extract_xyz(result), np.asarray(EXPECTED_REFLECT), rtol=0, atol=1e-12
    )


def test_reflect_vector_roundtrip_is_identity(reflect_op, vector_3d) -> None:
    """A reflection composed with its inverse returns the original point."""
    fwd = cxfm.act(reflect_op, None, vector_3d)
    back = cxfm.act(reflect_op.inverse, None, fwd)
    np.testing.assert_allclose(
        _extract_xyz(back), np.asarray(EXPECTED_IDENTITY), rtol=0, atol=1e-12
    )


def test_reflect_coordinate_preserves_coordinate_type(reflect_op, coord_3d) -> None:
    """Reflect acts on Points with frames and preserves the Point type."""
    result = cxfm.act(reflect_op, None, coord_3d)
    assert isinstance(result, cx.Point)
    np.testing.assert_allclose(
        _extract_xyz(result), np.asarray(EXPECTED_REFLECT), rtol=0, atol=1e-12
    )


def test_reflect_simplify_keeps_nontrivial_reflection(reflect_op) -> None:
    """A nontrivial reflection does not simplify away."""
    simplified = cxfm.simplify(reflect_op)
    assert isinstance(simplified, cxfm.Reflect)
