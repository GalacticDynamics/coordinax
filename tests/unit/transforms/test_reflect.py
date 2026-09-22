"""Tests for the Reflect frame transform."""

__all__: tuple[str, ...] = ()

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax as cx
import coordinax.transforms as cxfm
from .conftest import EXPECTED_IDENTITY, EXPECTED_REFLECT


@pytest.mark.parametrize("bad", [0.0, jnp.nan, jnp.inf], ids=["zero", "nan", "inf"])
def test_reflect_from_normal_unnormalisable_raises_under_jit(bad: float) -> None:
    """A normal that cannot be normalised is rejected, even under jit."""
    build = eqx.filter_jit(cxfm.Reflect.from_normal)
    with pytest.raises(eqx.EquinoxRuntimeError, match="nonzero normal"):
        jax.block_until_ready(build(jnp.asarray([bad, 0.0, 0.0])).H)


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


class TestReflectionMatrixIsInvolutive:
    """``H`` must satisfy ``H @ H = I``, which is what makes `inverse` `self`.

    Regression for #939: `from_normal` was guarded, the bare constructor was
    not. A permutation matrix -- orthogonal, ``det = +1``, and *not* an
    involution -- gave ``op.inverse(op(p)) == [3, 1, 2]`` for ``p = [1, 2, 3]``.
    Orthogonality is too weak a check here; involutivity is the exact property.
    """

    _PERM = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    def test_the_permutation_matrix_is_orthogonal_and_still_refused(self) -> None:
        assert bool(jnp.allclose(self._PERM.T @ self._PERM, jnp.eye(3)))
        assert float(jnp.linalg.det(self._PERM)) == pytest.approx(1.0)
        with pytest.raises(eqx.EquinoxRuntimeError, match="H @ H = I"):
            cxfm.Reflect(self._PERM)

    def test_non_involutive_is_refused_under_jit(self) -> None:
        """Deferred `error_if`, so the guard traces instead of dying on a bool."""
        build = eqx.filter_jit(lambda m: cxfm.Reflect(m).H)
        with pytest.raises(eqx.EquinoxRuntimeError, match="H @ H = I"):
            jax.block_until_ready(build(self._PERM))

    @pytest.mark.parametrize(
        "H",
        [
            jnp.eye(3),
            jnp.diag(jnp.asarray([-1.0, 1.0, 1.0])),
            jnp.diag(jnp.asarray([-1.0, -1.0, 1.0])),
        ],
        ids=["identity", "one-axis", "two-axes"],
    )
    def test_involutions_still_construct(self, H) -> None:
        assert bool(jnp.allclose(cxfm.Reflect(H).matrix, H))

    def test_from_normal_still_passes_its_own_output(self) -> None:
        """A Householder matrix is an involution, so the guard is transparent."""
        op = cxfm.Reflect.from_normal([1.0, 2.0, 3.0])
        assert bool(jnp.allclose(op.H @ op.H, jnp.eye(3), atol=1e-12))

    def test_a_valid_reflection_round_trips(self) -> None:
        """What the permutation matrix broke: ``inverse`` really does undo it."""
        op = cxfm.Reflect.from_normal([0.0, 1.0, 0.0])
        q = u.Q(jnp.asarray([1.0, 2.0, 3.0]), "km")
        back = cxfm.act(op.inverse, None, cxfm.act(op, None, q))
        np.testing.assert_allclose(
            np.asarray(u.ustrip("km", cast("Any", back))), [1.0, 2.0, 3.0], atol=1e-12
        )

    def test_a_valid_reflection_constructs_under_jit(self) -> None:
        op = eqx.filter_jit(cxfm.Reflect)(jnp.diag(jnp.asarray([-1.0, 1.0, 1.0])))
        assert bool(jnp.allclose(op.H, jnp.diag(jnp.asarray([-1.0, 1.0, 1.0]))))
