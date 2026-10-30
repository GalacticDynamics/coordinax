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
from coordinax.transforms._src.actions.reflect import _not_a_reflection


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


def test_a_near_reflection_is_not_admitted_by_a_relative_tolerance() -> None:
    """The budget is ``1e-6`` absolute, and must not scale with ``n``.

    ``trace H`` is compared against ``n - 2``, so `jnp.isclose`'s default
    ``rtol=1e-5`` made the effective tolerance grow with the dimension --
    ``8.1e-5`` at ``n = 10``, ``9.8e-4`` at ``n = 100``. A matrix that is not
    a reflection then passes every clause.

    Perturbing one diagonal entry of a Householder matrix by ``5e-6`` is
    comfortably inside that dimension-scaled budget and comfortably outside
    the stated one.
    """
    n = 10
    v = np.zeros(n)
    v[0] = 1.0
    H = np.eye(n) - 2 * np.outer(v, v)
    H[1, 1] -= 5e-6
    H = jnp.asarray(H)

    assert abs(float(jnp.trace(H)) - (n - 2)) == pytest.approx(5e-6, rel=1e-3)
    # Inside `atol + rtol*(n-2)` = 8.1e-6 at n=10, outside the stated 1e-6.
    assert bool(jnp.isclose(jnp.trace(H), n - 2, atol=1e-6))
    assert not bool(jnp.isclose(jnp.trace(H), n - 2, atol=1e-6, rtol=0.0))

    assert _not_a_reflection(H)
    with pytest.raises(eqx.EquinoxRuntimeError):
        cxfm.Reflect(H)


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
        with pytest.raises(eqx.EquinoxRuntimeError, match="hyperplane reflection"):
            cxfm.Reflect(self._PERM)

    def test_non_involutive_is_refused_under_jit(self) -> None:
        """Deferred `error_if`, so the guard traces instead of dying on a bool."""
        build = eqx.filter_jit(lambda m: cxfm.Reflect(m).H)
        with pytest.raises(eqx.EquinoxRuntimeError, match="hyperplane reflection"):
            jax.block_until_ready(build(self._PERM))

    def test_a_single_reflected_axis_constructs(self) -> None:
        """One -1 eigenvalue is a hyperplane reflection, so it is accepted."""
        H = jnp.diag(jnp.asarray([-1.0, 1.0, 1.0]))
        assert bool(jnp.allclose(cxfm.Reflect(H).matrix, H))

    @pytest.mark.parametrize(
        "H",
        [
            jnp.eye(3),
            jnp.diag(jnp.asarray([-1.0, -1.0, 1.0])),
            -jnp.eye(3),
            jnp.asarray([[1.0, 1.0], [0.0, -1.0]]),
        ],
        ids=["identity", "pi-rotation", "point-inversion", "not-symmetric"],
    )
    def test_involutions_that_are_not_reflections_are_refused(self, H) -> None:
        """An involution is necessary but not sufficient.

        `Reflect` denotes *exactly* a hyperplane reflection (spec 5577), so the
        guard is symmetric + involutive + ``trace H == n - 2``. Each case here
        is an involution that fails one of those:

        - ``identity`` and ``pi-rotation`` have ``det = +1`` and are in SO(n);
          the second is itself a composition of two reflections, which the
          spec says has no closed `Reflect @ Reflect`.
        - ``point-inversion`` has ``det = -1`` -- so a determinant check alone
          would admit it -- but reflects every direction, not one.
        - ``not-symmetric`` satisfies involutivity, ``det = -1`` *and*
          ``trace == n - 2``, and is still neither symmetric nor orthogonal.
        """
        with pytest.raises(eqx.EquinoxRuntimeError, match="hyperplane reflection"):
            cxfm.Reflect(H)

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


def test_a_non_square_matrix_is_named_by_the_shape_check() -> None:
    """The involutivity check defers to `_validate_square` on shape.

    Mirrors `test_rotate.py`. A non-square matrix has no ``H @ H`` to compare,
    so `_not_a_reflection` declines and the error names the shape rather than
    claiming the matrix is not an involution.
    """
    rect = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert _not_a_reflection(rect) is False
    # `RuntimeError`: see the note in `test_checks.py` (gh#994).
    with pytest.raises(RuntimeError, match=r"square matrix; got shape"):
        cxfm.Reflect(rect)
