"""Tests for ``coordinax.transforms.Rotate``."""

__all__: tuple[str, ...] = ()

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.transforms as cxfm
from coordinax.transforms._src.actions.rotate import _not_orthogonal

_RZ90 = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class TestRotationMatrixIsDimensionless:
    """`R`'s entries are ratios, so a dimensionful matrix is refused.

    Regression: no converter, and quaxed's `asarray` returns a `Quantity`
    unchanged, so one was stored in an `Array` field and `matrix` failed later
    inside `jnp.linalg.det`.
    """

    @pytest.mark.parametrize(
        "R", [_RZ90, u.Q(_RZ90, "")], ids=["bare", "dimensionless"]
    )
    def test_r_is_stored_bare(self, R):
        """The converter runs despite `Rotate.__init__`.

        Equinox re-applies converters after ``__init__``, so that
        ``object.__setattr__`` does not bypass this one -- deleting it on that
        reading would restore the bug silently.
        """
        assert not isinstance(cxfm.Rotate(R).R, u.AbstractQuantity)

    def test_agrees_with_the_bare_array(self):
        """Stripping must give the same operator, not merely a working one."""
        got = cxfm.Rotate(u.Q(_RZ90, "")).matrix
        ref = cxfm.Rotate(_RZ90).matrix
        assert bool(jnp.allclose(got, ref, atol=1e-14))

    @pytest.mark.parametrize("unit", ["m", "s", "rad"])
    def test_dimensionful_matrix_is_refused(self, unit):
        with pytest.raises(ValueError, match="dimensionless"):
            cxfm.Rotate(u.Q(_RZ90, unit))


class TestRotationMatrixIsOrthogonal:
    """``R`` must satisfy ``R^T R = I``, the invariant the class relies on.

    Regression for #938: the docstring promised the check, nothing performed
    it, and `inverse` *transposes* -- the inverse only for an orthogonal
    matrix. ``Rotate([[1,2,3],[4,5,6],[7,8,10]])`` built fine and
    ``op.inverse(op(Point([1,2,3] kpc)))`` came back ``[513, 612, 764] kpc``.
    """

    _BAD = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])

    def test_non_orthogonal_is_refused(self):
        with pytest.raises(eqx.EquinoxRuntimeError, match="orthogonal"):
            cxfm.Rotate(self._BAD)

    def test_non_orthogonal_is_refused_under_jit(self):
        """Deferred `error_if`, so the guard traces instead of dying on a bool."""
        build = eqx.filter_jit(lambda m: cxfm.Rotate(m).R)
        with pytest.raises(eqx.EquinoxRuntimeError, match="orthogonal"):
            jax.block_until_ready(build(self._BAD))

    @pytest.mark.parametrize(
        "R",
        [_RZ90, jnp.eye(3), jnp.diag(jnp.asarray([-1.0, 1.0, 1.0]))],
        ids=["rz90", "identity", "improper"],
    )
    def test_orthogonal_still_constructs(self, R):
        assert bool(jnp.allclose(cxfm.Rotate(R).matrix, R))

    def test_orthogonal_still_constructs_under_jit(self):
        op = eqx.filter_jit(cxfm.Rotate)(_RZ90)
        assert bool(jnp.allclose(op.R, _RZ90))

    def test_a_non_square_matrix_is_named_by_the_shape_check(self):
        """The orthogonality check defers to `_validate_square` on shape.

        A non-square matrix has no ``R^T R`` to compare, so `_not_orthogonal`
        declines and the error the caller sees is the one that actually names
        the problem -- not a confusing "not orthogonal".
        """
        rect = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # The orthogonality predicate declines on a non-square matrix -- it has
        # no `R^T R` to compare -- so the shape check is what must catch it,
        # and it now does so in the constructor rather than at first use.
        assert _not_orthogonal(rect) is False
        with pytest.raises(
            eqx.EquinoxTracetimeError, match=r"square matrix; got shape"
        ):
            cxfm.Rotate(rect)

    def test_a_valid_rotation_round_trips(self):
        """What the bad matrix broke: `inverse` really does undo the map."""
        op = cxfm.Rotate(_RZ90)
        q = u.Q(jnp.asarray([1.0, 2.0, 3.0]), "kpc")
        back = op.inverse(None, op(None, q))
        assert bool(jnp.allclose(u.ustrip("kpc", back), jnp.asarray([1.0, 2.0, 3.0])))

    def test_a_numerically_drifted_frame_is_accepted(self):
        """Round-off must not be mistaken for a non-orthogonal matrix.

        `jnp.allclose`'s default ``atol=1e-8`` is the whole budget for the
        off-diagonal entries (they are compared against zero, where ``rtol``
        contributes nothing), and a parallel-transported Bishop triad
        (`coordinaxs.curveframes`) drifts further than that -- its own
        doctests assert orthogonality at ``1e-6``. This pins the tolerance
        from the other side, so tightening it back breaks here rather than in
        a downstream package.
        """
        drifted = _RZ90 + 2e-8 * jnp.asarray(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        )
        gram = drifted.T @ drifted
        assert not bool(jnp.allclose(gram, jnp.eye(3)))  # default atol refuses it
        assert bool(jnp.allclose(gram, jnp.eye(3), atol=1e-6))
        assert bool(jnp.allclose(cxfm.Rotate(drifted).matrix, drifted))

    def test_the_named_constructors_and_algebra_still_pass(self):
        """`from_euler`, `__matmul__` and `__neg__` all route through `__init__`."""
        a = cxfm.Rotate.from_euler("z", u.Q(45, "deg"))
        b = cxfm.Rotate.from_euler("x", u.Q(30, "deg"))
        for op in (a, b, a @ b, -a, a.inverse):
            gram = op.R.T @ op.R
            assert bool(jnp.allclose(gram, jnp.eye(3), atol=1e-12))
