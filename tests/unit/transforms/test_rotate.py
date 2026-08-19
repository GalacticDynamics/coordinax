"""Tests for ``coordinax.transforms.Rotate``."""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.transforms as cxfm

_RZ90 = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class TestRotationMatrixIsDimensionless:
    """`R`'s entries are ratios, so a dimensionful matrix is refused.

    Regression: the field is annotated `Shaped[Array, " N N"]` but had no
    converter, and `jnp` in that module is `quaxed.numpy`, whose `asarray`
    returns a `Quantity` unchanged. `Rotate(u.Q(eye, ""))` therefore stored a
    `Quantity` and failed later inside `jnp.linalg.det`, naming nothing the
    caller had written.
    """

    def test_dimensionless_quantity_is_stripped(self):
        op = cxfm.Rotate(u.Q(_RZ90, ""))
        assert not isinstance(op.R, u.AbstractQuantity)

    def test_agrees_with_the_bare_array(self):
        """Stripping must give the same operator, not merely a working one."""
        got = cxfm.Rotate(u.Q(_RZ90, "")).matrix
        ref = cxfm.Rotate(_RZ90).matrix
        assert bool(jnp.allclose(got, ref, atol=1e-14))

    @pytest.mark.parametrize("unit", ["m", "s", "rad"])
    def test_dimensionful_matrix_is_refused(self, unit):
        with pytest.raises(ValueError, match="dimensionless"):
            cxfm.Rotate(u.Q(_RZ90, unit))

    def test_bare_array_is_unaffected(self):
        """Positive control: the ordinary path still works."""
        assert not isinstance(cxfm.Rotate(_RZ90).R, u.AbstractQuantity)
