"""Tests for ``coordinax.transforms.Rotate``."""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.transforms as cxfm

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
