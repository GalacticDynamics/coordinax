"""The Frenet--Serret reductions run per row, on the last axis.

The norm and the Gram--Schmidt dot product are both per-vector, so they must
hold for ``(3,)`` and ``(..., 3)`` alike; both are asserted here. The ``(3,)``
cases also pin the contract `signedplanar` relies on, which shares
`_normalize`.
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc
from coordinaxs.curveframes._src.frenetserret import _normalize


class TestNormalizeIsRowWise:
    """Each row gets its own norm."""

    def test_a_single_vector_is_unchanged(self):
        """The ``(3,)`` contract every current caller relies on."""
        np.testing.assert_allclose(
            _normalize(jnp.array([3.0, 4.0, 0.0])), [0.6, 0.8, 0.0], atol=1e-15
        )

    def test_a_stack_normalises_per_row(self):
        """Each row is unit-length, not merely a direction."""
        got = _normalize(jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]))
        np.testing.assert_allclose(got, [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], atol=1e-15)
        np.testing.assert_allclose(jnp.linalg.norm(got, axis=-1), 1.0, atol=1e-15)

    def test_it_keeps_units(self):
        """A `unxt.Quantity` normalises to dimensionless, stacked or not."""
        got = _normalize(u.Q([[0.0, 0.0, 5.0], [0.0, 7.0, 0.0]], "m/s"))
        assert got.unit == u.unit("")
        np.testing.assert_allclose(
            got.value, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], atol=1e-15
        )


class TestTheGramSchmidtProjectionIsRowWise:
    """The normal of a stacked curve is one normal per point.

    Only the ``N`` row is asserted: ``T`` comes from `base.unit_tangent`, whose
    reduction is still global, so a stacked tangent row stays mis-scaled and
    the binormal inherits that.
    """

    def test_each_point_gets_its_own_unit_normal(self):
        # Two concentric circles evaluated at one tau, stacked into (2, 3).
        # The Frenet normal of a circle points at its centre, so both rows are
        # -(cos t, sin t, 0) -- the same direction, but each of unit length,
        # which is exactly what a global norm cannot produce from radii 1 and 2.
        def two_circles(tau: u.AbstractQuantity) -> u.AbstractQuantity:
            t = tau.ustrip("s")
            zero = jnp.zeros_like(t)
            row = jnp.stack([jnp.cos(t), jnp.sin(t), zero])
            return u.Q(jnp.stack([row, 2 * row]), "km")

        fs = cxfc.FrenetSerretBuilder(two_circles, "s")
        t = 0.3
        n_rows = fs.rotation_matrix(u.Q(t, "s"))[1]

        expected = [-jnp.cos(t), -jnp.sin(t), 0.0]
        np.testing.assert_allclose(n_rows, [expected, expected], atol=1e-12)
        np.testing.assert_allclose(jnp.linalg.norm(n_rows, axis=-1), 1.0, atol=1e-12)


class TestTheScalarPathIsUnchanged:
    """``keepdims`` must not disturb the ``(3,)`` contract.

    Closed form on the unit circle at tau=0: ``T = (0,1,0)``, ``N = (-1,0,0)``,
    ``B = (0,0,1)``.
    """

    def test_closed_form_on_the_unit_circle(self):
        def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
            t = tau.ustrip("s")
            return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

        R = cxfc.FrenetSerretBuilder(circle, "s").rotation_matrix(u.Q(0.0, "s"))
        np.testing.assert_allclose(
            R, [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-12
        )

    def test_the_zero_curvature_guard_still_fires(self):
        """The rejection ratio is a reduction too, and it feeds an `error_if`."""

        def line(tau: u.AbstractQuantity) -> u.AbstractQuantity:
            t = tau.ustrip("s")
            return u.Q(jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")

        with pytest.raises(Exception, match="curvature"):
            cxfc.FrenetSerretBuilder(line, "s").rotation_matrix(u.Q(1.0, "s"))
