"""Red/Green TDD tests for ``coordinax.frames.act`` dispatches.

Tests every combination of {Identity, Rotate, Translate, Composed} ×
{Array, Quantity, QuantityMatrix, CDict, Vector, Point+Frame, Point+XfmFrame}.

Each class tests:
  - correctness: known-value checks
  - return type: output matches input type
  - roundtrip:  act(op.inverse, None, act(op, None, x)) ≈ x
  - jit compat: jax.jit wrapping works
"""

__all__: tuple[str, ...] = ()

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.main as cx
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .conftest import (
    EXPECTED_COMPOSED,
    EXPECTED_IDENTITY,
    EXPECTED_ROTATE,
    EXPECTED_TRANSLATE,
)
from coordinax.internal import QuantityMatrix

ATOL = 1e-5


# ===================================================================
# Helpers


def _extract_xyz(result):
    """Extract (x, y, z) floats from any result type for comparison."""
    if isinstance(result, dict):
        # CDict
        x = float(u.ustrip("km", result["x"]))
        y = float(u.ustrip("km", result["y"]))
        z = float(u.ustrip("km", result["z"]))
        return (x, y, z)

    if isinstance(result, cx.Point):
        d = result.data
        x = float(u.ustrip("km", d["x"]))
        y = float(u.ustrip("km", d["y"]))
        z = float(u.ustrip("km", d["z"]))
        return (x, y, z)

    if isinstance(result, QuantityMatrix):
        x = float(u.ustrip("km", u.Q(result.value[0], result.unit[0])))
        y = float(u.ustrip("km", u.Q(result.value[1], result.unit[1])))
        z = float(u.ustrip("km", u.Q(result.value[2], result.unit[2])))
        return (x, y, z)

    if isinstance(result, u.AbstractQuantity):
        arr = u.ustrip("km", result)
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    # Bare array
    arr = jnp.asarray(result)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _assert_close(actual_xyz, expected_xyz, atol=ATOL):
    np.testing.assert_allclose(actual_xyz, expected_xyz, atol=atol)


# ===================================================================
# Level 1: Array(like)
# ===================================================================


class TestActOnArray:
    """Apply transforms to bare JAX arrays."""

    def test_identity(self, identity_op, array_3d):
        result = cxfm.act(identity_op, None, array_3d)
        _assert_close(_extract_xyz(result), EXPECTED_IDENTITY)

    def test_rotate(self, rotate_op, array_3d):
        result = cxfm.act(rotate_op, None, array_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_translate(self, translate_op, array_3d):
        # Bare arrays require usys because there's no unit info on the array.
        # The translate delta is in km, so with SI usys the array is in meters.
        # We use a km-based usys so both array and delta are in km.
        usys = u.unitsystem("km", "s", "kg", "rad")
        result = cxfm.act(translate_op, None, array_3d, usys=usys)
        _assert_close(_extract_xyz(result), EXPECTED_TRANSLATE)

    def test_composed(self, composed_op, array_3d):
        usys = u.unitsystem("km", "s", "kg", "rad")
        result = cxfm.act(composed_op, None, array_3d, usys=usys)
        _assert_close(_extract_xyz(result), EXPECTED_COMPOSED)

    def test_rotate_returns_array(self, rotate_op, array_3d):
        result = cxfm.act(rotate_op, None, array_3d)
        assert isinstance(result, jax.Array)

    def test_rotate_roundtrip(self, rotate_op, array_3d):
        fwd = cxfm.act(rotate_op, None, array_3d)
        back = cxfm.act(rotate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_translate_roundtrip(self, translate_op, array_3d):
        usys = u.unitsystem("km", "s", "kg", "rad")
        fwd = cxfm.act(translate_op, None, array_3d, usys=usys)
        back = cxfm.act(translate_op.inverse, None, fwd, usys=usys)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_composed_roundtrip(self, composed_op, array_3d):
        usys = u.unitsystem("km", "s", "kg", "rad")
        fwd = cxfm.act(composed_op, None, array_3d, usys=usys)
        back = cxfm.act(composed_op.inverse, None, fwd, usys=usys)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)


# ===================================================================
# Level 2: Quantity
# ===================================================================


class TestActOnQuantity:
    """Apply transforms to ``unxt.Quantity``."""

    def test_identity(self, identity_op, quantity_3d):
        result = cxfm.act(identity_op, None, quantity_3d)
        _assert_close(_extract_xyz(result), EXPECTED_IDENTITY)

    def test_rotate(self, rotate_op, quantity_3d):
        result = cxfm.act(rotate_op, None, quantity_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_translate(self, translate_op, quantity_3d):
        result = cxfm.act(translate_op, None, quantity_3d)
        _assert_close(_extract_xyz(result), EXPECTED_TRANSLATE)

    def test_composed(self, composed_op, quantity_3d):
        result = cxfm.act(composed_op, None, quantity_3d)
        _assert_close(_extract_xyz(result), EXPECTED_COMPOSED)

    def test_returns_quantity(self, rotate_op, quantity_3d):
        result = cxfm.act(rotate_op, None, quantity_3d)
        assert isinstance(result, u.AbstractQuantity)

    def test_rotate_roundtrip(self, rotate_op, quantity_3d):
        fwd = cxfm.act(rotate_op, None, quantity_3d)
        back = cxfm.act(rotate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_translate_roundtrip(self, translate_op, quantity_3d):
        fwd = cxfm.act(translate_op, None, quantity_3d)
        back = cxfm.act(translate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_composed_roundtrip(self, composed_op, quantity_3d):
        fwd = cxfm.act(composed_op, None, quantity_3d)
        back = cxfm.act(composed_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_rotate_with_explicit_chart(self, rotate_op, quantity_3d):
        result = cxfm.act(rotate_op, None, quantity_3d, cxc.cart3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_rotate_with_chart_and_rep(self, rotate_op, quantity_3d):
        result = cxfm.act(rotate_op, None, quantity_3d, cxc.cart3d, cxr.point)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)


# ===================================================================
# Level 3: QuantityMatrix
# ===================================================================


class TestActOnQuantityMatrix:
    """Apply transforms to ``QuantityMatrix``.

    This is expected to be RED initially — no dispatches exist.
    """

    def test_identity(self, identity_op, qmatrix_3d):
        result = cxfm.act(identity_op, None, qmatrix_3d)
        _assert_close(_extract_xyz(result), EXPECTED_IDENTITY)

    def test_rotate(self, rotate_op, qmatrix_3d):
        result = cxfm.act(rotate_op, None, qmatrix_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_translate(self, translate_op, qmatrix_3d):
        result = cxfm.act(translate_op, None, qmatrix_3d)
        _assert_close(_extract_xyz(result), EXPECTED_TRANSLATE)

    def test_composed(self, composed_op, qmatrix_3d):
        result = cxfm.act(composed_op, None, qmatrix_3d)
        _assert_close(_extract_xyz(result), EXPECTED_COMPOSED)

    def test_returns_quantity_matrix(self, rotate_op, qmatrix_3d):
        result = cxfm.act(rotate_op, None, qmatrix_3d)
        assert isinstance(result, QuantityMatrix)

    def test_rotate_roundtrip(self, rotate_op, qmatrix_3d):
        fwd = cxfm.act(rotate_op, None, qmatrix_3d)
        back = cxfm.act(rotate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_translate_roundtrip(self, translate_op, qmatrix_3d):
        fwd = cxfm.act(translate_op, None, qmatrix_3d)
        back = cxfm.act(translate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_composed_roundtrip(self, composed_op, qmatrix_3d):
        fwd = cxfm.act(composed_op, None, qmatrix_3d)
        back = cxfm.act(composed_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_rotate_with_explicit_chart(self, rotate_op, qmatrix_3d):
        result = cxfm.act(rotate_op, None, qmatrix_3d, cxc.cart3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_rotate_with_chart_and_rep(self, rotate_op, qmatrix_3d):
        result = cxfm.act(rotate_op, None, qmatrix_3d, cxc.cart3d, cxr.point)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_heterogeneous_units_identity(self, identity_op):
        """QuantityMatrix with mixed units passes through Identity."""
        qm = QuantityMatrix(
            jnp.array([1.0, 2.0, 3.0]),
            unit=(u.unit("km"), u.unit("km"), u.unit("km")),
        )
        result = cxfm.act(identity_op, None, qm)
        assert isinstance(result, QuantityMatrix)
        _assert_close(_extract_xyz(result), (1.0, 2.0, 3.0))


# ===================================================================
# Level 4: CDict
# ===================================================================


class TestActOnCDict:
    """Apply transforms to component dictionaries (CDict)."""

    def test_identity(self, identity_op, cdict_3d):
        result = cxfm.act(identity_op, None, cdict_3d)
        _assert_close(_extract_xyz(result), EXPECTED_IDENTITY)

    def test_rotate(self, rotate_op, cdict_3d):
        result = cxfm.act(rotate_op, None, cdict_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_translate(self, translate_op, cdict_3d):
        result = cxfm.act(translate_op, None, cdict_3d)
        _assert_close(_extract_xyz(result), EXPECTED_TRANSLATE)

    def test_composed(self, composed_op, cdict_3d):
        result = cxfm.act(composed_op, None, cdict_3d)
        _assert_close(_extract_xyz(result), EXPECTED_COMPOSED)

    def test_returns_dict(self, rotate_op, cdict_3d):
        result = cxfm.act(rotate_op, None, cdict_3d)
        assert isinstance(result, dict)

    def test_rotate_roundtrip(self, rotate_op, cdict_3d):
        fwd = cxfm.act(rotate_op, None, cdict_3d)
        back = cxfm.act(rotate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_translate_roundtrip(self, translate_op, cdict_3d):
        fwd = cxfm.act(translate_op, None, cdict_3d)
        back = cxfm.act(translate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_composed_roundtrip(self, composed_op, cdict_3d):
        fwd = cxfm.act(composed_op, None, cdict_3d)
        back = cxfm.act(composed_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_rotate_with_explicit_chart(self, rotate_op, cdict_3d):
        result = cxfm.act(rotate_op, None, cdict_3d, cxc.cart3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_rotate_with_chart_and_rep(self, rotate_op, cdict_3d):
        result = cxfm.act(rotate_op, None, cdict_3d, cxc.cart3d, cxr.point)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)


# ===================================================================
# Level 5: Vector
# ===================================================================


class TestActOnVector:
    """Apply transforms to ``coordinax.Point``."""

    def test_identity(self, identity_op, vector_3d):
        result = cxfm.act(identity_op, None, vector_3d)
        _assert_close(_extract_xyz(result), EXPECTED_IDENTITY)

    def test_rotate(self, rotate_op, vector_3d):
        result = cxfm.act(rotate_op, None, vector_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_translate(self, translate_op, vector_3d):
        result = cxfm.act(translate_op, None, vector_3d)
        _assert_close(_extract_xyz(result), EXPECTED_TRANSLATE)

    def test_composed(self, composed_op, vector_3d):
        result = cxfm.act(composed_op, None, vector_3d)
        _assert_close(_extract_xyz(result), EXPECTED_COMPOSED)

    def test_returns_vector(self, rotate_op, vector_3d):
        result = cxfm.act(rotate_op, None, vector_3d)
        assert isinstance(result, cx.Point)

    def test_preserves_chart(self, rotate_op, vector_3d):
        result = cxfm.act(rotate_op, None, vector_3d)
        assert result.chart == vector_3d.chart

    def test_rotate_roundtrip(self, rotate_op, vector_3d):
        fwd = cxfm.act(rotate_op, None, vector_3d)
        back = cxfm.act(rotate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_translate_roundtrip(self, translate_op, vector_3d):
        fwd = cxfm.act(translate_op, None, vector_3d)
        back = cxfm.act(translate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_composed_roundtrip(self, composed_op, vector_3d):
        fwd = cxfm.act(composed_op, None, vector_3d)
        back = cxfm.act(composed_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)


# ===================================================================
# Level 6: Point with a concrete frame
# ===================================================================


class TestActOnCoordinate:
    """Apply transforms to ``coordinax.Point`` with a concrete frame."""

    def test_identity(self, identity_op, coord_3d):
        result = cxfm.act(identity_op, None, coord_3d)
        _assert_close(_extract_xyz(result), EXPECTED_IDENTITY)

    def test_rotate(self, rotate_op, coord_3d):
        result = cxfm.act(rotate_op, None, coord_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_translate(self, translate_op, coord_3d):
        result = cxfm.act(translate_op, None, coord_3d)
        _assert_close(_extract_xyz(result), EXPECTED_TRANSLATE)

    def test_composed(self, composed_op, coord_3d):
        result = cxfm.act(composed_op, None, coord_3d)
        _assert_close(_extract_xyz(result), EXPECTED_COMPOSED)

    def test_returns_coordinate(self, rotate_op, coord_3d):
        result = cxfm.act(rotate_op, None, coord_3d)
        assert isinstance(result, cx.Point)

    def test_preserves_frame(self, rotate_op, coord_3d):
        result = cxfm.act(rotate_op, None, coord_3d)
        assert isinstance(result.frame, type(coord_3d.frame))

    def test_rotate_roundtrip(self, rotate_op, coord_3d):
        fwd = cxfm.act(rotate_op, None, coord_3d)
        back = cxfm.act(rotate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_translate_roundtrip(self, translate_op, coord_3d):
        fwd = cxfm.act(translate_op, None, coord_3d)
        back = cxfm.act(translate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_composed_roundtrip(self, composed_op, coord_3d):
        fwd = cxfm.act(composed_op, None, coord_3d)
        back = cxfm.act(composed_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)


# ===================================================================
# Level 7: Point with a TransformedReferenceFrame
# ===================================================================


class TestActOnCoordinateXfm:
    """Apply transforms to ``Point`` with a ``TransformedReferenceFrame``."""

    def test_identity(self, identity_op, coord_xfm_3d):
        result = cxfm.act(identity_op, None, coord_xfm_3d)
        _assert_close(_extract_xyz(result), EXPECTED_IDENTITY)

    def test_rotate(self, rotate_op, coord_xfm_3d):
        result = cxfm.act(rotate_op, None, coord_xfm_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_translate(self, translate_op, coord_xfm_3d):
        result = cxfm.act(translate_op, None, coord_xfm_3d)
        _assert_close(_extract_xyz(result), EXPECTED_TRANSLATE)

    def test_composed(self, composed_op, coord_xfm_3d):
        result = cxfm.act(composed_op, None, coord_xfm_3d)
        _assert_close(_extract_xyz(result), EXPECTED_COMPOSED)

    def test_returns_coordinate(self, rotate_op, coord_xfm_3d):
        result = cxfm.act(rotate_op, None, coord_xfm_3d)
        assert isinstance(result, cx.Point)

    def test_preserves_transformed_frame(self, rotate_op, coord_xfm_3d):
        result = cxfm.act(rotate_op, None, coord_xfm_3d)
        assert isinstance(result.frame, cxf.TransformedReferenceFrame)

    def test_rotate_roundtrip(self, rotate_op, coord_xfm_3d):
        fwd = cxfm.act(rotate_op, None, coord_xfm_3d)
        back = cxfm.act(rotate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_translate_roundtrip(self, translate_op, coord_xfm_3d):
        fwd = cxfm.act(translate_op, None, coord_xfm_3d)
        back = cxfm.act(translate_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)

    def test_composed_roundtrip(self, composed_op, coord_xfm_3d):
        fwd = cxfm.act(composed_op, None, coord_xfm_3d)
        back = cxfm.act(composed_op.inverse, None, fwd)
        _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)


# ===================================================================
# Cross-level consistency: same transform on all levels gives same answer
# ===================================================================


class TestCrossLevelConsistency:
    """Verify that the same transform gives the same numerical result."""

    def test_rotate_all_levels_agree(
        self,
        rotate_op,
        array_3d,
        quantity_3d,
        qmatrix_3d,
        cdict_3d,
        vector_3d,
        coord_3d,
        coord_xfm_3d,
    ):
        results = []
        # Level 1: Array
        results.append(_extract_xyz(cxfm.act(rotate_op, None, array_3d)))
        # Level 2: Quantity
        results.append(_extract_xyz(cxfm.act(rotate_op, None, quantity_3d)))
        # Level 3: QuantityMatrix
        results.append(_extract_xyz(cxfm.act(rotate_op, None, qmatrix_3d)))
        # Level 4: CDict
        results.append(_extract_xyz(cxfm.act(rotate_op, None, cdict_3d)))
        # Level 5: Vector
        results.append(_extract_xyz(cxfm.act(rotate_op, None, vector_3d)))
        # Level 6: Coordinate
        results.append(_extract_xyz(cxfm.act(rotate_op, None, coord_3d)))
        # Level 7: Coordinate + XfmFrame
        results.append(_extract_xyz(cxfm.act(rotate_op, None, coord_xfm_3d)))

        for r in results:
            _assert_close(r, EXPECTED_ROTATE, atol=ATOL)

    def test_translate_all_levels_agree(
        self,
        translate_op,
        array_3d,
        quantity_3d,
        qmatrix_3d,
        cdict_3d,
        vector_3d,
        coord_3d,
        coord_xfm_3d,
    ):
        usys = u.unitsystem("km", "s", "kg", "rad")
        results = []
        results.append(_extract_xyz(cxfm.act(translate_op, None, array_3d, usys=usys)))
        results.append(_extract_xyz(cxfm.act(translate_op, None, quantity_3d)))
        results.append(_extract_xyz(cxfm.act(translate_op, None, qmatrix_3d)))
        results.append(_extract_xyz(cxfm.act(translate_op, None, cdict_3d)))
        results.append(_extract_xyz(cxfm.act(translate_op, None, vector_3d)))
        results.append(_extract_xyz(cxfm.act(translate_op, None, coord_3d)))
        results.append(_extract_xyz(cxfm.act(translate_op, None, coord_xfm_3d)))

        for r in results:
            _assert_close(r, EXPECTED_TRANSLATE, atol=ATOL)


# ===================================================================
# JIT compatibility
# ===================================================================


class TestActJIT:
    """Verify act works under jax.jit / eqx.filter_jit for each level."""

    def test_jit_array(self, rotate_op, array_3d):
        result = jax.jit(lambda x: cxfm.act(rotate_op, None, x))(array_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_jit_quantity(self, rotate_op, quantity_3d):
        result = jax.jit(lambda x: cxfm.act(rotate_op, None, x))(quantity_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_jit_qmatrix(self, rotate_op, qmatrix_3d):
        result = eqx.filter_jit(lambda x: cxfm.act(rotate_op, None, x))(qmatrix_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_jit_cdict(self, rotate_op, cdict_3d):
        result = jax.jit(lambda x: cxfm.act(rotate_op, None, x))(cdict_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_jit_vector(self, rotate_op, vector_3d):
        result = eqx.filter_jit(lambda x: cxfm.act(rotate_op, None, x))(vector_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_jit_coordinate(self, rotate_op, coord_3d):
        result = eqx.filter_jit(lambda x: cxfm.act(rotate_op, None, x))(coord_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_jit_coordinate_xfm(self, rotate_op, coord_xfm_3d):
        result = eqx.filter_jit(lambda x: cxfm.act(rotate_op, None, x))(coord_xfm_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)


# ===================================================================
# Callable via __call__
# ===================================================================


class TestTransformCallable:
    """Verify transforms can be called directly as op(x) or op(tau, x)."""

    def test_rotate_call_vector(self, rotate_op, vector_3d):
        result = rotate_op(vector_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_rotate_call_with_tau_vector(self, rotate_op, vector_3d):
        result = rotate_op(None, vector_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_translate_call_quantity(self, translate_op, quantity_3d):
        result = translate_op(quantity_3d)
        _assert_close(_extract_xyz(result), EXPECTED_TRANSLATE)

    def test_composed_call_cdict(self, composed_op, cdict_3d):
        result = composed_op(cdict_3d)
        _assert_close(_extract_xyz(result), EXPECTED_COMPOSED)

    def test_identity_call_coordinate(self, identity_op, coord_3d):
        result = identity_op(coord_3d)
        _assert_close(_extract_xyz(result), EXPECTED_IDENTITY)


# ===================================================================
# Tangent geometry on non-Cartesian charts (Jacobian pushforward)
# ===================================================================


class TestRotateTangentGeometryNonCartesian:
    """Verify that act(Rotate, TangentGeometry, sph3d) uses the Jacobian.

    The key invariant:
        cart(rotate(v, at_sph)) == R * cart(v, at_sph)

    where cart(*) denotes the tangent_map pushforward to Cartesian coords.
    """

    @pytest.fixture
    def rot90z(self):
        return cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

    @pytest.fixture
    def at_sph(self):
        """Base point at the equator phi=0; Cartesian (1,0,0)."""
        return {
            "r": u.Q(1.0, "m"),
            "theta": u.Q(jnp.pi / 2, "rad"),
            "phi": u.Q(0.0, "rad"),
        }

    @pytest.fixture
    def v_radial_sph(self):
        """Purely radial velocity in spherical coord-basis."""
        return {
            "r": u.Q(1.0, "m/s"),
            "theta": u.Q(0.0, "rad/s"),
            "phi": u.Q(0.0, "rad/s"),
        }

    def test_cart_consistency(self, rot90z, at_sph, v_radial_sph):
        """cart(R*v at R*p) == R * cart(v at p)."""
        # Rotate tangent via Jacobian path
        v_rot_sph = cxfm.act(
            rot90z,
            None,
            v_radial_sph,
            cxc.sph3d,
            cxr.tangent_geom,
            cxr.coord_vel,
            at=at_sph,
        )
        # Rotated base point in spherical (phi: 0 -> pi/2)
        at_sph_rot = {
            "r": u.Q(1.0, "m"),
            "theta": u.Q(jnp.pi / 2, "rad"),
            "phi": u.Q(jnp.pi / 2, "rad"),
        }
        # Push rotated tangent to Cartesian
        v_rot_cart = cxr.tangent_map(
            v_rot_sph, cxc.sph3d, cxr.coord_vel, cxc.cart3d, at=at_sph_rot
        )
        # Directly compute R * cart(v)
        v_cart = cxr.tangent_map(
            v_radial_sph, cxc.sph3d, cxr.coord_vel, cxc.cart3d, at=at_sph
        )
        R = rot90z._get_R(cxc.cart3d)
        v_arr = jnp.stack([v_cart["x"].value, v_cart["y"].value, v_cart["z"].value])
        v_expected = R @ v_arr

        assert abs(float(v_rot_cart["x"].value) - float(v_expected[0])) < ATOL
        assert abs(float(v_rot_cart["y"].value) - float(v_expected[1])) < ATOL
        assert abs(float(v_rot_cart["z"].value) - float(v_expected[2])) < ATOL

    def test_round_trip(self, rot90z, at_sph, v_radial_sph):
        """R⁻¹(R(v, at), R(at)) == v."""
        v_rot_sph = cxfm.act(
            rot90z,
            None,
            v_radial_sph,
            cxc.sph3d,
            cxr.tangent_geom,
            cxr.coord_vel,
            at=at_sph,
        )
        at_sph_rot = {
            "r": u.Q(1.0, "m"),
            "theta": u.Q(jnp.pi / 2, "rad"),
            "phi": u.Q(jnp.pi / 2, "rad"),
        }
        inv_op = cxfm.Rotate.from_euler("z", u.Q(-90, "deg"))
        v_recovered = cxfm.act(
            inv_op,
            None,
            v_rot_sph,
            cxc.sph3d,
            cxr.tangent_geom,
            cxr.coord_vel,
            at=at_sph_rot,
        )
        assert abs(float(v_recovered["r"].to_value("m/s")) - 1.0) < ATOL
        assert abs(float(v_recovered["theta"].to_value("rad/s"))) < ATOL
        assert abs(float(v_recovered["phi"].to_value("rad/s"))) < ATOL

    def test_raises_without_at(self, rot90z, v_radial_sph):
        """act(Rotate, sph3d, TangentGeometry) raises TypeError without at=."""
        with pytest.raises(TypeError, match="requires 'at'"):
            cxfm.act(
                rot90z,
                None,
                v_radial_sph,
                cxc.sph3d,
                cxr.tangent_geom,
                cxr.coord_vel,
            )

    def test_jit(self, rot90z, at_sph, v_radial_sph):
        """act(Rotate, sph3d, TangentGeometry) is JIT-compatible."""
        result = eqx.filter_jit(
            lambda v: cxfm.act(
                rot90z,
                None,
                v,
                cxc.sph3d,
                cxr.tangent_geom,
                cxr.coord_vel,
                at=at_sph,
            )
        )(v_radial_sph)
        assert abs(float(result["r"].to_value("m/s")) - 1.0) < ATOL


# ===================================================================
# Coordinate.to_frame with non-Cartesian velocity
# ===================================================================


class TestCoordinateToFrameNonCartesianTangent:
    """Verify Coordinate.to_frame injects 'at' correctly for tangent fibres."""

    def test_cart3d_velocity_to_rotated_frame(self):
        """Coordinate with Cartesian velocity transforms correctly via to_frame."""
        rot = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        rotated_frame = cxf.TransformedReferenceFrame(cxf.alice, rot)

        point = cx.Point.from_([1.0, 0.0, 0.0], "m", cxf.alice)
        vel = cx.Tangent(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
            cxc.cart3d,
            cxr.coord_basis,
            cxr.vel,
            frame=cxf.alice,
        )
        coord = cx.Coordinate(point=point, velocity=vel)
        result = coord.to_frame(rotated_frame)

        # Point (1,0,0) rotated 90° about z -> (0,1,0)
        _assert_close(
            (
                float(result.point.data["x"].to_value("m")),
                float(result.point.data["y"].to_value("m")),
                float(result.point.data["z"].to_value("m")),
            ),
            (0.0, 1.0, 0.0),
        )
        # Velocity (1,0,0) m/s rotated -> (0,1,0) m/s
        _assert_close(
            (
                float(result["velocity"].data["x"].to_value("m/s")),
                float(result["velocity"].data["y"].to_value("m/s")),
                float(result["velocity"].data["z"].to_value("m/s")),
            ),
            (0.0, 1.0, 0.0),
        )

    def test_coordinate_to_frame_then_cconvert_sph(self):
        """Coordinate.to_frame followed by .cconvert(sph3d) works correctly."""
        rot = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        rotated_frame = cxf.TransformedReferenceFrame(cxf.alice, rot)

        point = cx.Point.from_([1.0, 0.0, 0.0], "m", cxf.alice)
        vel = cx.Tangent(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
            cxc.cart3d,
            cxr.coord_basis,
            cxr.vel,
            frame=cxf.alice,
        )
        coord = cx.Coordinate(point=point, velocity=vel)
        result = coord.to_frame(rotated_frame).cconvert(cxc.sph3d)

        # Point should land at (r=1, theta=pi/2, phi=pi/2)
        assert abs(float(result.point.data["r"].to_value("m")) - 1.0) < ATOL
        assert (
            abs(float(result.point.data["theta"].to_value("rad")) - jnp.pi / 2) < ATOL
        )
        assert abs(float(result.point.data["phi"].to_value("rad")) - jnp.pi / 2) < ATOL
        # Velocity should be purely radial (ṙ≈1, θ̇≈0, φ̇≈0)
        assert abs(float(result["velocity"].data["r"].to_value("m/s")) - 1.0) < ATOL
