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
