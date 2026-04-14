"""Usage tests for ``coordinax.frames.act``.

These are eye-verifiable, human-readable scenarios that demonstrate
applying transforms (Identity, Rotate, Translate, Composed) to every
supported object level:

1. ArrayLike  (+usys)
2. Quantity
3. QuantityMatrix
4. CDict
5. Vector
6. Coordinate (with a concrete frame)
7. Coordinate (with a TransformedReferenceFrame)

Each test prints its inputs and outputs so a reviewer can visually
verify correctness without reading implementation details.
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.main as cx
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from coordinax.internal import QuantityMatrix

# ===================================================================
# Helpers


def _assert_close(a, b, *, atol: float = 1e-6):
    """Assert two arrays are element-wise close."""
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol)


# ===================================================================
# Fixtures


@pytest.fixture
def usys():
    """Fixture for unit system."""
    return u.unitsystem("m", "s", "kg", "rad")


@pytest.fixture
def rot90z():
    """90° rotation about the z-axis."""
    return cxfm.Rotate.from_euler("z", u.Q(90, "deg"))


@pytest.fixture
def shift_1_2_3():
    """Translate by (1, 2, 3) km."""
    return cxfm.Translate.from_([1, 2, 3], "km")


# ===================================================================
# Identity: just returns the input unchanged
# ===================================================================


class TestIdentityUsage:
    """Identity transform leaves every object unchanged."""

    def test_on_quantity(self):
        q = u.Q([1.0, 2.0, 3.0], "km")
        result = cxfm.act(cxfm.Identity(), None, q)
        assert result is q, "Identity should return the exact same object"

    def test_on_cdict(self):
        v = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
        result = cxfm.act(cxfm.Identity(), None, v)
        assert result is v

    def test_on_vector(self):
        vec = cx.Point.from_({"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")})
        result = cxfm.act(cxfm.Identity(), None, vec)
        assert result is vec


# ===================================================================
# Rotate: apply a rotation matrix
# ===================================================================


class TestRotateUsage:
    """Rotating (1, 0, 0) by 90° about z gives (0, 1, 0)."""

    def test_on_array(self, rot90z, usys):
        """Array (1,0,0) m → (0,1,0) m after 90° z-rotation.

        Must pass usys so the framework knows how to interpret the bare
        array as having metre units.
        """
        x = jnp.array([1.0, 0.0, 0.0])
        result = cxfm.act(rot90z, None, x, cxc.cart3d, cxr.point, usys=usys)
        _assert_close(result, [0.0, 1.0, 0.0])

    def test_on_quantity(self, rot90z):
        """Quantity [1,0,0] km → [0,1,0] km."""
        q = u.Q([1.0, 0.0, 0.0], "km")
        result = cxfm.act(rot90z, None, q)
        _assert_close(result.value, [0.0, 1.0, 0.0])
        assert result.unit == u.unit("km")

    def test_on_quantity_matrix(self, rot90z):
        """QuantityMatrix (1,0,0) km → (0,1,0) km."""
        qm = QuantityMatrix(
            jnp.array([1.0, 0.0, 0.0]),
            unit=(u.unit("km"), u.unit("km"), u.unit("km")),
        )
        result = cxfm.act(rot90z, None, qm)
        assert isinstance(result, QuantityMatrix)
        _assert_close(result.value, [0.0, 1.0, 0.0])

    def test_on_quantity_matrix_mixed_units(self, rot90z):
        """QuantityMatrix with km,m,m: converted to common unit internally."""
        qm = QuantityMatrix(
            jnp.array([1.0, 0.0, 0.0]),
            unit=(u.unit("km"), u.unit("m"), u.unit("m")),
        )
        result = cxfm.act(rot90z, None, qm)
        assert isinstance(result, QuantityMatrix)
        # Internal conversion normalizes to common unit (km); x→0, y→1km, z→0
        _assert_close(result.value, [0.0, 1.0, 0.0], atol=1e-12)

    def test_on_cdict(self, rot90z):
        """CDict {x:1, y:0, z:0} km → {x:0, y:1, z:0} km."""
        v = {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
        result = cxfm.act(rot90z, None, v, cxc.cart3d, cxr.point)
        _assert_close(result["y"].value, 1.0)
        _assert_close(result["x"].value, 0.0)

    def test_on_vector(self, rot90z):
        """Vector(x=1km) → Vector(y=1km) under 90° z-rotation."""
        vec = cx.Point.from_({"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")})
        result = cxfm.act(rot90z, None, vec)
        _assert_close(result.data["y"].value, 1.0)
        _assert_close(result.data["x"].value, 0.0)

    def test_on_coordinate(self, rot90z):
        """Point at (1,0,0) km in Alice frame → (0,1,0) km."""
        coord = cx.Point.from_(
            {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")},
            cxf.alice,
        )
        result = cxfm.act(rot90z, None, coord)
        _assert_close(result.data["y"].value, 1.0)
        _assert_close(result.data["x"].value, 0.0)


# ===================================================================
# Translate: add a displacement
# ===================================================================


class TestTranslateUsage:
    """Translate (0,0,0) by (1,2,3) km gives (1,2,3) km."""

    def test_on_array(self, shift_1_2_3, usys):
        """Array [0,0,0] + shift [1,2,3] km → [1000,2000,3000] (in metres)."""
        x = jnp.array([0.0, 0.0, 0.0])
        result = cxfm.act(shift_1_2_3, None, x, cxc.cart3d, cxr.point, usys=usys)
        _assert_close(result, [1000.0, 2000.0, 3000.0])

    def test_on_quantity(self, shift_1_2_3):
        """Quantity [0,0,0] km + shift → [1,2,3] km."""
        q = u.Q([0.0, 0.0, 0.0], "km")
        result = cxfm.act(shift_1_2_3, None, q)
        _assert_close(result.value, [1.0, 2.0, 3.0])

    def test_on_quantity_matrix(self, shift_1_2_3):
        """QuantityMatrix [0,0,0] km + shift → [1,2,3] km."""
        qm = QuantityMatrix(
            jnp.array([0.0, 0.0, 0.0]),
            unit=(u.unit("km"), u.unit("km"), u.unit("km")),
        )
        result = cxfm.act(shift_1_2_3, None, qm)
        assert isinstance(result, QuantityMatrix)
        _assert_close(result.value, [1.0, 2.0, 3.0])

    def test_on_cdict(self, shift_1_2_3):
        """CDict {x:0, y:0, z:0} km → {x:1, y:2, z:3} km."""
        v = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
        result = cxfm.act(shift_1_2_3, None, v, cxc.cart3d, cxr.point)
        _assert_close(result["x"].value, 1.0)
        _assert_close(result["y"].value, 2.0)
        _assert_close(result["z"].value, 3.0)

    def test_on_vector(self, shift_1_2_3):
        """Vector at origin + (1,2,3) km shift."""
        vec = cx.Point.from_({"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")})
        result = cxfm.act(shift_1_2_3, None, vec)
        _assert_close(result.data["x"].value, 1.0)
        _assert_close(result.data["y"].value, 2.0)
        _assert_close(result.data["z"].value, 3.0)

    def test_on_coordinate(self, shift_1_2_3):
        """Point at origin in Alice → translated to (1,2,3) km."""
        coord = cx.Point.from_(
            {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")},
            cxf.alice,
        )
        result = cxfm.act(shift_1_2_3, None, coord)
        _assert_close(result.data["x"].value, 1.0)
        _assert_close(result.data["y"].value, 2.0)
        _assert_close(result.data["z"].value, 3.0)


# ===================================================================
# Composed: chain transforms
# ===================================================================


class TestComposedUsage:
    """Composed(translate, rotate): first shift to (1,0,0), then rotate 90°z."""

    @pytest.fixture
    def pipe(self, shift_1_2_3, rot90z):
        """Translate by (1,2,3) km then rotate 90° about z."""
        return cxfm.Composed((shift_1_2_3, rot90z))

    def test_on_quantity(self, pipe):
        """[0,0,0] km → translate → [1,2,3] km → rotate 90°z → [-2,1,3] km."""
        q = u.Q([0.0, 0.0, 0.0], "km")
        result = cxfm.act(pipe, None, q)
        _assert_close(result.value, [-2.0, 1.0, 3.0])

    def test_on_quantity_matrix(self, pipe):
        """QuantityMatrix through composed pipeline."""
        qm = QuantityMatrix(
            jnp.array([0.0, 0.0, 0.0]),
            unit=(u.unit("km"), u.unit("km"), u.unit("km")),
        )
        result = cxfm.act(pipe, None, qm)
        assert isinstance(result, QuantityMatrix)
        _assert_close(result.value, [-2.0, 1.0, 3.0])

    def test_on_cdict(self, pipe):
        """CDict through composed pipeline."""
        v = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
        result = cxfm.act(pipe, None, v, cxc.cart3d, cxr.point)
        _assert_close(result["x"].value, -2.0)
        _assert_close(result["y"].value, 1.0)
        _assert_close(result["z"].value, 3.0)

    def test_on_vector(self, pipe):
        """Vector through composed pipeline."""
        vec = cx.Point.from_({"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")})
        result = cxfm.act(pipe, None, vec)
        _assert_close(result.data["x"].value, -2.0)
        _assert_close(result.data["y"].value, 1.0)
        _assert_close(result.data["z"].value, 3.0)

    def test_on_coordinate(self, pipe):
        """Point through composed pipeline."""
        coord = cx.Point.from_(
            {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")},
            cxf.alice,
        )
        result = cxfm.act(pipe, None, coord)
        _assert_close(result.data["x"].value, -2.0)
        _assert_close(result.data["y"].value, 1.0)
        _assert_close(result.data["z"].value, 3.0)


# ===================================================================
# Roundtrip: op then op.inverse recovers the original
# ===================================================================


class TestRoundtripUsage:
    """Apply a transform, then its inverse, and verify recovery."""

    def test_rotate_roundtrip(self, rot90z):
        """Rotate then inverse-rotate a Quantity recovers original."""
        q = u.Q([3.0, -1.0, 2.0], "km")
        fwd = cxfm.act(rot90z, None, q)
        back = cxfm.act(rot90z.inverse, None, fwd)
        _assert_close(back.value, q.value)

    def test_translate_roundtrip(self, shift_1_2_3):
        """Translate then inverse-translate recovers original."""
        q = u.Q([5.0, 7.0, -3.0], "km")
        fwd = cxfm.act(shift_1_2_3, None, q)
        back = cxfm.act(shift_1_2_3.inverse, None, fwd)
        _assert_close(back.value, q.value)

    def test_composed_roundtrip(self, rot90z, shift_1_2_3):
        """Composed pipeline and its inverse recover original."""
        pipe = cxfm.Composed((shift_1_2_3, rot90z))
        q = u.Q([2.0, 4.0, 6.0], "km")
        fwd = cxfm.act(pipe, None, q)
        back = cxfm.act(pipe.inverse, None, fwd)
        _assert_close(back.value, q.value)

    def test_roundtrip_on_vector(self, rot90z):
        """Rotate then inverse-rotate a Vector recovers original data."""
        vec = cx.Point.from_({"x": u.Q(3, "km"), "y": u.Q(-1, "km"), "z": u.Q(2, "km")})
        fwd = cxfm.act(rot90z, None, vec)
        back = cxfm.act(rot90z.inverse, None, fwd)
        _assert_close(back.data["x"].value, 3.0)
        _assert_close(back.data["y"].value, -1.0)
        _assert_close(back.data["z"].value, 2.0)

    def test_roundtrip_on_quantity_matrix(self, rot90z):
        """Rotate then inverse-rotate a QuantityMatrix recovers original."""
        qm = QuantityMatrix(
            jnp.array([3.0, -1.0, 2.0]),
            unit=(u.unit("km"), u.unit("km"), u.unit("km")),
        )
        fwd = cxfm.act(rot90z, None, qm)
        back = cxfm.act(rot90z.inverse, None, fwd)
        _assert_close(back.value, [3.0, -1.0, 2.0])


# ===================================================================
# TransformCallable: op(x) is syntactic sugar for act(op, None, x)
# ===================================================================


class TestCallSyntaxUsage:
    """Transforms are callable: op(x) == act(op, None, x)."""

    def test_rotate_call(self, rot90z):
        q = u.Q([1.0, 0.0, 0.0], "km")
        via_act = cxfm.act(rot90z, None, q)
        via_call = rot90z(q)
        _assert_close(via_call.value, via_act.value)

    def test_translate_call(self, shift_1_2_3):
        q = u.Q([0.0, 0.0, 0.0], "km")
        via_act = cxfm.act(shift_1_2_3, None, q)
        via_call = shift_1_2_3(q)
        _assert_close(via_call.value, via_act.value)
