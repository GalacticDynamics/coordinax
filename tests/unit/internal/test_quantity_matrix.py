"""Tests for coordinax._src.quantity_matrix.QuantityMatrix."""

import math

import astropy.units as apu
import jax
import jax.numpy as jnp
import plum
import pytest
import quax
from astropy.units import imperial  # registers °F

import unxt as u

from coordinax.internal import QuantityMatrix as QMat, UnitsMatrix
from coordinax.internal._quantity_matrix import (
    _convert_value_matrix,
    _convert_value_vector,
)

# ---------------------------------------------------------------------------
# Unit shorthands (visual noise reduction)
# ---------------------------------------------------------------------------

_m = u.unit("m")
_s = u.unit("s")
_kg = u.unit("kg")
_rad = u.unit("rad")
_km = u.unit("km")
_ms = u.unit("ms")
_g = u.unit("g")
_deg = u.unit("deg")
_min = u.unit("min")
_dimless = u.unit("")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_2x2():
    """Return a simple 2x2 unit grid: m, s, kg, rad."""
    return ((_m, _s), (_kg, _rad))


@pytest.fixture
def qm_2x2(unit_2x2):
    """Return a 2x2 QuantityMatrix with values 1-4."""
    return QMat(
        value=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        unit=unit_2x2,
    )


@pytest.fixture
def unit_2x2_alt():
    """Alternative 2x2 units convertible to unit_2x2: km, ms, g, deg."""
    return ((_km, _ms), (_g, _deg))


@pytest.fixture
def unit_1d():
    """Return a simple 1D unit tuple: m, s, kg."""
    return (_m, _s, _kg)


@pytest.fixture
def qm_1d(unit_1d):
    """Return a 1D QuantityMatrix (vector) with values 1-3."""
    return QMat(
        value=jnp.array([1.0, 2.0, 3.0]),
        unit=unit_1d,
    )


@pytest.fixture
def unit_1d_alt():
    """Alternative 1D units convertible to unit_1d: km, ms, g."""
    return (_km, _ms, _g)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for QuantityMatrix construction and basic properties."""

    def test_shape(self, qm_2x2):
        assert qm_2x2.shape == (2, 2)

    def test_n_rows(self, qm_2x2):
        assert qm_2x2.n_rows == 2

    def test_n_cols(self, qm_2x2):
        assert qm_2x2.n_cols == 2

    def test_value(self, qm_2x2):
        expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.array_equal(qm_2x2.value, expected)

    def test_unit(self, qm_2x2, unit_2x2):
        assert qm_2x2.unit == unit_2x2

    def test_aval(self, qm_2x2):
        aval = qm_2x2.aval()
        assert aval.shape == (2, 2)
        assert jnp.issubdtype(aval.dtype, jnp.floating)

    def test_materialise_raises(self, qm_2x2):
        with pytest.raises(RuntimeError, match="materialise"):
            qm_2x2.materialise()

    def test_batch_dims(self):
        """Batch dimensions are supported via leading axes."""
        qm = QMat(jnp.ones((5, 3, 2)), unit=((_m, _s), (_m, _s), (_m, _s)))
        assert qm.shape == (5, 3, 2)
        assert qm.n_rows == 3
        assert qm.n_cols == 2

    def test_1x1(self):
        """Degenerate 1x1 matrix."""
        qm = QMat(jnp.array([[42.0]]), unit=((_m,),))
        assert qm.n_rows == 1
        assert qm.n_cols == 1

    def test_nonsquare(self):
        """Non-square 2x3 matrix."""
        qm = QMat(jnp.ones((2, 3)), unit=((_m, _s, _kg), (_m, _s, _kg)))
        assert qm.n_rows == 2
        assert qm.n_cols == 3

    def test_unit_is_unitsmatrix(self, qm_2x2):
        """The ``unit`` field is always a ``UnitsMatrix`` instance."""
        assert isinstance(qm_2x2.unit, UnitsMatrix)

    def test_unit_converter_from_plain_tuples(self):
        """Plain nested tuples (of strings) are converted to ``UnitsMatrix``."""
        qm = QMat(jnp.array([[1.0]]), unit=(("m",),))
        assert isinstance(qm.unit, UnitsMatrix)
        assert qm.unit[0, 0] == _m

    def test_1d_construction(self, qm_1d, unit_1d):
        """1D vector construction."""
        assert qm_1d.ndim == 1
        assert qm_1d.shape == (3,)
        assert qm_1d.n_elems == 3
        assert qm_1d.unit == unit_1d

    def test_1d_value(self, qm_1d):
        """1D vector value."""
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.array_equal(qm_1d.value, expected)

    def test_1d_from_strings(self):
        """1D vector from unit strings."""
        qm = QMat(jnp.array([7.0, 8.0]), unit=("m", "s"))
        assert isinstance(qm.unit, UnitsMatrix)
        assert qm.unit[0] == _m
        assert qm.unit[1] == _s

    def test_1d_batch_dims(self):
        """1D vector with batch dimensions."""
        qm = QMat(jnp.ones((5, 3)), unit=(_m, _s, _kg))
        assert qm.ndim == 1
        assert qm.shape == (5, 3)

    def test_ndim_property_1d(self, qm_1d):
        """Ndim property returns 1 for vectors."""
        assert qm_1d.ndim == 1

    def test_ndim_property_2d(self, qm_2x2):
        """Ndim property returns 2 for matrices."""
        assert qm_2x2.ndim == 2

    def test_n_elems_error_on_2d(self, qm_2x2):
        """n_elems raises error for 2D."""
        with pytest.raises(ValueError, match="n_elems only available for 1D"):
            _ = qm_2x2.n_elems

    def test_n_rows_error_on_1d(self, qm_1d):
        """n_rows raises error for 1D."""
        with pytest.raises(ValueError, match="n_rows only available for 2D"):
            _ = qm_1d.n_rows

    def test_n_cols_error_on_1d(self, qm_1d):
        """n_cols raises error for 1D."""
        with pytest.raises(ValueError, match="n_cols only available for 2D"):
            _ = qm_1d.n_cols

    def test_repr(self, qm_2x2):
        """``repr(QuantityMatrix(...))`` succeeds and contains key info."""
        r = repr(qm_2x2)
        assert "QuantityMatrix" in r
        assert "((m, s), (kg, rad))" in r

    def test_repr_1x1(self):
        """Repr for a 1x1 matrix includes trailing-comma tuple syntax."""
        qm = QMat(jnp.array([[42.0]]), unit=((_m,),))
        r = repr(qm)
        assert "QuantityMatrix" in r
        assert "((m,),)" in r


# ---------------------------------------------------------------------------
# UnitsMatrix
# ---------------------------------------------------------------------------


class TestUnitsMatrix:
    """Tests for the ``UnitsMatrix`` tuple subclass."""

    def test_isinstance_tuple(self):
        units = UnitsMatrix(((_m, _s),))
        assert isinstance(units, tuple)

    def test_to_string_2x2(self):
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert units.to_string() == "((m, s), (kg, rad))"

    def test_to_string_1x1(self):
        units = UnitsMatrix(((_m,),))
        assert units.to_string() == "((m,),)"

    def test_to_string_1x2(self):
        units = UnitsMatrix(((_m, _s),))
        assert units.to_string() == "((m, s),)"

    def test_to_string_2x1(self):
        units = UnitsMatrix(((_m,), (_s,)))
        assert units.to_string() == "((m,), (s,))"

    def test_to_string_1d(self):
        """1D units to_string."""
        units = UnitsMatrix((_m, _s, _kg))
        assert units.to_string() == "(m, s, kg)"

    def test_to_string_1d_single(self):
        """Single element 1D."""
        units = UnitsMatrix((_m,))
        assert units.to_string() == "(m,)"

    def test_from_strings_2d(self):
        """``UnitsMatrix`` accepts unit strings via the ``u.unit`` converter."""
        units = UnitsMatrix((("m", "s"), ("kg", "rad")))
        assert units[0, 0] == _m
        assert units[1, 1] == _rad

    def test_from_strings_1d(self):
        """1D from strings."""
        units = UnitsMatrix(("m", "s", "kg"))
        assert units[0] == _m
        assert units[2] == _kg

    def test_idempotent_2d(self):
        """Constructing from an existing ``UnitsMatrix`` returns an equal copy."""
        orig = UnitsMatrix(((_m, _s), (_kg, _rad)))
        copy = UnitsMatrix(orig)
        assert copy == orig
        assert isinstance(copy, UnitsMatrix)

    def test_idempotent_1d(self):
        """1D idempotent."""
        orig = UnitsMatrix((_m, _s, _kg))
        copy = UnitsMatrix(orig)
        assert copy == orig
        assert isinstance(copy, UnitsMatrix)

    def test_shape_1d(self):
        """1D shape."""
        units = UnitsMatrix((_m, _s, _kg))
        assert units.shape == (3,)

    def test_shape_2d(self):
        """2D shape."""
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert units.shape == (2, 2)

    def test_ndim_1d(self):
        """1D ndim."""
        units = UnitsMatrix((_m, _s))
        assert units.ndim == 1

    def test_ndim_2d(self):
        """2D ndim."""
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert units.ndim == 2

    def test_indexing_1d(self):
        """1D indexing."""
        units = UnitsMatrix((_m, _s, _kg))
        assert units[0] == _m
        assert units[1] == _s
        assert units[2] == _kg

    def test_indexing_2d_tuple(self):
        """2D tuple indexing."""
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        assert units[0, 0] == _m
        assert units[0, 1] == _s
        assert units[1, 0] == _kg
        assert units[1, 1] == _rad

    def test_indexing_2d_single(self):
        """2D single-index returns a row."""
        units = UnitsMatrix(((_m, _s), (_kg, _rad)))
        row = units[0]
        assert isinstance(row, UnitsMatrix)
        assert row.shape == (2,)
        assert row[0] == _m
        assert row[1] == _s


# ---------------------------------------------------------------------------
# _convert_value_matrix / _convert_value_vector  (internal helpers)
# ---------------------------------------------------------------------------


class TestConvertValueMatrix:
    """Tests for the element-wise unit conversion helper (2D)."""

    def test_noop_same_units(self, unit_2x2):
        """If from_units == to_units no conversion happens."""
        val = jnp.array([[7.0, 8.0], [9.0, 10.0]])
        out = _convert_value_matrix(val, unit_2x2, unit_2x2)
        assert jnp.array_equal(out, val)

    def test_km_to_m(self):
        """1 km → 1000 m."""
        out = _convert_value_matrix(jnp.array([[3.0]]), ((_km,),), ((_m,),))
        assert jnp.isclose(out[0, 0], 3000.0)

    def test_mixed_conversion(self, unit_2x2, unit_2x2_alt):
        """Convert from (km, ms, g, deg) → (m, s, kg, rad)."""
        val = jnp.array([[1.0, 1000.0], [3000.0, 180.0]])
        out = _convert_value_matrix(val, unit_2x2_alt, unit_2x2)
        assert jnp.isclose(out[0, 0], 1000.0)  # 1 km -> 1000 m
        assert jnp.isclose(out[0, 1], 1.0)  # 1000 ms -> 1 s
        assert jnp.isclose(out[1, 0], 3.0)  # 3000 g -> 3 kg
        assert jnp.isclose(out[1, 1], math.pi, atol=1e-4)  # 180 deg -> pi rad

    def test_preserves_batch(self):
        """Batch dimensions are preserved."""
        val = jnp.array([[[2.0]], [[5.0]]])  # (2, 1, 1)
        out = _convert_value_matrix(val, ((_km,),), ((_m,),))
        assert out.shape == (2, 1, 1)
        assert jnp.isclose(out[0, 0, 0], 2000.0)
        assert jnp.isclose(out[1, 0, 0], 5000.0)


class TestConvertValueVector:
    """Tests for the element-wise unit conversion helper (1D)."""

    def test_noop_same_units(self, unit_1d):
        """If from_units == to_units no conversion happens."""
        val = jnp.array([7.0, 8.0, 9.0])
        out = _convert_value_vector(val, unit_1d, unit_1d)
        assert jnp.array_equal(out, val)

    def test_km_to_m(self):
        """1 km → 1000 m."""
        out = _convert_value_vector(jnp.array([3.0]), (_km,), (_m,))
        assert jnp.isclose(out[0], 3000.0)

    def test_mixed_conversion(self, unit_1d, unit_1d_alt):
        """Convert from (km, ms, g) → (m, s, kg)."""
        val = jnp.array([1.0, 1000.0, 3000.0])
        out = _convert_value_vector(val, unit_1d_alt, unit_1d)
        assert jnp.isclose(out[0], 1000.0)  # 1 km -> 1000 m
        assert jnp.isclose(out[1], 1.0)  # 1000 ms -> 1 s
        assert jnp.isclose(out[2], 3.0)  # 3000 g -> 3 kg

    def test_preserves_batch(self):
        """Batch dimensions are preserved."""
        val = jnp.array([[2.0], [5.0]])  # (2, 1)
        out = _convert_value_vector(val, (_km,), (_m,))
        assert out.shape == (2, 1)
        assert jnp.isclose(out[0, 0], 2000.0)
        assert jnp.isclose(out[1, 0], 5000.0)


# ---------------------------------------------------------------------------
# Addition  (quax lax.add_p)
# ---------------------------------------------------------------------------


@quax.quaxify
def _add(a, b):
    return a + b


class TestAddition:
    """Tests for QuantityMatrix + QuantityMatrix."""

    def test_same_units(self, qm_2x2, unit_2x2):
        """Simple add, same units."""
        other = QMat(
            value=jnp.array([[10.0, 20.0], [30.0, 40.0]]),
            unit=unit_2x2,
        )
        result = _add(qm_2x2, other)
        expected = jnp.array([[11.0, 22.0], [33.0, 44.0]])
        assert jnp.allclose(result.value, expected)
        assert result.unit == unit_2x2

    def test_result_keeps_lhs_units(self, qm_2x2, unit_2x2, unit_2x2_alt):
        """Result units come from the LHS."""
        other = QMat(
            value=jnp.array([[1.0, 1000.0], [3000.0, 180.0]]),
            unit=unit_2x2_alt,
        )
        result = _add(qm_2x2, other)
        assert result.unit == unit_2x2

    def test_mixed_unit_values(self, qm_2x2, unit_2x2_alt):
        """Values are correctly converted before addition."""
        other = QMat(
            value=jnp.array([[1.0, 1000.0], [3000.0, 180.0]]),
            unit=unit_2x2_alt,
        )
        res_val = _add(qm_2x2, other).value
        assert jnp.isclose(res_val[0, 0], 1001.0)  # 1 + 1000 m = 1001
        assert jnp.isclose(res_val[0, 1], 3.0)  # 2 + 1.0 s = 3
        assert jnp.isclose(res_val[1, 0], 6.0)  # 3 + 3.0 kg = 6
        assert jnp.isclose(res_val[1, 1], 4.0 + math.pi, atol=1e-4)  # 4+pi rad≈7.14159

    def test_add_zeros(self, qm_2x2, unit_2x2):
        """Adding zeros gives original values."""
        zeros = QMat(jnp.zeros((2, 2)), unit=unit_2x2)
        result = _add(qm_2x2, zeros)
        assert jnp.allclose(result.value, qm_2x2.value)

    def test_commutativity_same_units(self, unit_2x2):
        """A + b == b + a when units are the same."""
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=unit_2x2)
        b = QMat(jnp.array([[5.0, 6.0], [7.0, 8.0]]), unit=unit_2x2)
        r1 = _add(a, b)
        r2 = _add(b, a)
        assert jnp.allclose(r1.value, r2.value)

    def test_batch_addition(self, unit_2x2):
        """Batch dimensions are supported."""
        a = QMat(jnp.ones((3, 2, 2)), unit=unit_2x2)
        b = QMat(2 * jnp.ones((3, 2, 2)), unit=unit_2x2)
        result = _add(a, b)
        assert result.shape == (3, 2, 2)
        assert jnp.allclose(result.value, 3 * jnp.ones((3, 2, 2)))

    def test_1x1(self):
        """1x1 addition."""
        a = QMat(jnp.array([[3.0]]), unit=((_m,),))
        b = QMat(jnp.array([[7.0]]), unit=((_m,),))
        result = _add(a, b)
        assert jnp.isclose(result.value[0, 0], 10.0)

    def test_1d_addition_same_units(self, qm_1d, unit_1d):
        """1D vector addition, same units."""
        other = QMat(jnp.array([10.0, 20.0, 30.0]), unit=unit_1d)
        result = _add(qm_1d, other)
        expected = jnp.array([11.0, 22.0, 33.0])
        assert jnp.allclose(result.value, expected)
        assert result.unit == unit_1d

    def test_1d_addition_mixed_units(self, qm_1d, unit_1d_alt):
        """1D vector addition with unit conversion."""
        other = QMat(jnp.array([1.0, 1000.0, 3000.0]), unit=unit_1d_alt)
        result = _add(qm_1d, other)
        assert jnp.isclose(result.value[0], 1001.0)  # 1 + 1000 m
        assert jnp.isclose(result.value[1], 3.0)  # 2 + 1.0 s
        assert jnp.isclose(result.value[2], 6.0)  # 3 + 3.0 kg
        assert result.unit == qm_1d.unit

    def test_1d_batch_addition(self, unit_1d):
        """1D batch addition."""
        a = QMat(jnp.ones((3, 3)), unit=unit_1d)
        b = QMat(2 * jnp.ones((3, 3)), unit=unit_1d)
        result = _add(a, b)
        assert result.shape == (3, 3)
        assert jnp.allclose(result.value, 3 * jnp.ones((3, 3)))


# ---------------------------------------------------------------------------
# Subtraction  (quax lax.sub_p)
# ---------------------------------------------------------------------------


@quax.quaxify
def _sub(a, b):
    return a - b


class TestSubtraction:
    """Tests for QuantityMatrix - QuantityMatrix."""

    def test_same_units(self, qm_2x2, unit_2x2):
        """Simple sub, same units."""
        other = QMat(
            value=jnp.array([[10.0, 20.0], [30.0, 40.0]]),
            unit=unit_2x2,
        )
        result = _sub(other, qm_2x2)
        expected = jnp.array([[9.0, 18.0], [27.0, 36.0]])
        assert jnp.allclose(result.value, expected)
        assert result.unit == unit_2x2

    def test_result_keeps_lhs_units(self, qm_2x2, unit_2x2, unit_2x2_alt):
        """Result units come from the LHS."""
        other = QMat(
            value=jnp.array([[1.0, 1000.0], [3000.0, 180.0]]),
            unit=unit_2x2_alt,
        )
        result = _sub(qm_2x2, other)
        assert result.unit == unit_2x2

    def test_mixed_unit_values(self, qm_2x2, unit_2x2_alt):
        """Values are correctly converted before subtraction."""
        other = QMat(
            value=jnp.array([[1.0, 1000.0], [3000.0, 180.0]]),
            unit=unit_2x2_alt,
        )
        res_val = _sub(qm_2x2, other).value
        assert jnp.isclose(res_val[0, 0], -999.0)  # 1 - 1000 m
        assert jnp.isclose(res_val[0, 1], 1.0)  # 2 - 1.0 s
        assert jnp.isclose(res_val[1, 0], 0.0)  # 3 - 3.0 kg
        assert jnp.isclose(res_val[1, 1], 4.0 - math.pi, atol=1e-4)  # 4-pi rad

    def test_sub_zeros(self, qm_2x2, unit_2x2):
        """Subtracting zeros gives original values."""
        zeros = QMat(jnp.zeros((2, 2)), unit=unit_2x2)
        result = _sub(qm_2x2, zeros)
        assert jnp.allclose(result.value, qm_2x2.value)

    def test_self_subtraction(self, qm_2x2, unit_2x2):
        """A - a == 0."""
        result = _sub(qm_2x2, qm_2x2)
        assert jnp.allclose(result.value, jnp.zeros((2, 2)))
        assert result.unit == unit_2x2

    def test_anticommutativity_same_units(self, unit_2x2):
        """A - b == -(b - a) when units are the same."""
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=unit_2x2)
        b = QMat(jnp.array([[5.0, 6.0], [7.0, 8.0]]), unit=unit_2x2)
        r1 = _sub(a, b)
        r2 = _sub(b, a)
        assert jnp.allclose(r1.value, -r2.value)

    def test_batch_subtraction(self, unit_2x2):
        """Batch dimensions are supported."""
        a = QMat(3 * jnp.ones((3, 2, 2)), unit=unit_2x2)
        b = QMat(jnp.ones((3, 2, 2)), unit=unit_2x2)
        result = _sub(a, b)
        assert result.shape == (3, 2, 2)
        assert jnp.allclose(result.value, 2 * jnp.ones((3, 2, 2)))

    def test_1x1(self):
        """1x1 subtraction."""
        a = QMat(jnp.array([[7.0]]), unit=((_m,),))
        b = QMat(jnp.array([[3.0]]), unit=((_m,),))
        result = _sub(a, b)
        assert jnp.isclose(result.value[0, 0], 4.0)

    def test_1d_subtraction_same_units(self, qm_1d, unit_1d):
        """1D vector subtraction, same units."""
        other = QMat(jnp.array([10.0, 20.0, 30.0]), unit=unit_1d)
        result = _sub(other, qm_1d)
        expected = jnp.array([9.0, 18.0, 27.0])
        assert jnp.allclose(result.value, expected)
        assert result.unit == unit_1d

    def test_1d_subtraction_mixed_units(self, qm_1d, unit_1d_alt):
        """1D vector subtraction with unit conversion."""
        other = QMat(jnp.array([1.0, 1000.0, 3000.0]), unit=unit_1d_alt)
        result = _sub(qm_1d, other)
        assert jnp.isclose(result.value[0], -999.0)  # 1 - 1000 m
        assert jnp.isclose(result.value[1], 1.0)  # 2 - 1.0 s
        assert jnp.isclose(result.value[2], 0.0)  # 3 - 3.0 kg
        assert result.unit == qm_1d.unit


# ---------------------------------------------------------------------------
# Dot product / matmul  (quax lax.dot_general_p)
# ---------------------------------------------------------------------------


@quax.quaxify
def _matmul(a, b):
    return a @ b


class TestDotProduct:
    """Tests for QuantityMatrix @ QuantityMatrix."""

    def test_simple_matmul_uniform_units(self):
        """2x2 @ 2x1 with uniform units along contraction axis."""
        a = QMat(jnp.array([[2.0, 3.0], [4.0, 5.0]]), unit=((_m, _m), (_kg, _kg)))
        b = QMat(jnp.array([[10.0], [20.0]]), unit=((_s,), (_s,)))
        result = _matmul(a, b)
        # C[0,0] = 2*10 + 3*20 = 80 in m*s
        # C[1,0] = 4*10 + 5*20 = 140 in kg*s
        assert jnp.isclose(result.value[0, 0], 80.0)
        assert jnp.isclose(result.value[1, 0], 140.0)
        assert result.unit == (
            (_m * _s,),
            (_kg * _s,),
        )
        assert result.n_rows == 2
        assert result.n_cols == 1

    def test_matmul_with_unit_conversion(self):
        """Contraction axis has mixed units that need conversion."""
        # A is 2x2 with units [[m, km], [kg, kg]]
        # B is 2x1 with units [[s], [s]]
        # C[0,0] = m*s + km*s -> converted to m*s (ref is j=0)
        #        = 2*10 + (3 km = 3000 m)*20 s = 20 + 60000 = 60020
        a = QMat(jnp.array([[2.0, 3.0], [4.0, 5.0]]), unit=((_m, _km), (_kg, _kg)))
        b = QMat(jnp.array([[10.0], [20.0]]), unit=((_s,), (_s,)))
        result = _matmul(a, b)
        assert jnp.isclose(result.value[0, 0], 60020.0)
        # C[1,0] = 4*10 + 5*20 = 140 (uniform kg*s, no conversion)
        assert jnp.isclose(result.value[1, 0], 140.0)

    def test_matmul_2x2_by_2x2(self):
        """Square 2x2 @ 2x2 matmul."""
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=((_m, _m), (_m, _m)))
        b = QMat(jnp.array([[5.0, 6.0], [7.0, 8.0]]), unit=((_s, _s), (_s, _s)))
        result = _matmul(a, b)
        # Standard matmul: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        #                = [[19, 22], [43, 50]]
        expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
        assert jnp.allclose(result.value, expected)
        ms = _m * _s
        assert result.unit == ((ms, ms), (ms, ms))

    def test_matmul_identity(self):
        """Multiply by identity matrix."""
        a = QMat(jnp.array([[3.0, 7.0], [11.0, 13.0]]), unit=((_m, _m), (_m, _m)))
        identity = QMat(jnp.eye(2), unit=((_dimless, _dimless), (_dimless, _dimless)))
        result = _matmul(a, identity)
        assert jnp.allclose(result.value, a.value)

    def test_matmul_output_units(self):
        """Output unit[i][k] = lhs.unit[i][0] * rhs.unit[0][k]."""
        a = QMat(jnp.array([[1.0, 1.0]]), unit=((_m, _m),))
        b = QMat(jnp.array([[2.0, 3.0], [4.0, 5.0]]), unit=((_s, _kg), (_s, _kg)))
        result = _matmul(a, b)
        # Output shape: 1x2
        assert result.n_rows == 1
        assert result.n_cols == 2
        # Output units: row 0 from A.unit[0][0]=m, col 0 from B.unit[0][0]=s,
        # col 1 from B.unit[0][1]=kg
        assert result.unit[0][0] == _m * _s
        assert result.unit[0][1] == _m * _kg

    def test_matmul_1x1(self):
        """1x1 @ 1x1 is scalar product."""
        a = QMat(jnp.array([[3.0]]), unit=((_m,),))
        b = QMat(jnp.array([[7.0]]), unit=((_s,),))
        result = _matmul(a, b)
        assert jnp.isclose(result.value[0, 0], 21.0)
        assert result.unit == ((_m * _s,),)

    def test_matmul_rhs_unit_conversion(self):
        """Unit conversion needed on the RHS contraction axis."""
        # A: 1x2, units [[m, m]]
        # B: 2x1, units [[s], [min]]
        # C[0,0] = m*s + m*min -> ref = m*s
        #        = 1*1 + 1*1 min -> 1*1 + 1*60 s = 1 + 60 = 61 in m*s
        a = QMat(jnp.array([[1.0, 1.0]]), ((_m, _m),))
        b = QMat(jnp.array([[1.0], [1.0]]), ((_s,), (_min,)))
        result = _matmul(a, b)
        assert jnp.isclose(result.value[0, 0], 61.0)
        assert result.unit == ((_m * _s,),)

    def test_1d_dot_product_uniform_units(self):
        """1D @ 1D vector dot product with uniform units."""
        a = QMat(jnp.array([2.0, 3.0]), unit=(_m, _m))
        b = QMat(jnp.array([4.0, 5.0]), unit=(_s, _s))
        result = _matmul(a, b)
        # Result should be a scalar Quantity, not a QuantityMatrix
        # 2*4 + 3*5 = 8 + 15 = 23 in m*s
        assert isinstance(result, u.Quantity)
        assert jnp.isclose(result.value, 23.0)
        assert result.unit == _m * _s

    def test_1d_dot_product_mixed_units(self):
        """1D @ 1D with mixed units requiring conversion."""
        # a: [1 m, 1 km], b: [1 s, 1 s]
        # Result = 1*1 + 1000*1 = 1 + 1000 = 1001 in m*s
        a = QMat(jnp.array([1.0, 1.0]), unit=(_m, _km))
        b = QMat(jnp.array([1.0, 1.0]), unit=(_s, _s))
        result = _matmul(a, b)
        assert isinstance(result, u.Quantity)
        assert jnp.isclose(result.value, 1001.0)
        assert result.unit == _m * _s

    def test_1d_dot_product_batch(self):
        """1D @ 1D with batch dimensions."""
        # Batch of 3 vectors, each length 2
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), unit=(_m, _m))
        b = QMat(jnp.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]), unit=(_s, _s))

        @quax.quaxify
        def dot_batched(x, y):
            return x @ y

        result = jax.vmap(dot_batched)(a, b)
        # [1*7 + 2*8, 3*9 + 4*10, 5*11 + 6*12] = [23, 67, 127]
        assert jnp.isclose(result.value[0], 23.0)
        assert jnp.isclose(result.value[1], 67.0)
        assert jnp.isclose(result.value[2], 127.0)


# ---------------------------------------------------------------------------
# JAX integration
# ---------------------------------------------------------------------------


class TestJaxIntegration:
    """QuantityMatrix works with JAX transformations."""

    def test_jit_add(self, unit_2x2):
        """jit-compiled addition."""
        a = QMat(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=unit_2x2)
        b = QMat(jnp.array([[5.0, 6.0], [7.0, 8.0]]), unit=unit_2x2)
        result = jax.jit(_add)(a, b)
        expected = jnp.array([[6.0, 8.0], [10.0, 12.0]])
        assert jnp.allclose(result.value, expected)

    def test_jit_matmul(self):
        """jit-compiled matmul."""
        a = QMat(jnp.array([[2.0, 3.0]]), unit=((_m, _m),))
        b = QMat(jnp.array([[4.0], [5.0]]), unit=((_s,), (_s,)))
        result = jax.jit(_matmul)(a, b)
        assert jnp.isclose(result.value[0, 0], 23.0)

    def test_pytree_flatten_unflatten(self, qm_2x2, unit_2x2):
        """QuantityMatrix is a proper PyTree."""
        leaves, treedef = jax.tree.flatten(qm_2x2)
        restored = jax.tree.unflatten(treedef, leaves)
        assert jnp.array_equal(restored.value, qm_2x2.value)
        assert restored.unit == unit_2x2

    def test_vmap_add(self, unit_2x2):
        """Vmap over batch dimension for addition."""
        a = QMat(jnp.ones((4, 2, 2)), unit=unit_2x2)
        b = QMat(2 * jnp.ones((4, 2, 2)), unit=unit_2x2)

        @quax.quaxify
        def add_batched(x, y):
            return x + y

        result = jax.vmap(add_batched)(a, b)
        assert result.shape == (4, 2, 2)
        assert jnp.allclose(result.value, 3 * jnp.ones((4, 2, 2)))


# ---------------------------------------------------------------------------
# plum.convert registration
# ---------------------------------------------------------------------------


class TestPlumConversion:
    """Tests for ``plum.convert`` registrations involving ``QuantityMatrix``."""

    def test_quantitymatrix_to_quantity_uniform_1d(self):
        """1D uniform-unit ``QuantityMatrix`` converts to ``u.Quantity``."""
        qm = QMat(value=jnp.array([1.0, 2.0, 3.0]), unit=(_m, _m, _m))

        result = plum.convert(qm, u.Quantity)

        assert isinstance(result, u.Quantity)
        assert result.unit == _m
        assert jnp.array_equal(result.value, qm.value)

    def test_quantitymatrix_to_quantity_uniform_2d(self):
        """2D uniform-unit ``QuantityMatrix`` converts to ``u.Quantity``."""
        qm = QMat(
            value=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            unit=((_s, _s), (_s, _s)),
        )

        result = plum.convert(qm, u.Quantity)

        assert isinstance(result, u.Quantity)
        assert result.unit == _s
        assert result.shape == (2, 2)
        assert jnp.array_equal(result.value, qm.value)

    def test_quantitymatrix_to_quantity_heterogeneous_units_raises(self):
        """Mixed units cannot be converted to a single ``u.Quantity``."""
        qm = QMat(value=jnp.array([1.0, 2.0]), unit=(_m, _s))

        with pytest.raises(
            ValueError,
            match="all units are identical",
        ):
            plum.convert(qm, u.Quantity)


# ---------------------------------------------------------------------------
# Affine / logarithmic product-unit guards
# ---------------------------------------------------------------------------


class TestAffineProductUnitsRejected:
    """Verify that astropy rejects product conversions for affine units.

    The ``dot_general_qm_qm`` implementation uses a multiplicative scale
    factor (``scale_3d``).  This is correct because affine units (°C, °F)
    are the only units where a multiplicative scale would be wrong (they
    have an additive offset), and astropy rejects product conversions
    involving them.  Logarithmic units (dex, mag) in products become
    plain ``CompositeUnit`` objects whose conversion IS multiplicative.

    If astropy ever starts accepting affine product conversions, these
    tests will *fail* — that's intentional: it means the assumption
    behind ``scale_3d`` must be revisited.
    """

    def test_degC_times_s_not_convertible(self):
        """°C·s → °F·s must fail (affine offset is undefined for products)."""
        with pytest.raises(apu.UnitConversionError, match="not convertible"):
            (apu.deg_C * apu.s).to(imperial.deg_F * apu.s, 1.0)

    def test_degF_times_s_not_convertible(self):
        """°F·s → °C·s must fail (symmetric check)."""
        with pytest.raises(apu.UnitConversionError, match="not convertible"):
            (imperial.deg_F * apu.s).to(apu.deg_C * apu.s, 1.0)

    def test_degC_times_m_not_convertible(self):
        """°C·m → °F·m must fail."""
        with pytest.raises(apu.UnitConversionError, match="not convertible"):
            (apu.deg_C * apu.m).to(imperial.deg_F * apu.m, 1.0)

    def test_dex_times_s_is_convertible(self):
        """dex·s → dex·ms succeeds.

        dex in a product is a plain CompositeUnit; only the s → ms part
        converts, multiplicatively.
        """
        result = (apu.dex() * apu.s).to(apu.dex() * apu.ms, 1.0)
        assert math.isclose(result, 1000.0, rel_tol=1e-12)

    def test_mag_times_s_is_convertible(self):
        """mag·s → mag·ms succeeds (same reasoning as dex)."""
        result = (apu.mag() * apu.s).to(apu.mag() * apu.ms, 1.0)
        assert math.isclose(result, 1000.0, rel_tol=1e-12)

    def test_kelvin_times_s_is_convertible(self):
        """K·s → K·ms MUST succeed (Kelvin is absolute / linear).

        This is the *positive* control: linear temperature products work
        fine and the multiplicative scale is correct.
        """
        result = (apu.K * apu.s).to(apu.K * apu.ms, 1.0)
        assert math.isclose(result, 1000.0, rel_tol=1e-12)
