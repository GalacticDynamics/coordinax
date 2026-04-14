"""Tests for coordinax.interop.astropy qmatrix converters."""

import astropy.units as apyu
import jax.numpy as jnp
import numpy as np
import plum
import pytest

import unxt as u

from coordinax.internal import QuantityMatrix, UnitsMatrix


class TestUnitsMatrixToStructuredUnit:
    """Tests for UnitsMatrix → apyu.StructuredUnit conversion."""

    def test_1d(self) -> None:
        umat = UnitsMatrix(("m", "s", "kg"))
        result = plum.convert(umat, apyu.StructuredUnit)
        assert isinstance(result, apyu.StructuredUnit)
        assert result == apyu.StructuredUnit(("m", "s", "kg"))

    def test_1d_values(self) -> None:
        umat = UnitsMatrix(("km", "s"))
        result = plum.convert(umat, apyu.StructuredUnit)
        vals = result.values()
        assert vals[0] == apyu.Unit("km")
        assert vals[1] == apyu.Unit("s")

    def test_2d(self) -> None:
        umat = UnitsMatrix((("m", "s"), ("kg", "rad")))
        result = plum.convert(umat, apyu.StructuredUnit)
        assert isinstance(result, apyu.StructuredUnit)
        assert result == apyu.StructuredUnit((("m", "s"), ("kg", "rad")))

    def test_roundtrip_via_structured_unit(self) -> None:
        """UnitsMatrix → StructuredUnit → UnitsMatrix is identity."""
        umat = UnitsMatrix(("m", "s", "kg"))
        su = plum.convert(umat, apyu.StructuredUnit)
        result = plum.convert(su, UnitsMatrix)
        assert result == umat


class TestStructuredUnitToUnitsMatrix:
    """Tests for apyu.StructuredUnit → UnitsMatrix conversion."""

    def test_1d(self) -> None:
        su = apyu.StructuredUnit(("m", "s", "kg"))
        result = plum.convert(su, UnitsMatrix)
        assert isinstance(result, UnitsMatrix)
        assert result.shape == (3,)

    def test_1d_units(self) -> None:
        su = apyu.StructuredUnit(("m", "s", "kg"))
        result = plum.convert(su, UnitsMatrix)

        assert result[0] == u.unit("m")
        assert result[1] == u.unit("s")
        assert result[2] == u.unit("kg")

    def test_2d(self) -> None:
        su = apyu.StructuredUnit((("m", "s"), ("kg", "rad")))
        result = plum.convert(su, UnitsMatrix)
        assert isinstance(result, UnitsMatrix)
        assert result.shape == (2, 2)

    def test_2d_units(self) -> None:
        su = apyu.StructuredUnit((("m", "s"), ("kg", "rad")))
        result = plum.convert(su, UnitsMatrix)

        assert result[0, 0] == u.unit("m")
        assert result[0, 1] == u.unit("s")
        assert result[1, 0] == u.unit("kg")
        assert result[1, 1] == u.unit("rad")

    def test_roundtrip_via_unitsmatrix(self) -> None:
        """StructuredUnit → UnitsMatrix → StructuredUnit is identity."""
        su = apyu.StructuredUnit(("m", "s", "kg"))
        umat = plum.convert(su, UnitsMatrix)
        result = plum.convert(umat, apyu.StructuredUnit)
        assert result == su


class TestQuantityMatrixToAstropyQuantity:
    """Tests for QuantityMatrix → apyu.Quantity conversion."""

    def test_1d(self) -> None:
        qmat = QuantityMatrix(jnp.array([1.0, 2.0]), unit=("km", "s"))
        result = plum.convert(qmat, apyu.Quantity)
        assert isinstance(result, apyu.Quantity)

    def test_1d_unit(self) -> None:
        qmat = QuantityMatrix(jnp.array([1.0, 2.0]), unit=("km", "s"))
        result = plum.convert(qmat, apyu.Quantity)
        assert result.unit == apyu.StructuredUnit(("km", "s"))

    def test_1d_values(self) -> None:
        qmat = QuantityMatrix(jnp.array([3.0, 4.0]), unit=("m", "kg"))
        result = plum.convert(qmat, apyu.Quantity)

        arr = np.array(result)
        assert float(arr["f0"]) == pytest.approx(3.0)
        assert float(arr["f1"]) == pytest.approx(4.0)
