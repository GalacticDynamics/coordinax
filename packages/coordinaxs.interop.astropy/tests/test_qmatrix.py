"""Converters between `unxts.linalg` matrix types and Astropy structured units."""

__all__: tuple[str, ...] = ()

import astropy.units as apyu
import jax.numpy as jnp
import numpy as np
import plum
import pytest

import unxt as u
import unxts.linalg as ul

#: Unit layouts, exercised in both directions.
UNIT_LAYOUTS = [
    pytest.param(("m", "s", "kg"), id="1d"),
    pytest.param((("m", "s"), ("kg", "rad")), id="2d"),
]

#: The same layouts, paired with the shape they must round-trip to.
UNIT_LAYOUTS_WITH_SHAPE = [
    pytest.param(("m", "s", "kg"), (3,), id="1d"),
    pytest.param((("m", "s"), ("kg", "rad")), (2, 2), id="2d"),
]


@pytest.mark.parametrize("units", UNIT_LAYOUTS)
def test_units_matrix_to_structured_unit(units: tuple) -> None:
    """UnitsMatrix -> StructuredUnit preserves layout and units."""
    result = plum.convert(ul.UnitsMatrix(units), apyu.StructuredUnit)

    assert isinstance(result, apyu.StructuredUnit)
    assert result == apyu.StructuredUnit(units)


@pytest.mark.parametrize(("units", "shape"), UNIT_LAYOUTS_WITH_SHAPE)
def test_structured_unit_to_units_matrix(units: tuple, shape: tuple) -> None:
    """StructuredUnit -> UnitsMatrix preserves layout and units."""
    result = plum.convert(apyu.StructuredUnit(units), ul.UnitsMatrix)

    assert isinstance(result, ul.UnitsMatrix)
    assert result.shape == shape
    for index in np.ndindex(shape):
        expected = units
        for i in index:
            expected = expected[i]
        assert result[index] == u.unit(expected)


@pytest.mark.parametrize("units", UNIT_LAYOUTS)
def test_units_matrix_roundtrip(units: tuple) -> None:
    """UnitsMatrix -> StructuredUnit -> UnitsMatrix is the identity."""
    umat = ul.UnitsMatrix(units)
    assert plum.convert(plum.convert(umat, apyu.StructuredUnit), ul.UnitsMatrix) == umat


@pytest.mark.parametrize("units", UNIT_LAYOUTS)
def test_structured_unit_roundtrip(units: tuple) -> None:
    """StructuredUnit -> UnitsMatrix -> StructuredUnit is the identity."""
    su = apyu.StructuredUnit(units)
    assert plum.convert(plum.convert(su, ul.UnitsMatrix), apyu.StructuredUnit) == su


class TestQuantityMatrixToAstropyQuantity:
    """QuantityMatrix -> apyu.Quantity carries values and structured units."""

    def test_type_and_unit(self) -> None:
        qmat = ul.QuantityMatrix(jnp.array([1, 2]), unit=("km", "s"))
        result = plum.convert(qmat, apyu.Quantity)

        assert isinstance(result, apyu.Quantity)
        assert result.unit == apyu.StructuredUnit(("km", "s"))

    def test_values(self) -> None:
        qmat = ul.QuantityMatrix(jnp.array([3, 4]), unit=("m", "kg"))
        arr = np.array(plum.convert(qmat, apyu.Quantity))

        assert float(arr["f0"]) == pytest.approx(3)
        assert float(arr["f1"]) == pytest.approx(4)
