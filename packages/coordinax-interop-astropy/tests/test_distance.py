"""Test distance conversions between Astropy and coordinax."""

import astropy.units as apyu
import pytest
from astropy.units import Quantity as AstropyQuantity
from hypothesis import given, strategies as st
from plum import convert

import unxt_hypothesis as ust

import coordinax.interop.astropy  # noqa: F401
from coordinax.distance import Distance, DistanceModulus, Parallax

# ==============================================================================
# Helper strategies


def float32s(
    min_value: float | None = None,
    max_value: float | None = None,
) -> st.SearchStrategy[float]:
    """Hypothesis strategy for float32-representable floats."""
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        width=32,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


# =============================================================================
# Distance


def test_astropy_quantity_to_distance() -> None:
    """Test converting AstropyQuantity to Distance."""
    q = AstropyQuantity(1.0, "km")
    dist = convert(q, Distance)

    assert isinstance(dist, Distance)
    assert dist.value == pytest.approx(1.0)
    assert str(dist.unit) == "km"


@given(unit=ust.units("length"))
def test_astropy_quantity_to_distance(unit: str) -> None:
    """Test converting Astropy Quantity to Distance."""
    apyq = apyu.Q(42.0, unit)
    dist = convert(apyq, Distance)

    assert isinstance(dist, Distance)
    assert dist.value == apyq.value
    assert dist.unit == apyq.unit


@given(
    dist=ust.quantities(
        ust.units("length"),
        elements=float32s(min_value=1.0, max_value=1e6),
        quantity_cls=Distance,
    )
)
def test_distance_to_astropy_quantity(dist: Distance) -> None:
    """Test converting Distance to AstropyQuantity."""
    apyq = convert(dist, apyu.Q)

    assert isinstance(apyq, apyu.Q)
    assert apyq.value == dist.value
    assert apyq.unit == dist.unit


@given(
    dist=ust.quantities(
        ust.units("length"),
        elements=float32s(min_value=1.0, max_value=1e6),
        quantity_cls=Distance,
    )
)
def test_distance_roundtrip(dist: Distance) -> None:
    apyq = convert(dist, apyu.Q)
    dist_back = convert(apyq, Distance)

    assert isinstance(dist_back, Distance)
    assert dist_back.value == dist.value
    assert dist_back.unit == dist.unit


# =============================================================================
# Parallax


def test_astropy_quantity_to_parallax() -> None:
    """Test converting AstropyQuantity to Parallax."""
    q = AstropyQuantity(1.0, "mas")
    plx = convert(q, Parallax)

    assert isinstance(plx, Parallax)
    assert plx.value == pytest.approx(1.0)
    assert str(plx.unit) == "mas"


@given(unit=ust.units("angle"))
def test_astropy_quantity_to_parallax(unit: str) -> None:
    """Test converting Astropy Quantity to Parallax."""
    apyq = apyu.Q(0.5, unit)
    plx = convert(apyq, Parallax)

    assert isinstance(plx, Parallax)
    assert plx.value == apyq.value
    assert plx.unit == apyq.unit


@given(
    plx=ust.quantities(
        ust.units("angle"),
        elements=float32s(min_value=0.0625, max_value=1.0),
        quantity_cls=Parallax,
    )
)
def test_parallax_to_astropy_quantity(plx: Parallax) -> None:
    """Test converting Parallax to AstropyQuantity."""
    apyq = convert(plx, apyu.Q)

    assert isinstance(apyq, apyu.Q)
    assert apyq.value == plx.value
    assert apyq.unit == plx.unit


@given(
    plx=ust.quantities(
        ust.units("angle"),
        elements=float32s(min_value=0.0625, max_value=1.0),
        quantity_cls=Parallax,
    )
)
def test_parallax_roundtrip(plx: Parallax) -> None:
    apyq = convert(plx, apyu.Q)
    plx_back = convert(apyq, Parallax)

    assert isinstance(plx_back, Parallax)
    assert plx_back.value == plx.value
    assert plx_back.unit == plx.unit


# =============================================================================
# DistanceModulus


def test_astropy_quantity_to_distancemodulus() -> None:
    """Test converting AstropyQuantity to DistanceModulus."""
    q = AstropyQuantity(5.0, "mag")
    dm = convert(q, DistanceModulus)

    assert isinstance(dm, DistanceModulus)
    assert dm.value == pytest.approx(5.0)
    assert str(dm.unit) == "mag"


def test_distancemodulus_to_astropy_quantity() -> None:
    """Test converting DistanceModulus to AstropyQuantity."""
    dm = DistanceModulus(5.0, "mag")
    apyq = convert(dm, apyu.Q)

    assert isinstance(apyq, apyu.Q)
    assert apyq.value == dm.value
    assert apyq.unit == dm.unit


def test_distancemodulus_roundtrip() -> None:
    """Test roundtrip conversion for DistanceModulus."""
    dm = DistanceModulus(5.0, "mag")
    apyq = convert(dm, apyu.Q)
    dm_back = convert(apyq, DistanceModulus)

    assert isinstance(dm_back, DistanceModulus)
    assert dm_back.value == dm.value
    assert dm_back.unit == dm.unit
