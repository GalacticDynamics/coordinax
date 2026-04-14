"""Test angle conversions between Astropy and coordinax."""

import pytest
from astropy.coordinates import Angle as AstropyAngle
from hypothesis import given
from plum import convert

import unxt_hypothesis as ust

import coordinax.angles as cxa
import coordinax.interop.astropy  # noqa: F401

# =============================================================================
# AstropyAngle -> cxa.Angle


def test_astropy_angle_to_cx_angle() -> None:
    """Test converting AstropyAngle to cxa.Angle."""
    apy = AstropyAngle(1.0, "rad")
    angle = convert(apy, cxa.Angle)

    assert isinstance(angle, cxa.Angle)
    assert angle.value == pytest.approx(1.0)
    assert str(angle.unit) == "rad"


@given(unit=ust.units("angle"))
def test_astropy_angle_to_cx_angle_hypothesis(unit: str) -> None:
    """Test converting AstropyAngle to cxa.Angle for various angle units."""
    apy = AstropyAngle(0.5, unit)
    angle = convert(apy, cxa.Angle)

    assert isinstance(angle, cxa.Angle)
    assert angle.value == pytest.approx(apy.value)
    assert angle.unit == apy.unit


# =============================================================================
# cxa.Angle -> AstropyAngle


def test_cx_angle_to_astropy_angle() -> None:
    """Test converting cxa.Angle to AstropyAngle."""
    angle = cxa.Angle(1.0, "rad")
    apy = convert(angle, AstropyAngle)

    assert isinstance(apy, AstropyAngle)
    assert apy.value == pytest.approx(1.0)
    assert str(apy.unit) == "rad"


@given(angle=ust.angles())
def test_cx_angle_to_astropy_angle_hypothesis(angle: cxa.Angle) -> None:
    """Test converting cxa.Angle to AstropyAngle for various angles."""
    apy = convert(angle, AstropyAngle)

    assert isinstance(apy, AstropyAngle)
    assert apy.value == pytest.approx(float(angle.value))
    assert apy.unit == angle.unit


# =============================================================================
# Roundtrip


def test_angle_roundtrip() -> None:
    """Test roundtrip: cxa.Angle -> AstropyAngle -> cxa.Angle."""
    angle = cxa.Angle(1.5, "deg")
    apy = convert(angle, AstropyAngle)
    angle_back = convert(apy, cxa.Angle)

    assert isinstance(angle_back, cxa.Angle)
    assert angle_back.value == pytest.approx(angle.value)
    assert angle_back.unit == angle.unit


@given(angle=ust.angles())
def test_angle_roundtrip_hypothesis(angle: cxa.Angle) -> None:
    """Test roundtrip: cxa.Angle -> AstropyAngle -> cxa.Angle."""
    apy = convert(angle, AstropyAngle)
    angle_back = convert(apy, cxa.Angle)

    assert isinstance(angle_back, cxa.Angle)
    assert angle_back.value == pytest.approx(float(angle.value))
    assert angle_back.unit == angle.unit
