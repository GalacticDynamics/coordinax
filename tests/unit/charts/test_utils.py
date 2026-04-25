"""Tests for chart utility helpers."""

import pytest

import quaxed.numpy as jnp
import unxt as u

from coordinax.charts._src.utils import uconvert_to_rad


def test_uconvert_to_rad_angle_quantity_uses_own_unit() -> None:
    """Angular quantities are converted from their native unit."""
    out = uconvert_to_rad(u.Q(90, "deg"), None)
    assert out.unit == u.unit("rad")
    assert float(u.ustrip("rad", out)) == pytest.approx(float(jnp.pi / 2))


def test_uconvert_to_rad_dimensionless_quantity_uses_from_unit() -> None:
    """Dimensionless quantities are interpreted via the unit system angle unit."""
    out = uconvert_to_rad(u.Q(90, ""), u.unitsystem("m", "deg"))
    assert out.unit == u.unit("rad")
    assert float(u.ustrip("rad", out)) == pytest.approx(float(jnp.pi / 2))


def test_uconvert_to_rad_non_angle_quantity_raises() -> None:
    """Non-angle, non-dimensionless quantities are rejected."""
    with pytest.raises(
        ValueError, match="Unsupported quantity dimension for angle conversion: length"
    ):
        uconvert_to_rad(u.Q(1, "m"), None)
