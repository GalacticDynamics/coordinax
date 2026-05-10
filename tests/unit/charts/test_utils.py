"""Tests for chart utility helpers."""

import pytest

import quaxed.numpy as jnp
import unxt as u

from coordinax._src.utils import uconvert_to_rad


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


def test_uconvert_to_rad_arraylike_without_usys_is_radians() -> None:
    """Plain numerics and JAX arrays are interpreted as radians by default."""
    scalar = uconvert_to_rad(float(jnp.pi / 3), None)
    vector = uconvert_to_rad(jnp.array([0.0, jnp.pi / 2, jnp.pi]), None)

    assert isinstance(scalar, float)
    assert scalar == pytest.approx(float(jnp.pi / 3))
    assert bool(jnp.allclose(vector, jnp.array([0.0, jnp.pi / 2, jnp.pi])))


def test_uconvert_to_rad_arraylike_with_usys_angle_unit() -> None:
    """Plain numerics and JAX arrays can be interpreted through usys['angle']."""
    usys = u.unitsystem("m", "deg")

    scalar = uconvert_to_rad(90.0, usys)
    vector = uconvert_to_rad(jnp.array([0.0, 90.0, 180.0]), usys)

    assert scalar == pytest.approx(float(jnp.pi / 2))
    assert bool(jnp.allclose(vector, jnp.array([0.0, jnp.pi / 2, jnp.pi])))
