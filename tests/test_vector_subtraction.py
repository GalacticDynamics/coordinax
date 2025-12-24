"""Unit tests for Vector.sub() and subtraction operator."""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx


class TestVectorSubtraction:
    """Test vector subtraction with role-aware semantics."""

    def test_point_minus_point_gives_pos(self):
        """Point - Point -> Pos (affine difference)."""
        p1 = cx.Vector(
            {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")},
            cx.charts.cart3d,
            cx.roles.point,
        )
        p2 = cx.Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")},
            cx.charts.cart3d,
            cx.roles.point,
        )

        # Using method
        result = p1.sub(p2)
        assert isinstance(result.role, cx.roles.Pos)
        assert jnp.allclose(result.data["x"].value, 2.0)
        assert jnp.allclose(result.data["y"].value, 3.0)

        # Using operator
        result = p1 - p2
        assert isinstance(result.role, cx.roles.Pos)
        assert jnp.allclose(result.data["x"].value, 2.0)
        assert jnp.allclose(result.data["y"].value, 3.0)

    def test_point_minus_pos_gives_point(self):
        """Point - Pos -> Point (backwards translation)."""
        point = cx.Vector(
            {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")},
            cx.charts.cart3d,
            cx.roles.point,
        )
        disp = cx.Vector(
            {"x": u.Q(0.5, "m"), "y": u.Q(0.5, "m"), "z": u.Q(0.0, "m")},
            cx.charts.cart3d,
            cx.roles.pos,
        )

        # Using method
        result = point.sub(disp)
        assert isinstance(result.role, cx.roles.Point)
        assert jnp.allclose(result.data["x"].value, 2.5)
        assert jnp.allclose(result.data["y"].value, 3.5)

        # Using operator
        result = point - disp
        assert isinstance(result.role, cx.roles.Point)
        assert jnp.allclose(result.data["x"].value, 2.5)
        assert jnp.allclose(result.data["y"].value, 3.5)

    def test_pos_minus_pos_gives_pos(self):
        """Pos - Pos -> Pos (vector subtraction)."""
        d1 = cx.Vector(
            {"x": u.Q(2.0, "m"), "y": u.Q(3.0, "m"), "z": u.Q(0.0, "m")},
            cx.charts.cart3d,
            cx.roles.pos,
        )
        d2 = cx.Vector(
            {"x": u.Q(0.5, "m"), "y": u.Q(1.5, "m"), "z": u.Q(0.0, "m")},
            cx.charts.cart3d,
            cx.roles.pos,
        )

        # Using method
        result = d1.sub(d2)
        assert isinstance(result.role, cx.roles.Pos)
        assert jnp.allclose(result.data["x"].value, 1.5)
        assert jnp.allclose(result.data["y"].value, 1.5)

        # Using operator
        result = d1 - d2
        assert isinstance(result.role, cx.roles.Pos)
        assert jnp.allclose(result.data["x"].value, 1.5)
        assert jnp.allclose(result.data["y"].value, 1.5)

    def test_pos_minus_point_raises_error(self):
        """Pos - Point is not allowed."""
        disp = cx.Vector(
            {"x": u.Q(2.0, "m"), "y": u.Q(3.0, "m"), "z": u.Q(0.0, "m")},
            cx.charts.cart3d,
            cx.roles.pos,
        )
        point = cx.Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")},
            cx.charts.cart3d,
            cx.roles.point,
        )

        with pytest.raises(TypeError, match="Cannot subtract Point from Pos"):
            disp.sub(point)

        with pytest.raises(TypeError):
            disp - point

    def test_cross_representation_subtraction(self):
        """Subtract points in different representations."""
        # Cartesian point at (1, 0, 0)
        p1_cart = cx.Vector.from_([1, 0, 0], "m")

        # Same point in spherical: r=1, theta=π/2, phi=0
        p2_sph = cx.Vector(
            {
                "r": u.Q(1.0, "m"),
                "theta": u.Q(jnp.pi / 2, "rad"),
                "phi": u.Q(0.0, "rad"),
            },
            cx.charts.sph3d,
            cx.roles.point,
        )

        # Subtract - should get ~zero displacement
        result = p1_cart - p2_sph
        assert isinstance(result.role, cx.roles.Pos)
        assert isinstance(result.chart, cx.charts.Cart3D)
        assert jnp.allclose(result.data["x"].value, 0.0, atol=1e-10)
        assert jnp.allclose(result.data["y"].value, 0.0, atol=1e-10)
        assert jnp.allclose(result.data["z"].value, 0.0, atol=1e-10)
