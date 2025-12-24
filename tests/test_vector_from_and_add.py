"""Tests for Vector.from_ and Vector + Quantity operations."""

import pytest

import unxt as u

import coordinax as cx


class TestVectorFromQuantity:
    """Test Vector.from_ with Quantity argument."""

    def test_from_quantity_defaults_to_point(self):
        """Test that Vector.from_(Quantity) defaults to Point role."""
        # Length quantity should default to Point
        vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
        assert isinstance(vec.role, cx.roles.Point)
        assert vec.chart == cx.charts.cart3d

    def test_from_quantity_velocity_infers_vel(self):
        """Velocity quantity infers Vel role."""
        vec = cx.Vector.from_(u.Q([1, 2, 3], "m/s"))
        assert isinstance(vec.role, cx.roles.Vel)

    def test_from_quantity_with_explicit_pos_role(self):
        """Test Vector.from_(Quantity, role=Pos) with valid dimension."""
        vec = cx.Vector.from_(u.Q([1, 2, 3], "m"), cx.roles.pos)
        assert isinstance(vec.role, cx.roles.Pos)

    def test_from_quantity_with_explicit_vel_role(self):
        """Test Vector.from_(Quantity, role=Vel) with valid dimension."""
        vec = cx.Vector.from_(u.Q([1, 2, 3], "m/s"), cx.roles.vel)
        assert isinstance(vec.role, cx.roles.Vel)

    def test_from_quantity_with_explicit_acc_role(self):
        """Test Vector.from_(Quantity, role=Acc) with valid dimension."""
        vec = cx.Vector.from_(u.Q([1, 2, 3], "m/s2"), cx.roles.acc)
        assert isinstance(vec.role, cx.roles.Acc)

    def test_from_quantity_pos_requires_length(self):
        """Test that Pos role requires length dimension."""
        with pytest.raises(ValueError, match="Pos role requires dimension=length"):
            cx.Vector.from_(u.Q([1, 2, 3], "m/s"), cx.roles.pos)

    def test_from_quantity_vel_requires_speed(self):
        """Test that Vel role requires speed dimension."""
        with pytest.raises(ValueError, match="Vel role requires dimension=speed"):
            cx.Vector.from_(u.Q([1, 2, 3], "m"), cx.roles.vel)

    def test_from_quantity_acc_requires_acceleration(self):
        """Test that Acc role requires acceleration dimension."""
        with pytest.raises(
            ValueError, match="Acc role requires dimension=acceleration"
        ):
            cx.Vector.from_(u.Q([1, 2, 3], "m"), cx.roles.acc)

    def test_from_quantity_with_chart_and_explicit_role(self):
        """Test Vector.from_(Quantity, chart, role) works."""
        vec = cx.Vector.from_(u.Q([1, 2, 3], "m"), cx.charts.cart3d, cx.roles.pos)
        assert isinstance(vec.role, cx.roles.Pos)
        assert vec.chart == cx.charts.cart3d

    def test_from_quantity_with_chart_defaults_to_point(self):
        """Test Vector.from_(Quantity, chart) defaults to Point."""
        vec = cx.Vector.from_(u.Q([1, 2, 3], "m"), cx.charts.cart3d)
        assert isinstance(vec.role, cx.roles.Point)
        assert vec.chart == cx.charts.cart3d

    def test_from_quantity_unknown_dimension_errors(self):
        """Quantities with unknown dimension should raise on role inference."""
        with pytest.raises(ValueError, match="Cannot infer Vector role from quantity"):
            cx.Vector.from_(u.Q([1, 2, 3], "kg"))


class TestVectorAddQuantity:
    """Test Vector + Quantity operations via quax dispatch."""

    def test_point_add_length_quantity(self):
        """Test Point + length Quantity desugars to Point + Pos."""
        point = cx.Vector.from_(u.Q([0, 0, 0], "m"), cx.roles.point)
        qty = u.Q([1, 2, 3], "m")

        result = point + qty

        # Point + Pos -> Point
        assert isinstance(result.role, cx.roles.Point)
        assert result["x"] == u.Q(1, "m")
        assert result["y"] == u.Q(2, "m")
        assert result["z"] == u.Q(3, "m")

    def test_pos_add_length_quantity(self):
        """Test Pos + length Quantity desugars to Pos + Pos."""
        pos = cx.Vector.from_(u.Q([1, 0, 0], "m"), cx.roles.pos)
        qty = u.Q([0, 2, 0], "m")

        result = pos + qty

        # Pos + Pos -> Pos
        assert isinstance(result.role, cx.roles.Pos)
        assert result["x"] == u.Q(1, "m")
        assert result["y"] == u.Q(2, "m")

    def test_vector_add_vector(self):
        """Test that Vector + Vector still works (not broken by new dispatches)."""
        p1 = cx.Vector.from_(u.Q([1, 2, 3], "m"), cx.roles.point)
        pos = cx.Vector.from_(u.Q([0.5, 0.5, 0.5], "m"), cx.roles.pos)

        result = p1 + pos

        assert isinstance(result.role, cx.roles.Point)
        assert result["x"] == u.Q(1.5, "m")
        assert result["y"] == u.Q(2.5, "m")
        assert result["z"] == u.Q(3.5, "m")


class TestVectorFromMappingWithRole:
    """Test Vector.from_ with mapping and inferred role."""

    def test_from_mapping_infers_pos_from_length(self):
        """Test that Mapping.from_ with length components infers Point role."""
        data = {
            "x": u.Q(1, "m"),
            "y": u.Q(2, "m"),
            "z": u.Q(3, "m"),
        }
        vec = cx.Vector.from_(data, cx.charts.cart3d)
        assert isinstance(vec.role, cx.roles.Point)

    def test_from_mapping_infers_vel_from_velocity(self):
        """Test that Mapping.from_ with velocity components infers Vel role."""
        data = {
            "x": u.Q(1, "m/s"),
            "y": u.Q(2, "m/s"),
            "z": u.Q(3, "m/s"),
        }
        vec = cx.Vector.from_(data, cx.charts.cart3d)
        assert isinstance(vec.role, cx.roles.Vel)

    def test_from_mapping_with_explicit_role(self):
        """Test that explicit role overrides inferred role."""
        data = {
            "x": u.Q(1, "m"),
            "y": u.Q(2, "m"),
            "z": u.Q(3, "m"),
        }
        vec = cx.Vector.from_(data, cx.charts.cart3d, cx.roles.point)
        assert isinstance(vec.role, cx.roles.Point)
