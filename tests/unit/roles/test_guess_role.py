"""Unit tests for guess_role() — role inference from dimensions and quantities."""

import pytest
from hypothesis import given, settings

import unxt as u
import unxt_hypothesis as ust

import coordinax.roles as cxr


class TestGuessRoleFromDimension:
    """Test guess_role(dimension) -> role inference."""

    @pytest.mark.parametrize(
        ("dim_name", "expected_type"),
        [
            ("length", cxr.Point),
            ("angle", cxr.Point),
            ("speed", cxr.PhysVel),
            ("angular speed", cxr.PhysVel),
            ("acceleration", cxr.PhysAcc),
            ("angular acceleration", cxr.PhysAcc),
        ],
    )
    def test_known_dimensions(
        self, dim_name: str, expected_type: type[cxr.AbstractRole]
    ) -> None:
        """Known physical dimensions map to the correct role type."""
        dim = u.dimension(dim_name)
        role = cxr.guess_role(dim)
        assert isinstance(role, expected_type)

    def test_unknown_dimension_raises(self) -> None:
        """Unmapped dimensions raise ValueError."""
        dim = u.dimension("mass")
        with pytest.raises(ValueError, match="Cannot infer role"):
            cxr.guess_role(dim)


class TestGuessRoleFromQuantity:
    """Test guess_role(quantity) -> role inference from Quantity unit."""

    @given(q=ust.quantities(unit=u.unit("m"), shape=()))
    @settings(deadline=None)
    def test_length_quantity_gives_point(self, q: u.AbstractQuantity) -> None:
        """Quantities with length units yield Point role."""
        role = cxr.guess_role(q)
        assert isinstance(role, cxr.Point)

    @given(q=ust.quantities(unit=u.unit("m/s"), shape=()))
    @settings(deadline=None)
    def test_speed_quantity_gives_phys_vel(self, q: u.AbstractQuantity) -> None:
        """Quantities with speed units yield PhysVel role."""
        role = cxr.guess_role(q)
        assert isinstance(role, cxr.PhysVel)

    @given(q=ust.quantities(unit=u.unit("m/s2"), shape=()))
    @settings(deadline=None)
    def test_acceleration_quantity_gives_phys_acc(self, q: u.AbstractQuantity) -> None:
        """Quantities with acceleration units yield PhysAcc role."""
        role = cxr.guess_role(q)
        assert isinstance(role, cxr.PhysAcc)

    @given(q=ust.quantities(unit=u.unit("rad"), shape=()))
    @settings(deadline=None)
    def test_angle_quantity_gives_point(self, q: u.AbstractQuantity) -> None:
        """Quantities with angle units yield Point role."""
        role = cxr.guess_role(q)
        assert isinstance(role, cxr.Point)

    @given(q=ust.quantities(unit=u.unit("rad/s"), shape=()))
    @settings(deadline=None)
    def test_angular_speed_quantity_gives_phys_vel(self, q: u.AbstractQuantity) -> None:
        """Quantities with angular speed units yield PhysVel role."""
        role = cxr.guess_role(q)
        assert isinstance(role, cxr.PhysVel)


class TestGuessRoleFromCsDict:
    """Test guess_role(csdict) -> role inference from component dictionaries."""

    def test_length_csdict_gives_point(self) -> None:
        """A dict of length quantities yields Point role."""
        d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        role = cxr.guess_role(d)
        assert isinstance(role, cxr.Point)

    def test_speed_csdict_gives_phys_vel(self) -> None:
        """A dict of speed quantities yields PhysVel role."""
        d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
        role = cxr.guess_role(d)
        assert isinstance(role, cxr.PhysVel)

    def test_acceleration_csdict_gives_phys_acc(self) -> None:
        """A dict of acceleration quantities yields PhysAcc role."""
        d = {
            "x": u.Q(1.0, "m/s2"),
            "y": u.Q(2.0, "m/s2"),
            "z": u.Q(3.0, "m/s2"),
        }
        role = cxr.guess_role(d)
        assert isinstance(role, cxr.PhysAcc)

    def test_mixed_dimensions_raises(self) -> None:
        """A dict with mixed dimensions raises ValueError."""
        d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m/s")}
        with pytest.raises(ValueError, match="Cannot infer role from mixed dimensions"):
            cxr.guess_role(d)

    def test_unknown_dimension_csdict_raises(self) -> None:
        """A dict with an unknown dimension raises ValueError."""
        d = {"x": u.Q(1.0, "kg"), "y": u.Q(2.0, "kg")}
        with pytest.raises(ValueError, match="Cannot infer role"):
            cxr.guess_role(d)
