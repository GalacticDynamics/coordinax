"""Unit tests for DIM_TO_ROLE_MAP — dimension-to-role mapping."""

import unxt as u

import coordinax.roles as cxr
from coordinax.roles._src import DIM_TO_ROLE_MAP


class TestDimToRoleMap:
    """Test the DIM_TO_ROLE_MAP constant."""

    def test_length_maps_to_point(self) -> None:
        """Length dimension maps to Point."""
        assert DIM_TO_ROLE_MAP[u.dimension("length")] is cxr.Point

    def test_angle_maps_to_point(self) -> None:
        """Angle dimension maps to Point."""
        assert DIM_TO_ROLE_MAP[u.dimension("angle")] is cxr.Point

    def test_speed_maps_to_phys_vel(self) -> None:
        """Speed dimension maps to PhysVel."""
        assert DIM_TO_ROLE_MAP[u.dimension("speed")] is cxr.PhysVel

    def test_angular_speed_maps_to_phys_vel(self) -> None:
        """Angular speed dimension maps to PhysVel."""
        assert DIM_TO_ROLE_MAP[u.dimension("angular speed")] is cxr.PhysVel

    def test_acceleration_maps_to_phys_acc(self) -> None:
        """Acceleration dimension maps to PhysAcc."""
        assert DIM_TO_ROLE_MAP[u.dimension("acceleration")] is cxr.PhysAcc

    def test_angular_acceleration_maps_to_phys_acc(self) -> None:
        """Angular acceleration dimension maps to PhysAcc."""
        assert DIM_TO_ROLE_MAP[u.dimension("angular acceleration")] is cxr.PhysAcc

    def test_map_only_contains_expected_entries(self) -> None:
        """The map contains exactly the 6 expected entries."""
        assert len(DIM_TO_ROLE_MAP) == 6

    def test_map_values_are_role_classes(self) -> None:
        """All map values are concrete role classes (not instances)."""
        for v in DIM_TO_ROLE_MAP.values():
            assert isinstance(v, type)
            assert issubclass(v, cxr.AbstractRole)
