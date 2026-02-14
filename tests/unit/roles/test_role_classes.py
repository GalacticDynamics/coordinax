"""Unit tests for role class hierarchy and basic semantics."""

import pytest
from hypothesis import given, settings

import coordinax.roles as cxr
import coordinax_hypothesis.core as cxst

# ============================================================================
# Concrete role classes and their singletons

ALL_ROLE_CLASSES = (
    cxr.Point,
    cxr.PhysDisp,
    cxr.PhysVel,
    cxr.PhysAcc,
    cxr.CoordDisp,
    cxr.CoordVel,
    cxr.CoordAcc,
)
ALL_SINGLETONS = (
    cxr.point,
    cxr.phys_disp,
    cxr.phys_vel,
    cxr.phys_acc,
    cxr.coord_disp,
    cxr.coord_vel,
    cxr.coord_acc,
)
PHYS_ROLE_CLASSES = (cxr.PhysDisp, cxr.PhysVel, cxr.PhysAcc)
COORD_ROLE_CLASSES = (cxr.CoordDisp, cxr.CoordVel, cxr.CoordAcc)


class TestRoleHierarchy:
    """Test the role class hierarchy follows the abstract-final pattern."""

    @given(role=cxst.roles())
    @settings(deadline=None)
    def test_all_roles_are_abstract_role(self, role: cxr.AbstractRole) -> None:
        """Every drawn role is an AbstractRole instance."""
        assert isinstance(role, cxr.AbstractRole)

    @given(role=cxst.physical_roles())
    @settings(deadline=None)
    def test_physical_roles_are_abstract_phys_role(
        self, role: cxr.AbstractPhysRole
    ) -> None:
        """Physical roles are AbstractPhysRole instances."""
        assert isinstance(role, cxr.AbstractPhysRole)
        assert isinstance(role, cxr.AbstractRole)
        assert not isinstance(role, cxr.Point)
        assert not isinstance(role, cxr.AbstractCoordRole)

    @pytest.mark.parametrize(
        "role",
        [cxr.coord_disp, cxr.coord_vel, cxr.coord_acc],
        ids=["CoordDisp", "CoordVel", "CoordAcc"],
    )
    def test_coord_roles_are_abstract_coord_role(
        self, role: cxr.AbstractCoordRole
    ) -> None:
        """Coordinate roles are AbstractCoordRole instances."""
        assert isinstance(role, cxr.AbstractCoordRole)
        assert isinstance(role, cxr.AbstractRole)
        assert not isinstance(role, cxr.Point)
        assert not isinstance(role, cxr.AbstractPhysRole)

    @given(role=cxst.point_role())
    @settings(deadline=None)
    def test_point_is_not_phys_or_coord(self, role: cxr.Point) -> None:
        """Point is neither a physical tangent role nor a coordinate role."""
        assert isinstance(role, cxr.Point)
        assert isinstance(role, cxr.AbstractRole)
        assert not isinstance(role, cxr.AbstractPhysRole)
        assert not isinstance(role, cxr.AbstractCoordRole)


class TestSingletons:
    """Test that role singletons match their classes."""

    @pytest.mark.parametrize(
        ("singleton", "cls"),
        [
            (cxr.point, cxr.Point),
            (cxr.phys_disp, cxr.PhysDisp),
            (cxr.phys_vel, cxr.PhysVel),
            (cxr.phys_acc, cxr.PhysAcc),
            (cxr.coord_disp, cxr.CoordDisp),
            (cxr.coord_vel, cxr.CoordVel),
            (cxr.coord_acc, cxr.CoordAcc),
        ],
    )
    def test_singleton_is_instance(
        self, singleton: cxr.AbstractRole, cls: type
    ) -> None:
        """Each singleton is an instance of its corresponding class."""
        assert isinstance(singleton, cls)

    @pytest.mark.parametrize(
        ("singleton", "expected_order"),
        [
            (cxr.point, 0),
            (cxr.phys_disp, 0),
            (cxr.phys_vel, 1),
            (cxr.phys_acc, 2),
            (cxr.coord_disp, 0),
            (cxr.coord_vel, 1),
            (cxr.coord_acc, 2),
        ],
    )
    def test_singleton_order(
        self, singleton: cxr.AbstractRole, expected_order: int
    ) -> None:
        """Order attribute matches the time-derivative order."""
        assert singleton.order == expected_order


class TestOrder:
    """Test the order class variable for all roles."""

    @given(role=cxst.roles())
    @settings(deadline=None)
    def test_order_is_nonneg_int(self, role: cxr.AbstractRole) -> None:
        """Order is a non-negative integer."""
        assert isinstance(role.order, int)
        assert role.order >= 0

    @pytest.mark.parametrize(
        "role",
        [cxr.point, cxr.phys_disp, cxr.coord_disp],
        ids=["Point", "PhysDisp", "CoordDisp"],
    )
    def test_displacement_like_order_zero(self, role: cxr.AbstractRole) -> None:
        """Point, PhysDisp, and CoordDisp have order 0."""
        assert role.order == 0

    @pytest.mark.parametrize(
        "role",
        [cxr.phys_vel, cxr.coord_vel],
        ids=["PhysVel", "CoordVel"],
    )
    def test_velocity_like_order_one(self, role: cxr.AbstractRole) -> None:
        """PhysVel and CoordVel have order 1."""
        assert role.order == 1

    @pytest.mark.parametrize(
        "role",
        [cxr.phys_acc, cxr.coord_acc],
        ids=["PhysAcc", "CoordAcc"],
    )
    def test_acceleration_like_order_two(self, role: cxr.AbstractRole) -> None:
        """PhysAcc and CoordAcc have order 2."""
        assert role.order == 2
