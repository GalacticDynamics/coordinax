"""Unit tests for role equality, hashing, and repr."""

import pytest
from hypothesis import given, settings

import coordinax.roles as cxr
import coordinax_hypothesis.core as cxst


class TestRoleEquality:
    """Test role equality semantics (type-based identity)."""

    @given(role=cxst.roles())
    @settings(deadline=None)
    def test_role_equals_itself(self, role: cxr.AbstractRole) -> None:
        """A role is equal to itself (reflexivity)."""
        assert role == role  # noqa: PLR0124

    @given(role=cxst.roles())
    @settings(deadline=None)
    def test_role_equals_same_type(self, role: cxr.AbstractRole) -> None:
        """A role is equal to another instance of the same type."""
        other = type(role)()
        assert role == other

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (cxr.point, cxr.phys_disp),
            (cxr.point, cxr.phys_vel),
            (cxr.point, cxr.phys_acc),
            (cxr.point, cxr.coord_disp),
            (cxr.phys_disp, cxr.phys_vel),
            (cxr.phys_disp, cxr.phys_acc),
            (cxr.phys_disp, cxr.coord_disp),
            (cxr.phys_vel, cxr.phys_acc),
            (cxr.phys_vel, cxr.coord_vel),
            (cxr.coord_disp, cxr.coord_vel),
            (cxr.coord_vel, cxr.coord_acc),
        ],
    )
    def test_different_roles_not_equal(
        self, a: cxr.AbstractRole, b: cxr.AbstractRole
    ) -> None:
        """Different role types are not equal."""
        assert a != b

    @given(role=cxst.roles())
    @settings(deadline=None)
    def test_role_not_equal_to_non_role(self, role: cxr.AbstractRole) -> None:
        """Roles are not equal to non-role objects."""
        assert role != "not-a-role"
        assert role != 42
        assert role != None  # noqa: E711


class TestRoleHashing:
    """Test role hashing semantics (hash by type)."""

    @given(role=cxst.roles())
    @settings(deadline=None)
    def test_role_is_hashable(self, role: cxr.AbstractRole) -> None:
        """All roles are hashable."""
        assert isinstance(hash(role), int)

    @given(role=cxst.roles())
    @settings(deadline=None)
    def test_equal_roles_same_hash(self, role: cxr.AbstractRole) -> None:
        """Equal roles have the same hash (hash consistency with __eq__)."""
        other = type(role)()
        assert role == other
        assert hash(role) == hash(other)

    def test_roles_usable_as_dict_keys(self) -> None:
        """Roles can be used as dictionary keys."""
        d = {cxr.point: "position", cxr.phys_vel: "velocity"}
        assert d[cxr.point] == "position"
        assert d[cxr.phys_vel] == "velocity"
        # A fresh instance of the same type also works as a key
        assert d[cxr.Point()] == "position"

    def test_roles_usable_in_sets(self) -> None:
        """Roles can be stored in sets without duplicates."""
        s = {cxr.point, cxr.phys_vel, cxr.Point(), cxr.PhysVel()}
        assert len(s) == 2


class TestRoleRepr:
    """Test role string representation."""

    @pytest.mark.parametrize(
        ("role", "expected_repr"),
        [
            (cxr.point, "Point()"),
            (cxr.phys_disp, "PhysDisp()"),
            (cxr.phys_vel, "PhysVel()"),
            (cxr.phys_acc, "PhysAcc()"),
            (cxr.coord_disp, "CoordDisp()"),
            (cxr.coord_vel, "CoordVel()"),
            (cxr.coord_acc, "CoordAcc()"),
        ],
    )
    def test_repr(self, role: cxr.AbstractRole, expected_repr: str) -> None:
        """Role repr is ClassName()."""
        assert repr(role) == expected_repr
