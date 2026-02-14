"""Tests for coordinax-hypothesis strategies."""

from hypothesis import given

import coordinax.roles as cxr
import coordinax_hypothesis.core as cxst


@given(role=cxst.roles())
def test_roles_returns_role_instance(role):
    """roles() must return role instances."""
    assert isinstance(role, cxr.AbstractRole)


@given(role=cxst.roles(include=(cxr.PhysDisp, cxr.PhysVel)))
def test_roles_include_filter(role):
    """roles(include=...) must only return included roles."""
    assert isinstance(role, (cxr.PhysDisp, cxr.PhysVel))


@given(role=cxst.roles(exclude=(cxr.Point,)))
def test_roles_exclude_filter(role):
    """roles(exclude=...) must not return excluded roles."""
    assert not isinstance(role, cxr.Point)


@given(role=cxst.point_role())
def test_point_role_only_point(role):
    """point_role() must return only Point."""
    assert isinstance(role, cxr.Point)


@given(role=cxst.physical_roles())
def test_physical_roles_only_tangent(role):
    """physical_roles() must return only PhysDisp, Vel, or PhysAcc."""
    assert isinstance(role, (cxr.PhysDisp, cxr.PhysVel, cxr.PhysAcc))
    assert not isinstance(role, cxr.Point)


@given(role=cxst.coord_roles())
def test_coord_roles_only_coord(role):
    """coord_roles() must return only CoordDisp, Vel, or CoordAcc."""
    assert isinstance(role, (cxr.CoordDisp, cxr.CoordVel, cxr.CoordAcc))
    assert not isinstance(role, cxr.Point)
