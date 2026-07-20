"""Tests for representation-aware CDict strategies."""

import pytest
import unxt as u
from hypothesis import given, strategies as st

import coordinax.charts as cxc
import coordinax.representations as cxr

import coordinaxs.hypothesis.main as cxst
import coordinaxs.hypothesis.representations as cxrst


class FakeBasis(cxr.AbstractBasis):
    """Concrete test basis used to validate point-geometry constraints."""


class FakeSemantic(cxr.AbstractSemanticKind):
    """Concrete test semantic used to validate point-geometry constraints."""

    @classmethod
    def coord_dimensions(cls, chart, /):
        return tuple(None for _ in chart.components)


@given(p=cxrst.cdicts(cxc.cart3d, cxst.representations()))
def test_cdicts_accepts_representation_strategy(p):
    """Cdicts should accept a representation strategy as the second argument."""
    assert set(p.keys()) == set(cxc.cart3d.components)


@given(p=cxrst.cdicts(cxc.sph3d, cxr.point))
def test_cdicts_accepts_representation_instance(p):
    """Cdicts should accept a concrete Representation instance."""
    assert set(p.keys()) == {"r", "theta", "phi"}
    assert u.dimension_of(p["r"]) == u.dimension("length")
    assert u.dimension_of(p["theta"]) == u.dimension("angle")
    assert u.dimension_of(p["phi"]) == u.dimension("angle")


@given(data=st.data())
def test_point_geometry_requires_no_basis(data):
    """PointGeometry cdicts should reject basis kinds other than NoBasis."""
    with pytest.raises(TypeError, match="NoBasis"):
        data.draw(
            cxrst.cdicts(cxc.cart3d, cxr.PointGeometry(), FakeBasis(), cxr.Location())
        )


@given(data=st.data())
def test_point_geometry_requires_location_semantic(data):
    """PointGeometry cdicts should reject semantic kinds other than Location."""
    with pytest.raises(TypeError, match="Location semantic kind"):
        data.draw(
            cxrst.cdicts(cxc.cart3d, cxr.PointGeometry(), cxr.NoBasis(), FakeSemantic())
        )


@given(p=cxrst.cdicts(cxc.cart3d, cxr.coord_disp))
def test_cdicts_tangent_coord_disp(p):
    """Cdicts with coord_disp (TangentGeometry, CoordinateBasis, Displacement) returns chart components."""
    assert set(p.keys()) == {"x", "y", "z"}


@given(p=cxrst.cdicts(cxc.sph3d, cxr.phys_vel))
def test_cdicts_tangent_phys_vel(p):
    """Cdicts with phys_vel (TangentGeometry, PhysicalBasis, Velocity) returns chart components."""
    assert set(p.keys()) == {"r", "theta", "phi"}


@given(
    p=cxrst.cdicts(cxc.cart3d, cxst.representations(geom_kind=cxr.TangentGeometry()))
)
def test_cdicts_tangent_with_rep_strategy(p):
    """Cdicts with a TangentGeometry representation strategy returns chart components."""
    assert set(p.keys()) == set(cxc.cart3d.components)
