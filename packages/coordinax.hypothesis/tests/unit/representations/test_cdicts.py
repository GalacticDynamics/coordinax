"""Tests for representation-aware CDict strategies."""

import pytest
import unxt as u
from hypothesis import given, strategies as st

import coordinax.charts as cxc
import coordinax.representations as cxr

import coordinax.hypothesis.main as cxst
import coordinax.hypothesis.representations as cxsr


class FakeBasis(cxr.AbstractBasis):
    """Concrete test basis used to validate point-geometry constraints."""


class FakeSemantic(cxr.AbstractSemanticKind):
    """Concrete test semantic used to validate point-geometry constraints."""

    @classmethod
    def coord_dimensions(cls, chart, /):
        return tuple(None for _ in chart.components)


@given(p=cxsr.cdicts(cxc.cart3d, cxst.representations()))
def test_cdicts_accepts_representation_strategy(p):
    """cdicts should accept a representation strategy as the second argument."""
    assert set(p.keys()) == set(cxc.cart3d.components)


@given(p=cxsr.cdicts(cxc.sph3d, cxr.point))
def test_cdicts_accepts_representation_instance(p):
    """cdicts should accept a concrete Representation instance."""
    assert set(p.keys()) == {"r", "theta", "phi"}
    assert u.dimension_of(p["r"]) == u.dimension("length")
    assert u.dimension_of(p["theta"]) == u.dimension("angle")
    assert u.dimension_of(p["phi"]) == u.dimension("angle")


@given(data=st.data())
def test_point_geometry_requires_no_basis(data):
    """PointGeometry cdicts should reject basis kinds other than NoBasis."""
    with pytest.raises(TypeError, match="NoBasis"):
        data.draw(
            cxsr.cdicts(cxc.cart3d, cxr.PointGeometry(), FakeBasis(), cxr.Location())
        )


@given(data=st.data())
def test_point_geometry_requires_location_semantic(data):
    """PointGeometry cdicts should reject semantic kinds other than Location."""
    with pytest.raises(TypeError, match="Location semantic kind"):
        data.draw(
            cxsr.cdicts(cxc.cart3d, cxr.PointGeometry(), cxr.NoBasis(), FakeSemantic())
        )
