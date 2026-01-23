"""Tests for coordinax-hypothesis strategies."""

import pytest
from hypothesis import given

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.roles as cxr
import coordinax_hypothesis as cxst


class TestCsDictStrategy:
    """Test pdict strategy for generating valid CsDict objects."""

    @given(p=cxst.pdicts(cxc.cart3d, cxr.point))
    def test_pdict_keys_match_chart(self, p):
        """CsDict keys must exactly match chart.components."""
        assert set(p.keys()) == set(cxc.cart3d.components)

    @given(p=cxst.pdicts(cxc.cart3d, cxr.point))
    def test_pdict_all_quantities(self, p):
        """All values must be quantity-like."""
        for v in p.values():
            assert hasattr(v, "unit") or isinstance(v, (int, float))

    @given(p=cxst.pdicts(cxc.cart3d, cxr.phys_disp))
    def test_pdict_pos_uniform_dimension(self, p):
        """Pos role requires uniform length dimension across components."""
        for v in p.values():
            assert u.dimension_of(v) == u.dimension("length")

    @given(p=cxst.pdicts(cxc.cart3d, cxr.phys_vel))
    def test_pdict_vel_uniform_dimension(self, p):
        """Vel role requires uniform length/time dimension across components."""
        expected_dim = u.dimension("length") / u.dimension("time")
        for v in p.values():
            assert u.dimension_of(v) == expected_dim

    @given(p=cxst.pdicts(cxc.cart3d, cxr.phys_acc))
    def test_pdict_acc_uniform_dimension(self, p):
        """Acc role requires uniform length/time^2 dimension across components."""
        expected_dim = u.dimension("length") / (u.dimension("time") ** 2)
        for v in p.values():
            assert u.dimension_of(v) == expected_dim

    @given(p=cxst.pdicts(cxc.sph3d, cxr.point))
    def test_pdict_mixed_dimensions(self, p):
        """Point role allows mixed dimensions from chart.coord_dimensions."""
        # Spherical has (length, angle, angle)
        assert set(p.keys()) == {"r", "theta", "phi"}
        assert u.dimension_of(p["r"]) == u.dimension("length")
        assert u.dimension_of(p["theta"]) == u.dimension("angle")
        assert u.dimension_of(p["phi"]) == u.dimension("angle")


class TestRoleStrategies:
    """Test role generation strategies."""

    @given(role=cxst.roles())
    def test_roles_returns_role_instance(self, role):
        """roles() must return role instances."""
        assert isinstance(role, cxr.AbstractRole)

    @given(role=cxst.physical_roles())
    def test_physical_roles_only_tangent(self, role):
        """physical_roles() must return only PhysDisp, Vel, or PhysAcc."""
        assert isinstance(role, (cxr.PhysDisp, cxr.PhysVel, cxr.PhysAcc))
        assert not isinstance(role, cxr.Point)

    @given(role=cxst.point_role())
    def test_point_role_only_point(self, role):
        """point_role() must return only Point."""
        assert isinstance(role, cxr.Point)

    @given(role=cxst.roles(include=(cxr.PhysDisp, cxr.PhysVel)))
    def test_roles_include_filter(self, role):
        """roles(include=...) must only return included roles."""
        assert isinstance(role, (cxr.PhysDisp, cxr.PhysVel))

    @given(role=cxst.roles(exclude=(cxr.Point,)))
    def test_roles_exclude_filter(self, role):
        """roles(exclude=...) must not return excluded roles."""
        assert not isinstance(role, cxr.Point)


class TestVectorStrategies:
    """Test vector generation strategies."""

    @given(vec=cxst.vectors(chart=cxc.cart3d, role=cxr.phys_disp))
    def test_vector_construction(self, vec):
        """vectors() must produce valid Vector instances."""
        assert isinstance(vec, cx.Vector)
        assert vec.chart == cxc.cart3d
        assert isinstance(vec.role, cxr.PhysDisp)

    @given(vec=cxst.vectors())
    def test_vector_data_keys(self, vec):
        """Vector.data keys must match chart.components."""
        assert set(vec.data.keys()) == set(vec.chart.components)

    @given(vec=cxst.vectors(role=cxr.phys_disp))
    def test_vector_pos_uniform_dimension(self, vec):
        """Pos vectors must have uniform length dimension."""
        for v in vec.data.values():
            assert u.dimension_of(v) == u.dimension("length")

    @given(
        vec_reps=cxst.vectors_with_target_chart(chart=cxc.cart3d, role=cxr.phys_disp)
    )
    def test_vectors_with_target_chart(self, vec_reps):
        """vectors_with_target_chart() returns (vector, target_chain)."""
        vec, target_chain = vec_reps
        assert isinstance(vec, cx.Vector)
        assert isinstance(target_chain, tuple)
        assert all(isinstance(c, cxc.AbstractChart) for c in target_chain)


class TestPointedVectorStrategy:
    """Test fiber point / bundle generation."""

    @given(fp=cxst.fiber_points(base_chart=cxc.cart3d))
    def test_fiber_point_construction(self, fp):
        """fiber_points() must produce valid PointedVector instances."""
        assert isinstance(fp, cx.PointedVector)
        assert isinstance(fp.base.role, cxr.Point)

    @given(
        fp=cxst.fiber_points(
            base_chart=cxc.cart2d,
            field_keys=("velocity", "acceleration"),
            field_roles=(cxr.PhysVel, cxr.PhysAcc),
        )
    )
    def test_fiber_point_custom_fields(self, fp):
        """fiber_points() with custom fields must include them."""
        assert "velocity" in fp
        assert "acceleration" in fp

    @given(
        fp=cxst.fiber_points(
            base_chart=cxc.cart1d,
            field_keys=("velocity",),
            field_roles=(cxr.PhysVel,),
        )
    )
    def test_fiber_point_custom_chart(self, fp):
        """fiber_points() must respect base_chart parameter."""
        # Base chart should be cart1d
        assert fp.base.chart == cxc.cart1d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
