"""Tests for the vectors strategy."""

import pytest
from hypothesis import given, strategies as st

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinax.representations as cxr
import coordinax.vectors as cxv

import coordinax.hypothesis.charts as cxcst
import coordinax.hypothesis.representations as cxsr
from coordinax.hypothesis.vectors import vectors as vector_strategy

SUPPORTED_CHARTS = cxcst.charts(exclude=(cxc.CartND, cxc.SpaceTimeCT, cxc.Time1D))

# Shared float32 element strategies (width=32 matches the default JAX dtype).
_F32_POS = st.floats(min_value=1.0, max_value=100.0, width=32)
_F32_NEG = st.floats(min_value=-100.0, max_value=-1.0, width=32)
_F32_BOUNDED = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, width=32)


@given(vec=vector_strategy())
def test_vectors_basic(vec: cxv.Point) -> None:
    """vectors() should generate valid Point instances."""
    assert isinstance(vec, cxv.Point)
    assert set(vec.data.keys()) == set(vec.chart.components)
    assert vec.M.has_chart(vec.chart)


@given(vec=vector_strategy(cxc.cart3d))
def test_vectors_concrete_chart_only(vec: cxv.Point) -> None:
    """vectors(chart) should generate vectors for that chart (rep/manifold inferred)."""
    assert vec.chart is cxc.cart3d
    assert isinstance(vec, cxv.Point)
    assert set(vec.data.keys()) == {"x", "y", "z"}


@given(vec=vector_strategy(SUPPORTED_CHARTS))
def test_vectors_chart_strategy(vec: cxv.Point) -> None:
    """vectors(chart_strategy) should draw a chart and generate a valid point."""
    assert isinstance(vec, cxv.Point)
    assert set(vec.data.keys()) == set(vec.chart.components)
    assert vec.M.has_chart(vec.chart)


@given(vec=vector_strategy(cxc.cart3d, cxr.point))
def test_vectors_concrete_chart_and_rep(vec: cxv.Point) -> None:
    """vectors(chart, rep) should fix chart and rep on the generated point."""
    assert vec.chart is cxc.cart3d
    assert vec.rep == cxr.point
    assert set(vec.data.keys()) == {"x", "y", "z"}


@given(vec=vector_strategy(SUPPORTED_CHARTS, cxr.point))
def test_vectors_chart_strategy_concrete_rep(vec: cxv.Point) -> None:
    """vectors(chart_strategy, rep) should draw a chart then fix the rep."""
    assert vec.rep == cxr.point
    assert isinstance(vec, cxv.Point)
    assert vec.M.has_chart(vec.chart)


@given(vec=vector_strategy(cxc.cart3d, cxsr.representations(check_valid=True)))
def test_vectors_concrete_chart_rep_strategy(vec: cxv.Point) -> None:
    """vectors(chart, rep_strategy) should draw a rep then build the point."""
    assert vec.chart is cxc.cart3d
    assert isinstance(vec, cxv.Point)


@given(vec=vector_strategy(cxc.cart3d, shape=(5,)))
def test_vectors_propagate_shape(vec: cxv.Point) -> None:
    """**kw forwarding: shape=(...) should be reflected in vec.shape."""
    assert vec.chart is cxc.cart3d
    assert vec.shape == (5,)


@given(data=st.data())
def test_vectors_infer_manifold_from_chart(data: st.DataObject) -> None:
    """When manifold is not given, it should be inferred from the chart."""
    vec = data.draw(vector_strategy(cxc.sph3d, cxr.point))
    assert vec.M == cxm.guess_manifold(cxc.sph3d)


@given(data=st.data())
def test_vectors_explicit_manifold(data: st.DataObject) -> None:
    """vectors(chart, rep, manifold) should preserve the provided manifold."""
    M = cxm.EuclideanManifold(3)
    vec = data.draw(vector_strategy(cxc.cart3d, cxr.point, M))
    assert vec.M is M


@given(data=st.data())
def test_vectors_manifold_strategy(data: st.DataObject) -> None:
    """vectors(chart, rep, manifold_strategy) should draw a manifold."""
    M = cxm.EuclideanManifold(3)
    vec = data.draw(vector_strategy(cxc.cart3d, cxr.point, st.just(M)))
    assert vec.M is M


@given(data=st.data())
def test_vectors_incompatible_manifold_raises(data: st.DataObject) -> None:
    """Passing a manifold that does not support the chart must raise ValueError."""
    with pytest.raises(ValueError, match="support"):
        data.draw(vector_strategy(cxc.cart3d, cxr.point, cxm.HyperSphericalManifold()))


@given(vec=st.from_type(cxv.Point))
def test_vector_from_type_basic(vec: cxv.Point) -> None:
    """from_type(Point) should resolve to the vectors strategy."""
    assert isinstance(vec, cxv.Point)
    assert set(vec.data.keys()) == set(vec.chart.components)
    assert vec.M.has_chart(vec.chart)


class TestPointValueControl:
    """Tests showing how to control the values (e.g. quadrant) of generated points."""

    @given(vec=vector_strategy(cxc.cart2d, cxr.point, elements=_F32_POS))
    def test_first_quadrant_via_elements(self, vec: cxv.Point) -> None:
        """elements= constrains all components to positive values (first quadrant)."""
        assert vec.data["x"].ustrip("m") > 0
        assert vec.data["y"].ustrip("m") > 0

    @given(vec=vector_strategy(cxc.cart2d, cxr.point, elements=_F32_NEG))
    def test_third_quadrant_via_elements(self, vec: cxv.Point) -> None:
        """elements= constrains all components to negative values (third quadrant)."""
        assert vec.data["x"].ustrip("m") < 0
        assert vec.data["y"].ustrip("m") < 0

    @given(vec=vector_strategy(cxc.cart3d, cxr.point, elements=_F32_POS))
    def test_first_octant_via_elements(self, vec: cxv.Point) -> None:
        """elements= constrains all Cartesian 3D components to positive values."""
        assert vec.data["x"].ustrip("m") > 0
        assert vec.data["y"].ustrip("m") > 0
        assert vec.data["z"].ustrip("m") > 0

    @given(data=st.data())
    def test_second_quadrant_per_component(self, data: st.DataObject) -> None:
        """Use st.data() to draw different element ranges per component.

        Second quadrant: x < 0, y > 0.  Each axis gets its own draw so that
        independent sign constraints can be applied.
        """
        vec_x = data.draw(vector_strategy(cxc.cart2d, cxr.point, elements=_F32_NEG))
        vec_y = data.draw(vector_strategy(cxc.cart2d, cxr.point, elements=_F32_POS))

        assert vec_x.data["x"].ustrip("m") < 0
        assert vec_y.data["y"].ustrip("m") > 0

    @given(vec=vector_strategy(cxc.cart2d, cxr.point, elements=_F32_BOUNDED))
    def test_bounded_range(self, vec: cxv.Point) -> None:
        """elements= with explicit bounds keeps all component magnitudes in range."""
        assert -10.0 <= vec.data["x"].ustrip("m") <= 10.0
        assert -10.0 <= vec.data["y"].ustrip("m") <= 10.0
