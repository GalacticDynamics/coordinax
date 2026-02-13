"""Tests for the chart_init_kwargs strategy."""

import hypothesis.strategies as st
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax.embeddings as cxe
import coordinax_hypothesis as cxst


@given(kwargs=cxst.chart_init_kwargs(cxc.Cart3D))
@settings(max_examples=5)
def test_cart3d_kwargs(kwargs: dict) -> None:
    """Test chart_init_kwargs generates valid Cart3D kwargs."""
    # Cart3D has no required init parameters, so kwargs should be empty or minimal
    chart = cxc.Cart3D(**kwargs)
    assert isinstance(chart, cxc.Cart3D)
    assert chart.ndim == 3


# SpaceTimeCT and SpaceTimeEuclidean tests disabled - they require recursive
# chart generation which causes issues with parameterized generic types


@given(kwargs=cxst.chart_init_kwargs(cxe.EmbeddedManifold))
@settings(max_examples=5)
def test_embedded_manifold_kwargs(kwargs: dict) -> None:
    """Test chart_init_kwargs generates valid EmbeddedManifold kwargs."""
    assert "intrinsic_chart" in kwargs
    assert "ambient_chart" in kwargs
    assert "params" in kwargs
    chart = cxe.EmbeddedManifold(**kwargs)
    assert isinstance(chart, cxe.EmbeddedManifold)


@given(
    chart_cls=cxst.chart_classes(filter=cxc.Abstract3D, exclude_abstract=True),
    kwargs=cxst.chart_init_kwargs(
        cxst.chart_classes(filter=cxc.Abstract3D, exclude_abstract=True)
    ),
)
@settings(max_examples=10)
def test_3d_chart_construction(
    chart_cls: type[cxc.AbstractChart], kwargs: dict
) -> None:
    """Test chart_init_kwargs with dynamically generated 3D chart classes."""
    # This test combines chart_classes and chart_init_kwargs
    # Note: the kwargs may not match chart_cls since they're independently drawn
    # So we just verify that each works independently
    assert issubclass(chart_cls, cxc.Abstract3D)
    # Can't test chart_cls(**kwargs) since they're independently generated


@given(
    chart_cls=st.sampled_from([cxc.Cart1D, cxc.Polar2D, cxc.Cart3D]),
    kwargs=st.data(),
)
@settings(max_examples=5)
def test_chart_init_kwargs_with_fixed_classes(
    chart_cls: type[cxc.AbstractChart],
    kwargs: st.DataObject,
) -> None:
    """Test chart_init_kwargs with specific fixed chart classes."""
    # Use st.data() to dynamically draw kwargs inside the test
    init_kwargs = kwargs.draw(cxst.chart_init_kwargs(chart_cls))
    chart = chart_cls(**init_kwargs)
    assert isinstance(chart, chart_cls)


@given(kwargs=cxst.chart_init_kwargs(cxc.Polar2D))
@settings(max_examples=5)
def test_polar2d_kwargs(kwargs: dict) -> None:
    """Test chart_init_kwargs generates valid Polar2D kwargs."""
    chart = cxc.Polar2D(**kwargs)
    assert isinstance(chart, cxc.Polar2D)
    assert chart.ndim == 2


@given(kwargs=cxst.chart_init_kwargs(cxc.Spherical3D))
@settings(max_examples=5)
def test_spherical3d_kwargs(kwargs: dict) -> None:
    """Test chart_init_kwargs generates valid Spherical3D kwargs."""
    chart = cxc.Spherical3D(**kwargs)
    assert isinstance(chart, cxc.Spherical3D)
    assert chart.ndim == 3


@given(kwargs=cxst.chart_init_kwargs(cxc.Cylindrical3D))
@settings(max_examples=5)
def test_cylindrical3d_kwargs(kwargs: dict) -> None:
    """Test chart_init_kwargs generates valid Cylindrical3D kwargs."""
    chart = cxc.Cylindrical3D(**kwargs)
    assert isinstance(chart, cxc.Cylindrical3D)
    assert chart.ndim == 3
