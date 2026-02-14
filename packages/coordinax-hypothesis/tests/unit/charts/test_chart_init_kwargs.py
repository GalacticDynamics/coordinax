"""Tests for the chart_init_kwargs strategy."""

import hypothesis.strategies as st
from hypothesis import example, given

import unxt as u

import coordinax.charts as cxc
import coordinax_hypothesis.core as cxst


@given(kwargs=cxst.chart_init_kwargs(chart_class=cxst.chart_classes()))
@example(kwargs={})
def test_chart_init_kwargs_exists(kwargs: dict) -> None:
    """Test chart_init_kwargs generates kwargs.

    This test will unearth any fatal errors in the strategy itself such that it
    fails to generate any kwargs at all. It does not verify the correctness of
    the generated kwargs, just that they can be generated.
    """
    assert isinstance(kwargs, dict)


@st.composite
def _chart_class_with_kwargs(draw):
    """Draw a (chart_class, kwargs) pair where kwargs are valid for that class."""
    chart_class = draw(cxst.chart_classes())
    kwargs = draw(cxst.chart_init_kwargs(chart_class))
    return (chart_class, kwargs)


@given(pair=_chart_class_with_kwargs())
@example(pair=(cxc.Cart0D, {}))
@example(pair=(cxc.Cart1D, {}))
@example(pair=(cxc.Cart2D, {}))
@example(pair=(cxc.Polar2D, {}))
@example(pair=(cxc.Cart3D, {}))
@example(pair=(cxc.Cylindrical3D, {}))
@example(pair=(cxc.Spherical3D, {}))
@example(pair=(cxc.ProlateSpheroidal3D, {"Delta": u.StaticQuantity(1.0, "kpc")}))
@example(pair=(cxc.PoincarePolar6D, {}))
@example(pair=(cxc.CartND, {}))
def test_chart_init_kwargs_instantiates(pair) -> None:
    """Test chart_init_kwargs instantiates charts."""
    chart_class, kwargs = pair
    chart = chart_class(**kwargs)
    assert isinstance(chart, cxc.AbstractChart)
