"""The `charts_like` strategy: draw charts matching a template's shape."""

__all__: tuple[str, ...] = ()

import hypothesis.strategies as st
import pytest
from hypothesis import given

import coordinax.charts as cxc

import coordinaxs.hypothesis.main as cxst

#: (template chart, ndim it implies, base classes every draw must satisfy).
TEMPLATES = [
    pytest.param(cxc.cart3d, 3, (cxc.Abstract3D,), id="cart3d"),
    pytest.param(
        cxc.sph3d, 3, (cxc.Abstract3D, cxc.AbstractSpherical3D), id="spherical3d"
    ),
    pytest.param(cxc.polar2d, 2, (cxc.Abstract2D,), id="polar2d"),
    pytest.param(cxc.radial1d, 1, (cxc.Abstract1D,), id="radial1d"),
    pytest.param(cxc.sph2, 2, (cxc.Abstract2D,), id="sph2"),
]


@pytest.mark.parametrize(("template", "ndim", "bases"), TEMPLATES)
@given(data=st.data())
def test_charts_like_matches_template(
    template: cxc.AbstractChart,
    ndim: int,
    bases: tuple[type, ...],
    data: st.DataObject,
) -> None:
    """Every draw shares the template's dimensionality and base classes."""
    chart = data.draw(cxst.charts_like(template))
    assert chart.ndim == ndim
    for base in bases:
        assert isinstance(chart, base)
