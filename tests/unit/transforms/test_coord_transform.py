"""Property tests for coord_transform."""

from hypothesis import HealthCheck, given, reject, settings, strategies as st

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.embeddings as cxe
import coordinax_hypothesis.core as cxst
from coordinax.api import CsDict

COORD_TRANSFORM_SUPPORTED: dict[int, tuple[cxc.AbstractChart, ...]] = {
    1: (cxc.cart1d, cxc.radial1d),
    2: (cxc.cart2d, cxc.polar2d),
    3: (
        cxc.cart3d,
        cxc.cyl3d,
        cxc.sph3d,
        cxc.lonlatsph3d,
        cxc.loncoslatsph3d,
        cxc.mathsph3d,
    ),
}


def point_ok(chart: cxc.AbstractChart, point: CsDict, /) -> bool:
    try:
        chart.check_data(point)
    except Exception:  # noqa: BLE001
        return False
    return True


@st.composite
def chart_pairs_same_dim(
    draw: st.DrawFn,
) -> tuple[cxc.AbstractChart, cxc.AbstractChart]:
    dim = draw(st.sampled_from(sorted(COORD_TRANSFORM_SUPPORTED)))
    choices = COORD_TRANSFORM_SUPPORTED[dim]
    return draw(st.sampled_from(choices)), draw(st.sampled_from(choices))


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.filter_too_much]
)
@given(chart=cxst.charts(exclude=(cxe.EmbeddedManifold, cxc.TwoSphere)), data=st.data())
def test_coord_transform_identity(chart, data):
    at = data.draw(cxst.cdicts(chart, cxr.point))
    if not point_ok(chart, at):
        reject()
    x = data.draw(cxst.cdicts(chart, cxr.coord_vel))
    y = cxt.coord_transform(chart, chart, x, at=at)
    for k in chart.components:
        assert jnp.allclose(u.ustrip(x[k].unit, y[k]), u.ustrip(x[k].unit, x[k]))


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.filter_too_much]
)
@given(charts_pair=chart_pairs_same_dim(), data=st.data())
def test_coord_transform_roundtrip_velocity(charts_pair, data):
    chart_a, chart_b = charts_pair
    at_b = data.draw(cxst.cdicts(chart_b, cxr.point))
    if not point_ok(chart_b, at_b):
        reject()
    vel_elems = st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    x_b = data.draw(cxst.cdicts(chart_b, cxr.coord_vel, elements=vel_elems))

    at_a = cxt.point_transform(chart_a, chart_b, at_b)
    if not point_ok(chart_a, at_a):
        reject()
    x_a = cxt.coord_transform(chart_a, chart_b, x_b, at=at_b)
    x_b2 = cxt.coord_transform(chart_b, chart_a, x_a, at=at_a)

    for k in chart_b.components:
        assert jnp.allclose(
            u.ustrip(x_b[k].unit, x_b2[k]),
            u.ustrip(x_b[k].unit, x_b[k]),
            rtol=1e-5,
            atol=1e-3,
        )


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.filter_too_much]
)
@given(
    charts_pair=chart_pairs_same_dim(),
    data=st.data(),
    a=st.floats(-2, 2),
    b=st.floats(-2, 2),
)
def test_coord_transform_linearity(charts_pair, data, a, b):
    chart_a, chart_b = charts_pair
    at_b = data.draw(cxst.cdicts(chart_b, cxr.point))
    if not point_ok(chart_b, at_b):
        reject()
    at_a = cxt.point_transform(chart_a, chart_b, at_b)
    if not point_ok(chart_a, at_a):
        reject()
    vel_elems = st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    x = data.draw(cxst.cdicts(chart_b, cxr.coord_vel, elements=vel_elems))
    y = data.draw(cxst.cdicts(chart_b, cxr.coord_vel, elements=vel_elems))

    def lincomb(p, q):
        return {k: a * p[k] + b * q[k] for k in p}

    lhs = cxt.coord_transform(chart_a, chart_b, lincomb(x, y), at=at_b)
    rhs_x = cxt.coord_transform(chart_a, chart_b, x, at=at_b)
    rhs_y = cxt.coord_transform(chart_a, chart_b, y, at=at_b)
    rhs = lincomb(rhs_x, rhs_y)

    for k in chart_a.components:
        assert jnp.allclose(
            u.ustrip(rhs[k].unit, lhs[k]),
            u.ustrip(rhs[k].unit, rhs[k]),
            rtol=1e-5,
            atol=1e-5,
        )
