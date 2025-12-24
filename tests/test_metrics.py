"""Tests for metric resolution."""

import jax.numpy as jnp

import unxt as u

import coordinax.charts as cxc
import coordinax.metrics as cxm
import coordinax.transforms as cxt


def test_metric_of_euclidean_3d() -> None:
    metric = cxm.metric_of(cxc.cart3d)
    assert isinstance(metric, cxm.EuclideanMetric)
    assert metric.signature == (1, 1, 1)


def test_metric_of_cylindrical_3d() -> None:
    metric = cxm.metric_of(cxc.cyl3d)
    assert isinstance(metric, cxm.EuclideanMetric)
    assert metric.signature == (1, 1, 1)


def test_metric_of_spacetime_euclidean() -> None:
    chart = cxc.SpaceTimeEuclidean(cxc.cyl3d)
    metric = cxm.metric_of(chart)
    assert isinstance(metric, cxm.EuclideanMetric)
    assert metric.signature == (1, 1, 1, 1)


def test_metric_of_spacetime_ct() -> None:
    chart = cxc.SpaceTimeCT(cxc.cyl3d)
    metric = cxm.metric_of(chart)
    assert isinstance(metric, cxm.MinkowskiMetric)
    assert metric.signature == (-1, 1, 1, 1)


def test_metric_of_twosphere() -> None:
    metric = cxm.metric_of(cxc.twosphere)
    assert isinstance(metric, cxm.SphereMetric)
    assert metric.signature == (1, 1)


def test_minkowski_inner_product_invariant() -> None:
    chart_from = cxc.SpaceTimeCT(cxc.cart3d)
    chart_to = cxc.SpaceTimeCT(cxc.cyl3d)

    p = {
        "ct": u.Q(1.0, "km"),
        "x": u.Q(2.0, "km"),
        "y": u.Q(3.0, "km"),
        "z": u.Q(4.0, "km"),
    }

    v = {
        "ct": u.Q(0.1, "km/s"),
        "x": u.Q(1.0, "km/s"),
        "y": u.Q(-2.0, "km/s"),
        "z": u.Q(0.5, "km/s"),
    }

    v_to = cxt.tangent_transform(chart_to, chart_from, v, p)

    def pack(chart, vals):
        unit = u.unit_of(vals["ct"])
        return jnp.stack([u.uconvert(unit, vals[k]).value for k in chart.components])

    v_from = pack(chart_from, v)
    B_from = cxt.frame_cart(chart_from, at=p)
    v_cart = cxt.pushforward(B_from, v_from)

    p_to = cxt.point_transform(chart_to, chart_from, p)
    v_to_vals = pack(chart_to, v_to)
    B_to = cxt.frame_cart(chart_to, at=p_to)
    v_cart2 = cxt.pushforward(B_to, v_to_vals)

    metric = cxm.metric_of(chart_from)
    eta = metric.metric_matrix(chart_from, p)
    inner = jnp.einsum("i,ij,j->", v_cart, eta, v_cart)
    inner2 = jnp.einsum("i,ij,j->", v_cart2, eta, v_cart2)

    assert jnp.allclose(inner, inner2, atol=1e-6)
