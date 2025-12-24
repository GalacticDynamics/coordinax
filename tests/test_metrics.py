"""Tests for metric resolution."""

import jax.numpy as jnp

import unxt as u

import coordinax as cx
from coordinax._src.representations.frames import frame_to_cart, pushforward
from coordinax._src.representations.metrics import (
    EuclideanMetric,
    MinkowskiMetric,
    SphereMetric,
)


def test_metric_of_euclidean_3d() -> None:
    metric = cx.r.metric_of(cx.r.cart3d)
    assert isinstance(metric, EuclideanMetric)
    assert metric.signature == (1, 1, 1)


def test_metric_of_cylindrical_3d() -> None:
    metric = cx.r.metric_of(cx.r.cyl3d)
    assert isinstance(metric, EuclideanMetric)
    assert metric.signature == (1, 1, 1)


def test_metric_of_spacetime_euclidean() -> None:
    rep = cx.r.SpaceTimeEuclidean(cx.r.cyl3d)
    metric = cx.r.metric_of(rep)
    assert isinstance(metric, EuclideanMetric)
    assert metric.signature == (1, 1, 1, 1)


def test_metric_of_spacetime_ct() -> None:
    rep = cx.r.SpaceTimeCT(cx.r.cyl3d)
    metric = cx.r.metric_of(rep)
    assert isinstance(metric, MinkowskiMetric)
    assert metric.signature == (-1, 1, 1, 1)


def test_metric_of_twosphere() -> None:
    metric = cx.r.metric_of(cx.r.twosphere)
    assert isinstance(metric, SphereMetric)
    assert metric.signature == (1, 1)


def test_minkowski_inner_product_invariant() -> None:
    rep_from = cx.r.SpaceTimeCT(cx.r.cart3d)
    rep_to = cx.r.SpaceTimeCT(cx.r.cyl3d)

    p = {
        "ct": u.Quantity(1.0, "km"),
        "x": u.Quantity(2.0, "km"),
        "y": u.Quantity(3.0, "km"),
        "z": u.Quantity(4.0, "km"),
    }

    v = {
        "ct": u.Quantity(0.1, "km/s"),
        "x": u.Quantity(1.0, "km/s"),
        "y": u.Quantity(-2.0, "km/s"),
        "z": u.Quantity(0.5, "km/s"),
    }

    v_to = cx.r.diff_map(rep_to, rep_from, v, p)

    def pack(rep, vals):
        unit = u.unit_of(vals["ct"])
        return jnp.stack([u.uconvert(unit, vals[k]).value for k in rep.components])

    v_from = pack(rep_from, v)
    B_from = frame_to_cart(rep_from, p)
    v_cart = pushforward(B_from, v_from)

    p_to = cx.r.coord_map(rep_to, rep_from, p)
    v_to_vals = pack(rep_to, v_to)
    B_to = frame_to_cart(rep_to, p_to)
    v_cart2 = pushforward(B_to, v_to_vals)

    metric = cx.r.metric_of(rep_from)
    eta = metric.metric_matrix(rep_from, p)
    inner = jnp.einsum("i,ij,j->", v_cart, eta, v_cart)
    inner2 = jnp.einsum("i,ij,j->", v_cart2, eta, v_cart2)

    assert jnp.allclose(inner, inner2, atol=1e-6)
