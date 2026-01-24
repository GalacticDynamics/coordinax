"""Tests for cdict function."""

import hypothesis.strategies as st
import jax.numpy as jnp
import pytest
from hypothesis import given, settings

import unxt as u
import unxt_hypothesis as ust

import coordinax as cx
import coordinax.charts as cxc
import coordinax_hypothesis as cxst
from .conftest import shapes_ending_in_123, xps
from coordinax._src.custom_types import CsDict


@given(cxst.cdicts(cxst.charts()))
@settings(max_examples=100)
def test_cdict_of_csdict_is_csdict(cdict: CsDict) -> None:
    """cdict(CsDict) should return a CsDict (dict with string keys)."""
    got = cx.cdict(cdict)
    assert got is cdict


@given(ust.quantities(shape=shapes_ending_in_123()))
def test_cdict_from_quantity(q):
    """cdict(Quantity) should return a CsDict."""
    got = cx.cdict(q)
    assert isinstance(got, dict)
    assert len(got.keys()) == q.shape[-1]
    for i, v in enumerate(got.values()):
        assert jnp.array_equal(v.value, q.value[..., i])


def test_cdict_from_quantity_chart_error():
    """cdict(Quantity, chart) with mismatched chart should raise ValueError."""
    q = u.Q([1.0, 2.0], "m")  # 2 components
    with pytest.raises(
        ValueError, match="Quantity last dimension 2 does not match chart"
    ):
        cx.cdict(q, cxc.cart3d)  # cart3d expects 3 components


@given(
    data=st.data(),
    chart=st.sampled_from([cxc.cart1d, cxc.radial1d, cxc.cart2d, cxc.cart3d]),
)
def test_cdict_from_quantity_chart(data, chart):
    """cdict(Quantity, chart) should return a CsDict with correct keys and values."""
    ndim = len(chart.components)
    q = data.draw(ust.quantities(shape=(ndim,)))

    got = cx.cdict(q, chart)

    assert isinstance(got, dict)
    assert set(got.keys()) == set(chart.components)
    for i, k in enumerate(chart.components):
        assert jnp.array_equal(got[k].value, q.value[..., i])


@given(
    data=st.data(),
    chart=st.sampled_from([cxc.cart1d, cxc.radial1d, cxc.cart2d, cxc.cart3d]),
)
def test_cdict_from_quantity_chart(data, chart):
    """cdict(Quantity, chart) should return a CsDict with correct keys and values."""
    ndim = len(chart.components)
    q = data.draw(xps.arrays(xps.real_dtypes(), shape=(ndim,)))

    got = cx.cdict(q, chart)

    assert isinstance(got, dict)
    assert set(got.keys()) == set(chart.components)
    for i, k in enumerate(chart.components):
        assert jnp.array_equal(got[k], q[..., i])
