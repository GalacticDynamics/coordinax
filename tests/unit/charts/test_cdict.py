"""Tests for cdict function."""

import hypothesis.strategies as st
import jax.numpy as jnp
import pytest
from hypothesis import given

import unxt as u
import unxt_hypothesis as ust

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst
from .conftest import shapes_ending_in_123, xps
from coordinax.internal import CDict


@given(cxst.cdicts(cxst.charts()))
def test_cdict_of_cdict_is_cdict(cdict: CDict) -> None:
    """cdict(CDict) should return a CDict (dict with string keys)."""
    got = cxc.cdict(cdict)
    assert got is cdict


@given(ust.quantities(shape=shapes_ending_in_123()))
def test_cdict_from_quantity(q):
    """cdict(Quantity) should return a CDict."""
    got = cxc.cdict(q)
    assert isinstance(got, dict)
    assert len(got.keys()) == q.shape[-1]
    for i, v in enumerate(got.values()):
        assert jnp.array_equal(v.value, q.value[..., i])


def test_cdict_from_quantity_chart_error():
    """cdict(Quantity, chart) with mismatched chart should raise ValueError."""
    q = u.Q([1, 2], "m")  # 2 components
    with pytest.raises(
        ValueError, match=r"Quantity last dimension 2 does not match provided keys 3."
    ):
        cxc.cdict(q, cxc.cart3d)  # cart3d expects 3 components


@given(
    data=st.data(),
    chart=st.sampled_from([cxc.cart1d, cxc.radial1d, cxc.cart2d, cxc.cart3d]),
)
def test_cdict_from_quantity_and_chart(data, chart):
    """cdict(Quantity, chart) should return a CDict with correct keys and values."""
    ndim = len(chart.components)
    q = data.draw(ust.quantities(shape=(ndim,)))

    got = cxc.cdict(q, chart)

    assert isinstance(got, dict)
    assert set(got.keys()) == set(chart.components)
    for i, k in enumerate(chart.components):
        assert jnp.array_equal(got[k].value, q.value[..., i])


@given(
    data=st.data(),
    chart=st.sampled_from([cxc.cart1d, cxc.radial1d, cxc.cart2d, cxc.cart3d]),
)
def test_cdict_from_array_and_chart(data, chart):
    """cdict(array, chart) should return a CDict with correct keys and values."""
    ndim = len(chart.components)
    dtype = data.draw(st.sampled_from([jnp.float32, jnp.float64]))
    q = data.draw(xps.arrays(dtype, shape=(ndim,)))

    got = cxc.cdict(q, chart)

    assert isinstance(got, dict)
    assert set(got.keys()) == set(chart.components)
    for i, k in enumerate(chart.components):
        assert jnp.array_equal(got[k], q[..., i], equal_nan=True)
