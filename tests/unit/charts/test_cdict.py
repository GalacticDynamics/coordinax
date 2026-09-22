"""Tests for cdict function."""

import hypothesis.strategies as st
import jax.numpy as jnp
import pytest
from hypothesis import given

import unxt as u
import unxts.hypothesis as ust

import coordinax.charts as cxc
import coordinaxs.hypothesis.main as cxst
from .conftest import shapes_ending_in_123, xps
from coordinaxs.api.custom_types import CDict


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


def test_cdict_from_a_quantity_matrix() -> None:
    """cdict(QuantityMatrix, chart) splits it into per-component quantities."""
    import unxts.linalg as ul

    qm = ul.QuantityMatrix(jnp.asarray([1.0, 2.0, 3.0]), unit=("m", "m", "m"))

    got = cxc.cdict(qm, cxc.cart3d)

    assert set(got.keys()) == set(cxc.cart3d.components)
    assert float(u.ustrip("m", got["y"])) == 2.0


def test_cdict_refuses_a_two_dimensional_quantity_matrix() -> None:
    """A `QuantityMatrix` reaching `cdict` must be one point, not a stack.

    The chart says how to name one row's components and nothing says how to
    name a second axis, so the guard refuses rather than guessing. Pinned
    because the shape that trips it -- the (1, n) a matrix naturally has -- is
    the one a caller is most likely to try.
    """
    import unxts.linalg as ul

    qm = ul.QuantityMatrix(jnp.zeros((1, 3)), unit=(("m", "m", "m"),))

    with pytest.raises(ValueError, match="must be 1D"):
        cxc.cdict(qm, cxc.cart3d)


@pytest.mark.parametrize("chart", [cxc.sph3d, cxc.lonlat_sph3d, cxc.cyl3d])
def test_cdict_refuses_one_unit_for_a_heterogeneous_chart(chart) -> None:
    """One unit cannot describe components of different dimensions (#949)."""
    with pytest.raises(ValueError, match="every component would take that unit"):
        cxc.cdict(u.Q([1.0, 2.0, 3.0], "kpc"), chart)


@pytest.mark.parametrize("unit", ["kpc", "km/s", "km/s2"])
def test_cdict_accepts_one_unit_for_a_homogeneous_chart(unit: str) -> None:
    """A Cartesian chart shares one dimension, so any single unit is fine (#949).

    In particular a tangent's units must keep working: the chart's coordinate
    dimensions are lengths, but the values are speeds or accelerations.
    """
    got = cxc.cdict(u.Q([1.0, 2.0, 3.0], unit), cxc.cart3d)
    assert set(got.keys()) == set(cxc.cart3d.components)
    assert all(u.unit_of(v) == u.unit(unit) for v in got.values())
