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


@pytest.mark.parametrize(
    ("chart", "units"),
    [
        (cxc.sph3d, ("kpc", "rad", "rad")),
        (cxc.lonlat_sph3d, ("deg", "deg", "kpc")),
        (cxc.cyl3d, ("kpc", "rad", "kpc")),
    ],
)
def test_cdict_units_matrix_gives_each_component_its_own_unit(chart, units) -> None:
    """cdict(array, UnitsMatrix, chart) distributes one unit per component.

    Regression for #973: the whole units tuple was handed to every component,
    so each one carried a composite "(kpc, rad, rad)" that is not a unit.
    """
    import unxts.linalg as ul

    arr = jnp.asarray([1.0, 0.5, 0.2])

    got = cxc.cdict(arr, ul.UnitsMatrix(units), chart)

    assert set(got.keys()) == set(chart.components)
    for i, k in enumerate(chart.components):
        assert got[k].unit == u.unit(units[i])
        assert float(got[k].value) == float(arr[i])


def test_cdict_units_matrix_with_keys_and_alone_also_distribute() -> None:
    """The sibling `keys` and unit-only overloads distribute too (#973)."""
    import unxts.linalg as ul

    arr = jnp.asarray([1.0, 0.5, 0.2])
    units = ul.UnitsMatrix(("kpc", "rad", "rad"))

    by_keys = cxc.cdict(arr, units, ("r", "theta", "phi"))
    assert [v.unit for v in by_keys.values()] == [u.unit(x) for x in units]

    alone = cxc.cdict(arr, units)  # guesses cart3d from the last dimension
    assert [v.unit for v in alone.values()] == [u.unit(x) for x in units]


@pytest.mark.parametrize("units", [("kpc", "rad"), ("kpc", "rad", "rad", "rad")])
def test_cdict_units_matrix_wrong_length_errors(units) -> None:
    """A units tuple that is not one-per-component is refused."""
    import unxts.linalg as ul

    with pytest.raises(ValueError, match="does not match provided keys 3"):
        cxc.cdict(jnp.asarray([1.0, 0.5, 0.2]), ul.UnitsMatrix(units), cxc.sph3d)


def test_cdict_homogeneous_units_matrix_still_works() -> None:
    """The Cartesian all-same-unit case keeps working (#973)."""
    import unxts.linalg as ul

    got = cxc.cdict(
        jnp.asarray([1.0, 2.0, 3.0]), ul.UnitsMatrix(("m", "m", "m")), cxc.cart3d
    )

    assert set(got.keys()) == set(cxc.cart3d.components)
    assert all(v.unit == u.unit("m") for v in got.values())
    assert float(u.ustrip("m", got["z"])) == 3.0


# CartND keeps all N values in its one component ``q`` (#1028). Assert on the
# values, not on a vector's `.shape`, which for CartND is tracked separately.
_ND_CASES = pytest.mark.parametrize(
    ("batch", "n"), [((), 4), ((), 5), ((), 7), ((2,), 4), ((3, 2), 5)]
)


@_ND_CASES
def test_cdict_quantity_cartnd_keeps_the_last_axis_whole(batch, n) -> None:
    """A Quantity with N >= 4 components maps to one ``q`` holding all of them."""
    q = u.Q(jnp.arange(float(n)) * jnp.ones((*batch, n)), "km")
    for got in (cxc.cdict(q), cxc.cdict(q, cxc.cartnd)):
        assert set(got) == {"q"}
        assert jnp.array_equal(u.ustrip("km", got["q"]), q.value)


@_ND_CASES
def test_cdict_array_cartnd_keeps_the_last_axis_whole(batch, n) -> None:
    """Arrays, with or without a unit or unit system, fill ``q`` whole."""
    a = jnp.arange(float(n)) * jnp.ones((*batch, n))
    assert jnp.array_equal(cxc.cdict(a, cxc.cartnd)["q"], a)
    assert jnp.array_equal(cxc.cdict(a, None, cxc.cartnd)["q"], a)
    for got in (cxc.cdict(a, "km"), cxc.cdict(a, "km", cxc.cartnd)):
        assert jnp.array_equal(u.ustrip("km", got["q"]), a)
    got = cxc.cdict(a, u.unitsystems.si, cxc.cartnd)
    assert jnp.array_equal(u.ustrip("m", got["q"]), a)


def test_cdict_cartnd_refuses_a_quantity_matrix() -> None:
    """CartND has one unit; per-component units have nowhere to go."""
    import unxts.linalg as ul

    q = ul.QuantityMatrix(jnp.ones(4), unit=("m", "m", "km", "m"))
    with pytest.raises(ValueError, match="one array in one unit"):
        cxc.cdict(q, cxc.cartnd)
