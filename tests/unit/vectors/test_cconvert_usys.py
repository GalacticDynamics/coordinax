"""`usys=` on vector `cconvert` takes effect or raises; it is never ignored.

``usys`` says what units bare-array components are in. On quantity data it
can do nothing, so it raises rather than being silently dropped (#1032, F5).
"""

import jax.numpy as jnp
import pytest
from hypothesis import assume, given, reject, strategies as st

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.hypothesis.vectors as cxvst

USYS = u.unitsystem("kpc", "Myr", "Msun", "deg")


def _velocity(point: cx.Point) -> cx.Tangent:
    data = {k: v / u.Q(1.0, "Myr") for k, v in point.data.items()}
    return cx.Tangent.from_(data, point.chart, cxr.coord_basis, cxr.vel)


@given(point=cxvst.vectors(), kind=st.sampled_from(["point", "tangent", "coord"]))
def test_usys_on_quantity_data_raises(point, kind):
    """Point, Tangent and Coordinate refuse a usys they would ignore."""
    chart = point.chart
    if kind == "point":
        call = lambda: point.cconvert(chart, usys=USYS)
    elif kind == "tangent":
        call = lambda: _velocity(point).cconvert(chart, at=point, usys=USYS)
    else:
        coord = cx.Coordinate(point=point, velocity=_velocity(point))
        call = lambda: coord.cconvert(chart, usys=USYS)
    with pytest.raises(TypeError, match="usys only says what units bare-array"):
        call()


@given(point=cxvst.vectors())
def test_usys_on_bare_data_takes_effect(point):
    """On bare arrays, usys gives the same numbers as the quantity conversion.

    Parameterized charts are left out: their parameters (e.g. a prolate
    ``Delta``) carry their own units, which bare data cannot match.
    """
    assume(not isinstance(point.chart, cxc.AbstractParameterizedChart))
    try:
        target = point.chart.cartesian
    except Exception:  # noqa: BLE001  # charts with no global Cartesian chart
        reject()
    assume(target != point.chart)

    def to_usys(data):
        return {k: u.ustrip(USYS[u.dimension_of(v)], v) for k, v in data.items()}

    expected = to_usys(point.cconvert(target).data)
    bare = cx.Point.from_(to_usys(point.data), point.chart)
    got = bare.cconvert(target, usys=USYS).data

    for k, v in expected.items():
        assert jnp.allclose(got[k], v, rtol=1e-6, atol=1e-12, equal_nan=True), k


def test_usys_error_names_uconvert():
    """The error points at `uconvert` for setting the output units."""
    p = cx.Point.from_([1.0, 2.0, 3.0], "km")
    with pytest.raises(TypeError, match=r"\.uconvert\(usys\)"):
        cx.cconvert(p, cx.sph3d, usys=USYS)
