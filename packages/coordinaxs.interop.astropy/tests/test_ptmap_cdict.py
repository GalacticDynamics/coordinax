"""`coordinax.charts.pt_map` on CDicts agrees with Astropy's `represent_as`.

Every supported 3-D chart is checked against its Astropy counterpart, in both
directions, for a known point and for arbitrary Hypothesis-generated points.
The chart <-> representation correspondence is the whole of the test:

* ``cart3d``       <-> ``CartesianRepresentation``        (x, y, z)
* ``cyl3d``        <-> ``CylindricalRepresentation``      (rho, phi, z)
* ``sph3d``        <-> ``PhysicsSphericalRepresentation`` (r, theta, phi)
* ``lonlat_sph3d`` <-> ``SphericalRepresentation``        (lon, lat, distance)

Only CDict (dict of `unxt.Quantity`) inputs are covered here;
`coordinax.vectors.Point` <-> Astropy lives in ``test_vectors.py``.
"""

__all__: tuple[str, ...] = ()

import math

from types import SimpleNamespace

import astropy.coordinates as apyc
import astropy.units as apyu
import jax.numpy as jnp
import pytest
from hypothesis import assume, given, strategies as st

import unxt as u
import unxts.hypothesis as ust

import coordinax.charts as cxc
from coordinaxs.interop.astropy import (
    convert_cx_cdict_to_astropy_cartrep,
    convert_cx_cdict_to_astropy_cylrep,
    convert_cx_cdict_to_astropy_physsphrep,
    convert_cx_cdict_to_astropy_sphrep,
)

# ---------------------------------------------------------------------------
# Strategies
#
# allow_subnormal=False: JAX flushes subnormals to zero (FTZ) while NumPy keeps
# them, causing divergence in atan2 near the origin.


def _q(unit: str, lower: float, upper: float) -> st.SearchStrategy:
    return ust.quantities(
        unit,
        dtype=jnp.float64,
        elements=st.floats(lower, upper, allow_nan=False, allow_subnormal=False),
    )


POS_KM = _q("km", 0.5, 100)
ANY_KM = _q("km", -100, 100)
AZIMUTH_RAD = _q("rad", -3, 3)
# 0.1 avoids the polar singularity; 3.04 avoids theta ~ pi (south pole)
POLAR_RAD = _q("rad", 0.1, 3.04)
LAT_RAD = _q("rad", -1.5, 1.5)

#: Components Astropy normalises onto a different branch than `atan2` does.
ANGULAR_COMPONENTS = frozenset({"phi", "lon"})


def _off_axis(p: dict) -> bool:
    """`atan2(y, x)` is undefined on the z-axis, and r=0 is degenerate."""
    x, y, z = (float(p[k].value) for k in ("x", "y", "z"))
    return math.hypot(x, y) > 1e-6 and math.hypot(x, y, z) > 1e-6


CHARTS = {
    "cart3d": SimpleNamespace(
        chart=cxc.cart3d,
        to_astropy=convert_cx_cdict_to_astropy_cartrep,
        astropy_rep=apyc.CartesianRepresentation,
        known={"x": u.Q(3, "km"), "y": u.Q(4, "km"), "z": u.Q(0, "km")},
        strategies={"x": ANY_KM, "y": ANY_KM, "z": ANY_KM},
        assume=_off_axis,
    ),
    "cyl3d": SimpleNamespace(
        chart=cxc.cyl3d,
        to_astropy=convert_cx_cdict_to_astropy_cylrep,
        astropy_rep=apyc.CylindricalRepresentation,
        known={"rho": u.Q(5, "km"), "phi": u.Q(0.6435942, "rad"), "z": u.Q(1, "km")},
        strategies={"rho": POS_KM, "phi": AZIMUTH_RAD, "z": ANY_KM},
        assume=None,
    ),
    "sph3d": SimpleNamespace(
        chart=cxc.sph3d,
        to_astropy=convert_cx_cdict_to_astropy_physsphrep,
        astropy_rep=apyc.PhysicsSphericalRepresentation,
        known={"r": u.Q(5, "km"), "theta": u.Q(1, "rad"), "phi": u.Q(0.6435942, "rad")},
        strategies={"r": POS_KM, "theta": POLAR_RAD, "phi": AZIMUTH_RAD},
        assume=None,
    ),
    "lonlat_sph3d": SimpleNamespace(
        chart=cxc.lonlat_sph3d,
        to_astropy=convert_cx_cdict_to_astropy_sphrep,
        astropy_rep=apyc.SphericalRepresentation,
        known={
            "lon": u.Q(0.6435942, "rad"),
            "lat": u.Q(0.4, "rad"),
            "distance": u.Q(5, "km"),
        },
        strategies={"lon": AZIMUTH_RAD, "lat": LAT_RAD, "distance": POS_KM},
        assume=None,
    ),
}

PAIRS = [(src, dst) for src in CHARTS for dst in CHARTS if src != dst]
PAIR_IDS = [f"{src}->{dst}" for src, dst in PAIRS]


# ---------------------------------------------------------------------------
# Comparison


def _approx_equal(got: u.AbstractQuantity, apy: apyu.Quantity, *, rel=1e-5) -> None:
    """Assert ``got`` ~= ``apy`` after converting both to a common unit."""
    assert float(u.ustrip(apy.unit, got)) == pytest.approx(
        float(apy.to(apy.unit).value), rel=rel, abs=1e-7
    )


def _approx_angle_equal(
    got: u.AbstractQuantity, apy: apyu.Quantity, *, abs_tol: float = 1e-5
) -> None:
    """Assert ``got`` ~= ``apy`` modulo 2*pi.

    Astropy normalises azimuthal angles (phi / lon) onto ``[0, 2pi)`` via its
    ``Longitude`` type, while coordinax returns ``atan2`` values in
    ``(-pi, pi]``. Both are physically identical, so compare the *circular*
    distance between them.
    """
    got_val = float(u.ustrip(apy.unit, got))
    apy_val = float(apy.to(apy.unit).value)
    scale = math.pi / 180 if apy.unit == "deg" else 1
    diff_rad = (got_val - apy_val) * scale
    diff_rad = (diff_rad + math.pi) % (2 * math.pi) - math.pi  # reduce to (-pi, pi]
    assert abs(diff_rad) == pytest.approx(0, abs=abs_tol)


def _assert_pt_map_matches_astropy(point: dict, src: str, dst: str) -> None:
    """`pt_map(point, src, dst)` equals Astropy's `represent_as` on every key."""
    source, target = CHARTS[src], CHARTS[dst]

    got = cxc.pt_map(point, source.chart, target.chart)
    ref = source.to_astropy(point).represent_as(target.astropy_rep)

    for key in target.chart.components:
        expected = getattr(ref, key)
        if key in ANGULAR_COMPONENTS:
            _approx_angle_equal(got[key], expected)
        else:
            _approx_equal(got[key], expected)


# ---------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize(("src", "dst"), PAIRS, ids=PAIR_IDS)
def test_known_point_matches_astropy(src: str, dst: str) -> None:
    """A fixed, human-checkable point maps the same way Astropy maps it."""
    _assert_pt_map_matches_astropy(CHARTS[src].known, src, dst)


@pytest.mark.parametrize(("src", "dst"), PAIRS, ids=PAIR_IDS)
@given(data=st.data())
def test_arbitrary_point_matches_astropy(
    src: str, dst: str, data: st.DataObject
) -> None:
    """Arbitrary bounded points map the same way Astropy maps them."""
    source = CHARTS[src]
    point = {k: data.draw(s, label=k) for k, s in source.strategies.items()}
    if source.assume is not None:
        assume(source.assume(point))
    _assert_pt_map_matches_astropy(point, src, dst)
