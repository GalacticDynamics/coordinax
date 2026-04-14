"""Tests: pt_map (CDict) vs Astropy representation transforms.

These tests verify that ``coordinax.charts.pt_map`` operating on
plain ``CDict`` dictionaries produces the same numerical results as Astropy's
``represent_as()`` method for all supported chart / representation pairs.

Scope
-----
* Only ``CDict`` (dict of ``unxt.Quantity``) inputs are tested here.
* ``coordinax.Vector`` ↔ Astropy comparisons live in a separate test module.

Chart ↔ Astropy representation mapping
---------------------------------------
* ``cart3d``      ↔ ``CartesianRepresentation``        (keys: x, y, z)
* ``cyl3d``       ↔ ``CylindricalRepresentation``      (keys: rho, phi, z)
* ``sph3d``       ↔ ``PhysicsSphericalRepresentation`` (keys: r, theta, phi)
* ``lonlat_sph3d``↔ ``SphericalRepresentation``        (keys: lon, lat, distance)
"""

__all__: tuple[str, ...] = ()

import math

import astropy.coordinates as apyc
import astropy.units as apyu
import jax.numpy as jnp
import pytest
from hypothesis import assume, given, settings, strategies as st

import unxt as u
import unxt_hypothesis as ust

import coordinax.charts as cxc
from coordinax.interop.astropy import (
    convert_cx_cdict_to_astropy_cartrep,
    convert_cx_cdict_to_astropy_cylrep,
    convert_cx_cdict_to_astropy_physsphrep,
    convert_cx_cdict_to_astropy_sphrep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# allow_subnormal=False: JAX flushes subnormals to zero (FTZ) while NumPy keeps
# them, causing divergence in atan2 near the origin.
def make_strat(
    unit: str, lower: float, upper: float
) -> st.SearchStrategy[u.AbstractQuantity]:
    return ust.quantities(
        unit,
        dtype=jnp.float64,
        elements=st.floats(lower, upper, allow_nan=False, allow_subnormal=False),
    )


_pos_km = make_strat("km", 0.5, 100.0)
_any_km = make_strat("km", -100.0, 100.0)
_phi_rad = make_strat("rad", -3.0, 3.0)
# 0.1 avoids the polar singularity; 3.04 avoids theta≈π (south pole)
_theta_rad = make_strat("rad", 0.1, 3.04)
_lon_rad = make_strat("rad", -3.0, 3.0)
_lat_rad = make_strat("rad", -1.5, 1.5)


def _approx_equal(got: u.AbstractQuantity, apy: apyu.Quantity, *, rel=1e-5) -> None:
    """Assert ``got`` ≈ ``apy`` after converting both to a common unit."""
    got_val = float(u.ustrip(apy.unit, got))
    apy_val = float(apy.to(apy.unit).value)
    assert got_val == pytest.approx(apy_val, rel=rel, abs=1e-7)


def _approx_angle_equal(
    got: u.AbstractQuantity, apy: apyu.Quantity, *, abs_tol: float = 1e-5
) -> None:
    """Assert ``got`` ≈ ``apy`` modulo 2π.

    Astropy normalises azimuthal angles (phi / lon) to ``[0, 2π)`` via its
    ``Longitude`` type, while coordinax returns values from ``atan2`` in
    ``(-π, π]``.  Both representations are physically identical; we compare
    the *circular* distance between the two angles.
    """
    got_val = float(u.ustrip(apy.unit, got))
    apy_val = float(apy.to(apy.unit).value)

    # Convert to radians for the modular comparison
    scale = math.pi / 180.0 if apy.unit == "deg" else 1.0

    diff_rad = (got_val - apy_val) * scale
    # Reduce to (-π, π]
    diff_rad = (diff_rad + math.pi) % (2 * math.pi) - math.pi
    assert abs(diff_rad) == pytest.approx(0.0, abs=abs_tol)


# ---------------------------------------------------------------------------
# Tests: Cartesian → other charts
# ---------------------------------------------------------------------------


class TestCart3DToOther:
    """pt_map(cart3d → X) matches Astropy represent_as(X)."""

    @pytest.fixture
    def point(self):
        return {"x": u.Q(3, "km"), "y": u.Q(4, "km"), "z": u.Q(0, "km")}

    def test_cart3d_to_cyl3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.cart3d, cxc.cyl3d)
        ref = convert_cx_cdict_to_astropy_cartrep(point).represent_as(
            apyc.CylindricalRepresentation
        )

        _approx_equal(got["rho"], ref.rho)
        _approx_angle_equal(got["phi"], ref.phi)
        _approx_equal(got["z"], ref.z)

    def test_cart3d_to_sph3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.cart3d, cxc.sph3d)
        ref = convert_cx_cdict_to_astropy_cartrep(point).represent_as(
            apyc.PhysicsSphericalRepresentation
        )

        _approx_equal(got["r"], ref.r)
        _approx_equal(got["theta"], ref.theta)
        _approx_angle_equal(got["phi"], ref.phi)

    def test_cart3d_to_lonlat_sph3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.cart3d, cxc.lonlat_sph3d)
        ref = convert_cx_cdict_to_astropy_cartrep(point).represent_as(
            apyc.SphericalRepresentation
        )

        _approx_angle_equal(got["lon"], ref.lon)
        _approx_equal(got["lat"], ref.lat)
        _approx_equal(got["distance"], ref.distance)

    @given(x=_any_km, y=_any_km, z=_any_km)
    @settings(deadline=None)
    def test_cart3d_to_cyl3d_hypothesis(self, x, y, z) -> None:
        # phi = atan2(y, x) is undefined (or implementation-dependent) when x=y=0
        assume(math.hypot(x.value, y.value) > 1e-6)
        p = {"x": x, "y": y, "z": z}
        got = cxc.pt_map(p, cxc.cart3d, cxc.cyl3d)
        ref = convert_cx_cdict_to_astropy_cartrep(p).represent_as(
            apyc.CylindricalRepresentation
        )

        _approx_equal(got["rho"], ref.rho)
        _approx_angle_equal(got["phi"], ref.phi)
        _approx_equal(got["z"], ref.z)

    @given(x=_any_km, y=_any_km, z=_pos_km)
    @settings(deadline=None)
    def test_cart3d_to_sph3d_hypothesis(self, x, y, z) -> None:
        # phi = atan2(y, x) is undefined when x=y=0 (along z-axis)
        assume(math.hypot(x.value, y.value) > 1e-6)
        p = {"x": x, "y": y, "z": z}
        got = cxc.pt_map(p, cxc.cart3d, cxc.sph3d)
        ref = convert_cx_cdict_to_astropy_cartrep(p).represent_as(
            apyc.PhysicsSphericalRepresentation
        )

        _approx_equal(got["r"], ref.r)
        _approx_equal(got["theta"], ref.theta)
        _approx_angle_equal(got["phi"], ref.phi)

    @given(x=_any_km, y=_any_km, z=_any_km)
    @settings(deadline=None)
    def test_cart3d_to_lonlat_sph3d_hypothesis(self, x, y, z) -> None:
        # lon = atan2(y, x) is undefined at poles (x=y=0); r=0 is degenerate
        assume(
            math.hypot(x.value, y.value) > 1e-6
            and math.hypot(x.value, y.value, float(z.value)) > 1e-6
        )
        p = {"x": x, "y": y, "z": z}
        got = cxc.pt_map(p, cxc.cart3d, cxc.lonlat_sph3d)
        ref = convert_cx_cdict_to_astropy_cartrep(p).represent_as(
            apyc.SphericalRepresentation
        )

        _approx_angle_equal(got["lon"], ref.lon)
        _approx_equal(got["lat"], ref.lat)
        _approx_equal(got["distance"], ref.distance)


# ---------------------------------------------------------------------------
# Tests: Cylindrical → other charts
# ---------------------------------------------------------------------------


class TestCyl3DToOther:
    """pt_map(cyl3d → X) matches Astropy represent_as(X)."""

    @pytest.fixture
    def point(self):
        return {"rho": u.Q(5, "km"), "phi": u.Q(0.6435942, "rad"), "z": u.Q(1, "km")}

    def test_cyl3d_to_cart3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.cyl3d, cxc.cart3d)
        ref = convert_cx_cdict_to_astropy_cylrep(point).represent_as(
            apyc.CartesianRepresentation
        )

        _approx_equal(got["x"], ref.x)
        _approx_equal(got["y"], ref.y)
        _approx_equal(got["z"], ref.z)

    def test_cyl3d_to_sph3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.cyl3d, cxc.sph3d)
        ref = convert_cx_cdict_to_astropy_cylrep(point).represent_as(
            apyc.PhysicsSphericalRepresentation
        )

        _approx_equal(got["r"], ref.r)
        _approx_equal(got["theta"], ref.theta)
        _approx_angle_equal(got["phi"], ref.phi)

    def test_cyl3d_to_lonlat_sph3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.cyl3d, cxc.lonlat_sph3d)
        ref = convert_cx_cdict_to_astropy_cylrep(point).represent_as(
            apyc.SphericalRepresentation
        )

        _approx_angle_equal(got["lon"], ref.lon)
        _approx_equal(got["lat"], ref.lat)
        _approx_equal(got["distance"], ref.distance)

    @given(rho=_pos_km, phi=_phi_rad, z=_any_km)
    @settings(deadline=None)
    def test_cyl3d_to_cart3d_hypothesis(self, rho, phi, z) -> None:
        p = {"rho": rho, "phi": phi, "z": z}
        got = cxc.pt_map(p, cxc.cyl3d, cxc.cart3d)
        ref = convert_cx_cdict_to_astropy_cylrep(p).represent_as(
            apyc.CartesianRepresentation
        )

        _approx_equal(got["x"], ref.x)
        _approx_equal(got["y"], ref.y)
        _approx_equal(got["z"], ref.z)

    @given(rho=_pos_km, phi=_phi_rad, z=_any_km)
    @settings(deadline=None)
    def test_cyl3d_to_sph3d_hypothesis(self, rho, phi, z) -> None:
        p = {"rho": rho, "phi": phi, "z": z}
        got = cxc.pt_map(p, cxc.cyl3d, cxc.sph3d)
        ref = convert_cx_cdict_to_astropy_cylrep(p).represent_as(
            apyc.PhysicsSphericalRepresentation
        )

        _approx_equal(got["r"], ref.r)
        _approx_equal(got["theta"], ref.theta)
        _approx_angle_equal(got["phi"], ref.phi)

    @given(rho=_pos_km, phi=_phi_rad, z=_any_km)
    @settings(deadline=None)
    def test_cyl3d_to_lonlat_sph3d_hypothesis(self, rho, phi, z) -> None:
        p = {"rho": rho, "phi": phi, "z": z}
        got = cxc.pt_map(p, cxc.cyl3d, cxc.lonlat_sph3d)
        ref = convert_cx_cdict_to_astropy_cylrep(p).represent_as(
            apyc.SphericalRepresentation
        )

        _approx_angle_equal(got["lon"], ref.lon)
        _approx_equal(got["lat"], ref.lat)
        _approx_equal(got["distance"], ref.distance)


# ---------------------------------------------------------------------------
# Tests: PhysicsSpherical → other charts
# ---------------------------------------------------------------------------


class TestSph3DToOther:
    """pt_map(sph3d → X) matches Astropy represent_as(X)."""

    @pytest.fixture
    def point(self):
        return {"r": u.Q(5, "km"), "theta": u.Q(1, "rad"), "phi": u.Q(0.6435942, "rad")}

    def test_sph3d_to_cart3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.sph3d, cxc.cart3d)
        ref = convert_cx_cdict_to_astropy_physsphrep(point).represent_as(
            apyc.CartesianRepresentation
        )

        _approx_equal(got["x"], ref.x)
        _approx_equal(got["y"], ref.y)
        _approx_equal(got["z"], ref.z)

    def test_sph3d_to_cyl3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.sph3d, cxc.cyl3d)
        ref = convert_cx_cdict_to_astropy_physsphrep(point).represent_as(
            apyc.CylindricalRepresentation
        )

        _approx_equal(got["rho"], ref.rho)
        _approx_angle_equal(got["phi"], ref.phi)
        _approx_equal(got["z"], ref.z)

    def test_sph3d_to_lonlat_sph3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.sph3d, cxc.lonlat_sph3d)
        ref = convert_cx_cdict_to_astropy_physsphrep(point).represent_as(
            apyc.SphericalRepresentation
        )

        _approx_angle_equal(got["lon"], ref.lon)
        _approx_equal(got["lat"], ref.lat)
        _approx_equal(got["distance"], ref.distance)

    @given(r=_pos_km, theta=_theta_rad, phi=_phi_rad)
    @settings(deadline=None)
    def test_sph3d_to_cart3d_hypothesis(self, r, theta, phi) -> None:
        p = {"r": r, "theta": theta, "phi": phi}
        got = cxc.pt_map(p, cxc.sph3d, cxc.cart3d)
        ref = convert_cx_cdict_to_astropy_physsphrep(p).represent_as(
            apyc.CartesianRepresentation
        )

        _approx_equal(got["x"], ref.x)
        _approx_equal(got["y"], ref.y)
        _approx_equal(got["z"], ref.z)

    @given(r=_pos_km, theta=_theta_rad, phi=_phi_rad)
    @settings(deadline=None)
    def test_sph3d_to_cyl3d_hypothesis(self, r, theta, phi) -> None:
        p = {"r": r, "theta": theta, "phi": phi}
        got = cxc.pt_map(p, cxc.sph3d, cxc.cyl3d)
        ref = convert_cx_cdict_to_astropy_physsphrep(p).represent_as(
            apyc.CylindricalRepresentation
        )

        _approx_equal(got["rho"], ref.rho)
        _approx_angle_equal(got["phi"], ref.phi)
        _approx_equal(got["z"], ref.z)

    @given(r=_pos_km, theta=_theta_rad, phi=_phi_rad)
    @settings(deadline=None)
    def test_sph3d_to_lonlat_sph3d_hypothesis(self, r, theta, phi) -> None:
        p = {"r": r, "theta": theta, "phi": phi}
        got = cxc.pt_map(p, cxc.sph3d, cxc.lonlat_sph3d)
        ref = convert_cx_cdict_to_astropy_physsphrep(p).represent_as(
            apyc.SphericalRepresentation
        )

        _approx_angle_equal(got["lon"], ref.lon)
        _approx_equal(got["lat"], ref.lat)
        _approx_equal(got["distance"], ref.distance)


# ---------------------------------------------------------------------------
# Tests: LonLatSpherical → other charts
# ---------------------------------------------------------------------------


class TestLonLatSph3DToOther:
    """pt_map(lonlat_sph3d → X) matches Astropy represent_as(X)."""

    @pytest.fixture
    def point(self):
        return {
            "lon": u.Q(0.6435942, "rad"),
            "lat": u.Q(0.4, "rad"),
            "distance": u.Q(5, "km"),
        }

    def test_lonlat_to_cart3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.lonlat_sph3d, cxc.cart3d)
        ref = convert_cx_cdict_to_astropy_sphrep(point).represent_as(
            apyc.CartesianRepresentation
        )

        _approx_equal(got["x"], ref.x)
        _approx_equal(got["y"], ref.y)
        _approx_equal(got["z"], ref.z)

    def test_lonlat_to_cyl3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.lonlat_sph3d, cxc.cyl3d)
        ref = convert_cx_cdict_to_astropy_sphrep(point).represent_as(
            apyc.CylindricalRepresentation
        )

        _approx_equal(got["rho"], ref.rho)
        _approx_angle_equal(got["phi"], ref.phi)
        _approx_equal(got["z"], ref.z)

    def test_lonlat_to_sph3d_known(self, point) -> None:
        got = cxc.pt_map(point, cxc.lonlat_sph3d, cxc.sph3d)
        ref = convert_cx_cdict_to_astropy_sphrep(point).represent_as(
            apyc.PhysicsSphericalRepresentation
        )

        _approx_equal(got["r"], ref.r)
        _approx_equal(got["theta"], ref.theta)
        _approx_angle_equal(got["phi"], ref.phi)

    @given(lon=_lon_rad, lat=_lat_rad, distance=_pos_km)
    @settings(deadline=None)
    def test_lonlat_to_cart3d_hypothesis(self, lon, lat, distance) -> None:
        p = {"lon": lon, "lat": lat, "distance": distance}
        got = cxc.pt_map(p, cxc.lonlat_sph3d, cxc.cart3d)
        ref = convert_cx_cdict_to_astropy_sphrep(p).represent_as(
            apyc.CartesianRepresentation
        )

        _approx_equal(got["x"], ref.x)
        _approx_equal(got["y"], ref.y)
        _approx_equal(got["z"], ref.z)

    @given(lon=_lon_rad, lat=_lat_rad, distance=_pos_km)
    @settings(deadline=None)
    def test_lonlat_to_cyl3d_hypothesis(self, lon, lat, distance) -> None:
        p = {"lon": lon, "lat": lat, "distance": distance}
        got = cxc.pt_map(p, cxc.lonlat_sph3d, cxc.cyl3d)
        ref = convert_cx_cdict_to_astropy_sphrep(p).represent_as(
            apyc.CylindricalRepresentation
        )

        _approx_angle_equal(got["phi"], ref.phi)
        _approx_equal(got["rho"], ref.rho)
        _approx_equal(got["z"], ref.z)

    @given(lon=_lon_rad, lat=_lat_rad, distance=_pos_km)
    @settings(deadline=None)
    def test_lonlat_to_sph3d_hypothesis(self, lon, lat, distance) -> None:
        p = {"lon": lon, "lat": lat, "distance": distance}
        got = cxc.pt_map(p, cxc.lonlat_sph3d, cxc.sph3d)
        ref = convert_cx_cdict_to_astropy_sphrep(p).represent_as(
            apyc.PhysicsSphericalRepresentation
        )

        _approx_equal(got["r"], ref.r)
        _approx_equal(got["theta"], ref.theta)
        _approx_angle_equal(got["phi"], ref.phi)
