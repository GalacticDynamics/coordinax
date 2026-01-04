"""Test conversions between Astropy and coordinax-astro frames."""

import functools as ft

import astropy.coordinates as apyc
import astropy.units as apyu
import pytest
from hypothesis import given, settings, strategies as st
from plum import convert

import quaxed.numpy as jnp
import unxt as u
import unxt_hypothesis as ust

import coordinax.vecs as cxv
import coordinax_astro as cxastro

float32s = ft.partial(st.floats, width=32, allow_subnormal=False)


def qf32s(
    unit: str | apyu.Unit | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> st.SearchStrategy[u.AbstractQuantity]:
    """Hypothesis strategy for generating float32 Astropy Quantities."""
    return ust.quantities(
        elements=float32s(min_value=min_value, max_value=max_value),
        dtype=jnp.float32,
        unit=unit,
    )


def apyqf32s(
    unit: str | apyu.Unit | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> st.SearchStrategy[apyu.Quantity]:
    """Hypothesis strategy for generating float32 Astropy Quantities."""
    return ust.quantities(
        elements=float32s(min_value=min_value, max_value=max_value),
        dtype=jnp.float32,
        unit=unit,
    ).map(lambda q: convert(q, apyu.Quantity))


class TestICRSFrameConversions:
    """Test ICRS frame conversions."""

    @staticmethod
    def test_coordinax_to_astropy_icrs() -> None:
        """Test converting coordinax ICRS to Astropy ICRS."""
        cx_frame = cxastro.ICRS()
        apy_frame = convert(cx_frame, apyc.ICRS)

        assert isinstance(apy_frame, apyc.ICRS)

    @staticmethod
    def test_astropy_to_coordinax_icrs() -> None:
        """Test converting Astropy ICRS to coordinax ICRS."""
        apy_frame = apyc.ICRS()
        cx_frame = convert(apy_frame, cxastro.ICRS)

        assert isinstance(cx_frame, cxastro.ICRS)

    @staticmethod
    def test_roundtrip_coordinax_to_astropy_icrs() -> None:
        """Test round-trip conversion starting from coordinax ICRS."""
        original = cxastro.ICRS()

        # Convert to Astropy and back
        apy_frame = convert(original, apyc.ICRS)
        result = convert(apy_frame, cxastro.ICRS)

        assert isinstance(result, cxastro.ICRS)
        assert result == original

    @staticmethod
    def test_roundtrip_astropy_to_coordinax_icrs() -> None:
        """Test round-trip conversion starting from Astropy ICRS."""
        original = apyc.ICRS()

        # Convert to coordinax and back
        cx_frame = convert(original, cxastro.ICRS)
        result = convert(cx_frame, apyc.ICRS)

        assert isinstance(result, apyc.ICRS)


class TestGalactocentricFrameConversions:
    """Test Galactocentric frame conversions."""

    @settings(max_examples=10, deadline=None)
    @given(
        lon=apyqf32s(unit="deg", min_value=0, max_value=360),
        lat=apyqf32s(unit="deg", min_value=-90, max_value=90),
        distance=apyqf32s(unit="kpc", min_value=1, max_value=100),
        z_sun=apyqf32s(unit="kpc", min_value=0.015625, max_value=0.09375),
        roll=apyqf32s(unit="deg", min_value=0, max_value=10),
        vx=apyqf32s(unit="km/s", min_value=-50, max_value=50),
        vy=apyqf32s(unit="km/s", min_value=-50, max_value=50),
        vz=apyqf32s(unit="km/s", min_value=-50, max_value=50),
    )
    def test_roundtrip_coordinax_to_astropy_galactocentric(
        self, lon, lat, distance, z_sun, roll, vx, vy, vz
    ) -> None:
        """Test round-trip conversion starting from coordinax Galactocentric."""
        # Create coordinax Galactocentric frame
        original = cxastro.Galactocentric(
            galcen=cxv.LonLatSphericalPos(lon=lon, lat=lat, distance=distance),
            z_sun=z_sun,
            roll=roll,
            galcen_v_sun=cxv.CartesianVel3D(x=vx, y=vy, z=vz),
        )

        # Convert to Astropy and back
        apy_frame = convert(original, apyc.Galactocentric)
        result = convert(apy_frame, cxastro.Galactocentric)

        # Check types
        assert isinstance(result, cxastro.Galactocentric)

        # Check values are close (accounting for unit conversions and floating point)
        assert jnp.allclose(
            result.galcen.lon, original.galcen.lon, rtol=1e-10, atol=u.Q(1e-10, "deg")
        )
        assert jnp.allclose(
            result.galcen.lat, original.galcen.lat, rtol=1e-10, atol=u.Q(1e-10, "deg")
        )
        assert jnp.allclose(
            result.galcen.distance,
            original.galcen.distance,
            rtol=1e-10,
            atol=u.Q(1e-10, "kpc"),
        )
        assert jnp.allclose(
            result.z_sun, original.z_sun, rtol=1e-10, atol=u.Q(1e-10, "pc")
        )
        assert jnp.allclose(
            result.roll, original.roll, rtol=1e-10, atol=u.Q(1e-10, "deg")
        )
        assert jnp.allclose(
            result.galcen_v_sun.x,
            original.galcen_v_sun.x,
            rtol=1e-10,
            atol=u.Q(1e-10, "km/s"),
        )
        assert jnp.allclose(
            result.galcen_v_sun.y,
            original.galcen_v_sun.y,
            rtol=1e-10,
            atol=u.Q(1e-10, "km/s"),
        )
        assert jnp.allclose(
            result.galcen_v_sun.z,
            original.galcen_v_sun.z,
            rtol=1e-10,
            atol=u.Q(1e-10, "km/s"),
        )

    @settings(max_examples=10, deadline=None)
    @given(
        lon=apyqf32s(unit="deg", min_value=0, max_value=360),
        lat=apyqf32s(unit="deg", min_value=-90, max_value=90),
        distance=apyqf32s(unit="kpc", min_value=1, max_value=100),
        z_sun=apyqf32s(unit="kpc", min_value=0.015625, max_value=0.09375),
        roll=apyqf32s(unit="deg", min_value=0, max_value=10),
        vx=apyqf32s(unit="km/s", min_value=-50, max_value=50),
        vy=apyqf32s(unit="km/s", min_value=-50, max_value=50),
        vz=apyqf32s(unit="km/s", min_value=-50, max_value=50),
    )
    def test_roundtrip_astropy_to_coordinax_galactocentric(
        self, lon, lat, distance, z_sun, roll, vx, vy, vz
    ) -> None:
        """Test round-trip conversion starting from Astropy Galactocentric."""
        # Create Astropy Galactocentric frame
        original = apyc.Galactocentric(
            galcen_coord=apyc.SphericalRepresentation(
                lon=lon, lat=lat, distance=distance
            ),
            galcen_distance=distance,
            galcen_v_sun=apyc.CartesianDifferential(d_x=vx, d_y=vy, d_z=vz),
            z_sun=z_sun,
            roll=roll,
        )

        # Convert to coordinax and back
        cx_frame = convert(original, cxastro.Galactocentric)
        result = convert(cx_frame, apyc.Galactocentric)

        # Check types
        assert isinstance(result, apyc.Galactocentric)

        # Check values are close
        assert result.galcen_coord.ra.to_value("deg") == pytest.approx(
            original.galcen_coord.ra.to_value("deg"), rel=1e-10
        )
        assert result.galcen_coord.dec.to_value("deg") == pytest.approx(
            original.galcen_coord.dec.to_value("deg"), rel=1e-10
        )
        assert result.galcen_distance.to_value("kpc") == pytest.approx(
            original.galcen_distance.to_value("kpc"), rel=1e-10
        )
        assert result.z_sun.to_value("kpc") == pytest.approx(
            original.z_sun.to_value("kpc"), rel=1e-10
        )
        assert result.roll.to_value("deg") == pytest.approx(
            original.roll.to_value("deg"), rel=1e-10
        )
        assert result.galcen_v_sun.d_x.to_value("km/s") == pytest.approx(
            original.galcen_v_sun.d_x.to_value("km/s"), rel=1e-10
        )
        assert result.galcen_v_sun.d_y.to_value("km/s") == pytest.approx(
            original.galcen_v_sun.d_y.to_value("km/s"), rel=1e-10
        )
        assert result.galcen_v_sun.d_z.to_value("km/s") == pytest.approx(
            original.galcen_v_sun.d_z.to_value("km/s"), rel=1e-10
        )

    def test_default_galactocentric_roundtrip(self) -> None:
        """Test round-trip with default Galactocentric parameters."""
        # Test with coordinax defaults
        cx_original = cxastro.Galactocentric()
        apy_frame = convert(cx_original, apyc.Galactocentric)
        cx_result = convert(apy_frame, cxastro.Galactocentric)

        assert isinstance(cx_result, cxastro.Galactocentric)

        # Test with Astropy defaults
        apy_original = apyc.Galactocentric()
        cx_frame = convert(apy_original, cxastro.Galactocentric)
        apy_result = convert(cx_frame, apyc.Galactocentric)

        assert isinstance(apy_result, apyc.Galactocentric)
