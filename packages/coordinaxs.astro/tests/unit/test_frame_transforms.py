"""Tests for astronomical frame transforms."""

__all__: tuple[str, ...] = ()

from collections.abc import Iterable

import numpy as np
import pytest
from hypothesis import given, settings

import quaxed.numpy as jnp
import unxt as u
import unxt_hypothesis as ust

import coordinax as cx
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import coordinax.vectors as cxv
import coordinaxs.astro as cxastro
from coordinaxs.astro._src.galactic import ICRS_TO_GALACTIC_MATRIX


def _to_np(x: object, unit: str) -> np.ndarray:
    assert isinstance(x, u.AbstractQuantity)
    return np.asarray(u.ustrip(unit, x), dtype=float)


def _as_astropy_galactocentric(frame: cxastro.Galactocentric):
    apyc = pytest.importorskip("astropy.coordinates")
    apyu = pytest.importorskip("astropy.units")

    galcen = frame.galcen.data
    galcen_coord = apyc.SkyCoord(
        ra=u.ustrip("deg", galcen["lon"]) * apyu.deg,
        dec=u.ustrip("deg", galcen["lat"]) * apyu.deg,
        distance=u.ustrip("kpc", galcen["distance"]) * apyu.kpc,
        frame="icrs",
    )
    v_sun = frame.galcen_v_sun.data
    kms = apyu.km / apyu.s
    return apyc.Galactocentric(
        galcen_coord=galcen_coord,
        galcen_distance=u.ustrip("kpc", galcen["distance"]) * apyu.kpc,
        z_sun=u.ustrip("pc", frame.z_sun) * apyu.pc,
        roll=u.ustrip("deg", frame.roll) * apyu.deg,
        galcen_v_sun=apyc.CartesianDifferential(
            d_x=u.ustrip("km/s", v_sun["x"]) * kms,
            d_y=u.ustrip("km/s", v_sun["y"]) * kms,
            d_z=u.ustrip("km/s", v_sun["z"]) * kms,
        ),
    )


def _astropy_icrs_to_gcf_xyz_pc(xyz_pc: Iterable[float], frame: cxastro.Galactocentric):
    apyc = pytest.importorskip("astropy.coordinates")
    apyu = pytest.importorskip("astropy.units")

    x, y, z = xyz_pc
    sc = apyc.SkyCoord(
        x=x * apyu.pc,
        y=y * apyu.pc,
        z=z * apyu.pc,
        representation_type="cartesian",
        frame=apyc.ICRS(),
    )
    out = sc.transform_to(_as_astropy_galactocentric(frame)).cartesian
    return np.array(
        [out.x.to_value(apyu.pc), out.y.to_value(apyu.pc), out.z.to_value(apyu.pc)],
        dtype=float,
    )


def _astropy_gcf_to_icrs_xyz_pc(xyz_pc: Iterable[float], frame: cxastro.Galactocentric):
    apyc = pytest.importorskip("astropy.coordinates")
    apyu = pytest.importorskip("astropy.units")

    x, y, z = xyz_pc
    gcf = _as_astropy_galactocentric(frame)
    sc = apyc.SkyCoord(
        x=x * apyu.pc,
        y=y * apyu.pc,
        z=z * apyu.pc,
        representation_type="cartesian",
        frame=gcf,
    )
    out = sc.transform_to(apyc.ICRS()).cartesian
    return np.array(
        [out.x.to_value(apyu.pc), out.y.to_value(apyu.pc), out.z.to_value(apyu.pc)],
        dtype=float,
    )


@pytest.mark.parametrize("xyz_pc", [(0, 0, 0), (100, -20, 50), (-5000, 3200, 1200)])
def test_icrs_to_galactocentric_matches_astropy_positions(xyz_pc) -> None:
    """ICRS->Galactocentric position transforms match Astropy."""
    gcf = cxastro.Galactocentric()
    op = cxf.frame_transition(cxastro.ICRS(), gcf)

    got = cxfm.act(op, None, u.Q(jnp.asarray(xyz_pc), "pc")).ustrip("pc")
    expected = _astropy_icrs_to_gcf_xyz_pc(xyz_pc, gcf)

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "xyz_pc", [(-8122, 0, 21), (-7800, 600, -200), (-9200, -500, 300)]
)
def test_galactocentric_to_icrs_matches_astropy_positions(xyz_pc) -> None:
    """Galactocentric->ICRS position transforms match Astropy."""
    gcf = cxastro.Galactocentric()
    op = cxf.frame_transition(gcf, cxastro.ICRS())

    got = cxfm.act(op, None, u.Q(jnp.asarray(xyz_pc), "pc")).ustrip("pc")
    expected = _astropy_gcf_to_icrs_xyz_pc(xyz_pc, gcf)

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_icrs_galactocentric_transitions_are_inverse_for_positions() -> None:
    """ICRS<->Galactocentric operators are inverses for position transforms."""
    icrs = cxastro.ICRS()
    gcf = cxastro.Galactocentric()

    fwd = cxf.frame_transition(icrs, gcf)
    bwd = cxf.frame_transition(gcf, icrs)

    q = u.Q(jnp.asarray([450, -100, 220]), "pc")
    back = cxfm.act(bwd, None, cxfm.act(fwd, None, q))

    np.testing.assert_allclose(_to_np(back, "pc"), _to_np(q, "pc"), rtol=0, atol=1e-6)


# ===================================================================
# Property-based tests


class TestFrameTransformProperties:
    """Hypothesis-driven property tests for ICRS <-> Galactocentric transforms."""

    @given(
        q=ust.quantities(
            "pc", shape=(3,), elements={"min_value": -5e4, "max_value": 5e4}
        )
    )
    @settings(deadline=None)
    def test_icrs_gcf_icrs_roundtrip(self, q: u.AbstractQuantity) -> None:
        """ICRS → GCF → ICRS is the identity for arbitrary bounded positions."""
        icrs = cxastro.ICRS()
        gcf = cxastro.Galactocentric()

        fwd = cxf.frame_transition(icrs, gcf)
        bwd = cxf.frame_transition(gcf, icrs)

        back = cxfm.act(bwd, None, cxfm.act(fwd, None, q))
        np.testing.assert_allclose(
            _to_np(back, "pc"), _to_np(q, "pc"), rtol=0, atol=1e-6
        )

    @given(
        q=ust.quantities(
            "pc", shape=(3,), elements={"min_value": -5e4, "max_value": 5e4}
        )
    )
    @settings(deadline=None)
    def test_gcf_icrs_gcf_roundtrip(self, q: u.AbstractQuantity) -> None:
        """GCF → ICRS → GCF is the identity for arbitrary bounded positions."""
        icrs = cxastro.ICRS()
        gcf = cxastro.Galactocentric()

        fwd = cxf.frame_transition(gcf, icrs)
        bwd = cxf.frame_transition(icrs, gcf)

        back = cxfm.act(bwd, None, cxfm.act(fwd, None, q))
        np.testing.assert_allclose(back.ustrip("pc"), q.ustrip("pc"), rtol=0, atol=1e-6)

    @given(
        q=ust.quantities(
            "pc", shape=(3,), elements={"min_value": -5e4, "max_value": 5e4}
        )
    )
    @settings(deadline=None)
    def test_inverse_is_frame_transition_in_reverse(
        self, q: u.AbstractQuantity
    ) -> None:
        """`.inverse` of ICRS→GCF operator equals `frame_transition(gcf,icrs)`.

        The active-semantics inverse law:
        ``(frame_transition(A, B)).inverse ≈ frame_transition(B, A)``.
        """
        icrs = cxastro.ICRS()
        gcf = cxastro.Galactocentric()

        fwd = cxf.frame_transition(icrs, gcf)
        bwd = cxf.frame_transition(gcf, icrs)

        q_gcf = cxfm.act(fwd, None, q)
        via_inverse = cxfm.act(fwd.inverse, None, q_gcf)
        via_bwd = cxfm.act(bwd, None, q_gcf)

        np.testing.assert_allclose(
            via_inverse.ustrip("pc"), via_bwd.ustrip("pc"), rtol=0, atol=1e-6
        )

    @given(
        q=ust.quantities(
            "pc", shape=(3,), elements={"min_value": -5e4, "max_value": 5e4}
        )
    )
    @settings(deadline=None)
    def test_icrs_to_gcf_matches_astropy_on_random_positions(
        self, q: u.AbstractQuantity
    ) -> None:
        """ICRS→GCF position matches Astropy for randomly generated positions."""
        gcf = cxastro.Galactocentric()
        op = cxf.frame_transition(cxastro.ICRS(), gcf)

        xyz = q.ustrip("pc")
        got = cxfm.act(op, None, q).ustrip("pc")
        expected = _astropy_icrs_to_gcf_xyz_pc((xyz[0], xyz[1], xyz[2]), gcf)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


# ===================================================================
# Velocity (phase-space) transforms


def _astropy_icrs_to_gcf_phase_space(
    xyz_pc: Iterable[float],
    vxyz_kms: Iterable[float],
    frame: cxastro.Galactocentric,
):
    apyc = pytest.importorskip("astropy.coordinates")
    apyu = pytest.importorskip("astropy.units")

    x, y, z = xyz_pc
    vx, vy, vz = vxyz_kms
    sc = apyc.SkyCoord(
        x=x * apyu.pc,
        y=y * apyu.pc,
        z=z * apyu.pc,
        v_x=vx * apyu.km / apyu.s,
        v_y=vy * apyu.km / apyu.s,
        v_z=vz * apyu.km / apyu.s,
        representation_type="cartesian",
        differential_type="cartesian",
        frame=apyc.ICRS(),
    )
    out = sc.transform_to(_as_astropy_galactocentric(frame))
    q = out.cartesian
    v = q.differentials["s"]
    kms = apyu.km / apyu.s
    return (
        np.array([q.x.to_value(apyu.pc), q.y.to_value(apyu.pc), q.z.to_value(apyu.pc)]),
        np.array([v.d_x.to_value(kms), v.d_y.to_value(kms), v.d_z.to_value(kms)]),
    )


def _coordinate(xyz_pc, vxyz_kms):
    return cx.Coordinate(
        point=cx.Point.from_(list(xyz_pc), "pc"),
        velocity=cx.Tangent.from_(list(vxyz_kms), "km/s"),
    )


@pytest.mark.parametrize(
    ("xyz_pc", "vxyz_kms"),
    [
        ((0, 0, 0), (0, 0, 0)),  # Sun at rest -> galcen_v_sun
        ((150, -220, 310), (30, -15, 22)),
        ((-5000, 3200, 1200), (-120, 80, 40)),
    ],
)
def test_icrs_to_galactocentric_matches_astropy_velocities(xyz_pc, vxyz_kms) -> None:
    """ICRS->Galactocentric phase-space transforms match Astropy."""
    gcf = cxastro.Galactocentric()
    op = cxf.frame_transition(cxastro.ICRS(), gcf)

    out = cxfm.act(op, None, _coordinate(xyz_pc, vxyz_kms))
    got_q = np.array([_to_np(v, "pc") for v in out.point.data.values()])
    got_v = np.array([_to_np(v, "km/s") for v in out["velocity"].data.values()])

    exp_q, exp_v = _astropy_icrs_to_gcf_phase_space(xyz_pc, vxyz_kms, gcf)
    np.testing.assert_allclose(got_q, exp_q, rtol=0, atol=1e-6)
    np.testing.assert_allclose(got_v, exp_v, rtol=0, atol=1e-6)


def test_star_at_rest_in_icrs_moves_with_solar_velocity() -> None:
    """A star at rest in ICRS has velocity galcen_v_sun in the GCF."""
    gcf = cxastro.Galactocentric()
    op = cxf.frame_transition(cxastro.ICRS(), gcf)

    out = cxfm.act(op, None, _coordinate((0, 0, 0), (0, 0, 0)))
    got_v = np.array([_to_np(v, "km/s") for v in out["velocity"].data.values()])
    exp_v = np.array([_to_np(v, "km/s") for v in gcf.galcen_v_sun.data.values()])
    np.testing.assert_allclose(got_v, exp_v, rtol=0, atol=1e-10)


def test_icrs_galactocentric_phase_space_roundtrip() -> None:
    """ICRS -> GCF -> ICRS is the identity on positions and velocities."""
    icrs = cxastro.ICRS()
    gcf = cxastro.Galactocentric()

    fwd = cxf.frame_transition(icrs, gcf)
    bwd = cxf.frame_transition(gcf, icrs)

    pv = _coordinate((450, -100, 220), (12.0, -34.0, 5.0))
    back = cxfm.act(bwd, None, cxfm.act(fwd, None, pv))

    np.testing.assert_allclose(
        np.array([_to_np(v, "pc") for v in back.point.data.values()]),
        np.array([_to_np(v, "pc") for v in pv.point.data.values()]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array([_to_np(v, "km/s") for v in back["velocity"].data.values()]),
        np.array([_to_np(v, "km/s") for v in pv["velocity"].data.values()]),
        rtol=0,
        atol=1e-9,
    )


def test_custom_galcen_v_sun_velocities_match_astropy() -> None:
    """A non-default galcen_v_sun is honored and matches Astropy."""
    apyc = pytest.importorskip("astropy.coordinates")
    apyu = pytest.importorskip("astropy.units")

    v_sun = cxv.Tangent.from_([11.1, 232.24, 7.25], "km/s")
    gcf = cxastro.Galactocentric(galcen_v_sun=v_sun)

    # coordinax side
    op = cxf.frame_transition(cxastro.ICRS(), gcf)
    out = cxfm.act(op, None, _coordinate((100, 200, -50), (5.0, -3.0, 8.0)))
    got_v = np.array([_to_np(v, "km/s") for v in out["velocity"].data.values()])

    # astropy side (rebuild the frame with the custom v_sun)
    apy_frame = _as_astropy_galactocentric(gcf)
    kms = apyu.km / apyu.s
    apy_frame = apyc.Galactocentric(
        galcen_coord=apy_frame.galcen_coord,
        galcen_distance=apy_frame.galcen_distance,
        z_sun=apy_frame.z_sun,
        roll=apy_frame.roll,
        galcen_v_sun=apyc.CartesianDifferential(
            d_x=11.1 * kms, d_y=232.24 * kms, d_z=7.25 * kms
        ),
    )
    sc = apyc.SkyCoord(
        x=100 * apyu.pc,
        y=200 * apyu.pc,
        z=-50 * apyu.pc,
        v_x=5.0 * kms,
        v_y=-3.0 * kms,
        v_z=8.0 * kms,
        representation_type="cartesian",
        differential_type="cartesian",
        frame=apyc.ICRS(),
    )
    v = sc.transform_to(apy_frame).cartesian.differentials["s"]
    exp_v = np.array([v.d_x.to_value(kms), v.d_y.to_value(kms), v.d_z.to_value(kms)])
    np.testing.assert_allclose(got_v, exp_v, rtol=0, atol=1e-6)


# ===================================================================
# Galactic frame


def _astropy_galactic_phase_space(xyz_pc, vxyz_kms, from_frame, to_frame):
    """Transform cartesian phase-space data between astropy frames."""
    apyc = pytest.importorskip("astropy.coordinates")
    apyu = pytest.importorskip("astropy.units")

    kms = apyu.km / apyu.s
    rep = apyc.CartesianRepresentation(
        x=xyz_pc[0] * apyu.pc,
        y=xyz_pc[1] * apyu.pc,
        z=xyz_pc[2] * apyu.pc,
        differentials=apyc.CartesianDifferential(
            d_x=vxyz_kms[0] * kms, d_y=vxyz_kms[1] * kms, d_z=vxyz_kms[2] * kms
        ),
    )
    out = from_frame.realize_frame(rep).transform_to(to_frame).cartesian
    v = out.differentials["s"]
    return (
        np.array(
            [out.x.to_value(apyu.pc), out.y.to_value(apyu.pc), out.z.to_value(apyu.pc)]
        ),
        np.array([v.d_x.to_value(kms), v.d_y.to_value(kms), v.d_z.to_value(kms)]),
    )


@pytest.mark.parametrize(
    ("xyz_pc", "vxyz_kms"),
    [
        ((100, 0, 0), (0, 0, 0)),
        ((150, -220, 310), (30, -15, 22)),
        ((-5000, 3200, 1200), (-120, 80, 40)),
    ],
)
def test_icrs_to_galactic_matches_astropy(xyz_pc, vxyz_kms) -> None:
    """ICRS->Galactic phase-space transforms match Astropy."""
    apyc = pytest.importorskip("astropy.coordinates")

    op = cxf.frame_transition(cxastro.icrs, cxastro.galactic)
    out = cxfm.act(op, None, _coordinate(xyz_pc, vxyz_kms))
    got_q = np.array([_to_np(v, "pc") for v in out.point.data.values()])
    got_v = np.array([_to_np(v, "km/s") for v in out["velocity"].data.values()])

    exp_q, exp_v = _astropy_galactic_phase_space(
        xyz_pc, vxyz_kms, apyc.ICRS(), apyc.Galactic()
    )
    np.testing.assert_allclose(got_q, exp_q, rtol=0, atol=1e-8)
    np.testing.assert_allclose(got_v, exp_v, rtol=0, atol=1e-8)


def test_galactic_icrs_roundtrip() -> None:
    """ICRS -> Galactic -> ICRS is the identity on positions and velocities."""
    fwd = cxf.frame_transition(cxastro.icrs, cxastro.galactic)
    bwd = cxf.frame_transition(cxastro.galactic, cxastro.icrs)

    pv = _coordinate((450, -100, 220), (12.0, -34.0, 5.0))
    back = cxfm.act(bwd, None, cxfm.act(fwd, None, pv))

    np.testing.assert_allclose(
        np.array([_to_np(v, "pc") for v in back.point.data.values()]),
        np.array([_to_np(v, "pc") for v in pv.point.data.values()]),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.array([_to_np(v, "km/s") for v in back["velocity"].data.values()]),
        np.array([_to_np(v, "km/s") for v in pv["velocity"].data.values()]),
        rtol=0,
        atol=1e-12,
    )


def test_galactic_rotation_is_orthogonal() -> None:
    """The ICRS->Galactic rotation matrix is a proper rotation."""
    R = np.asarray(ICRS_TO_GALACTIC_MATRIX)
    np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=0, atol=1e-13)


def test_ngp_maps_to_z_axis() -> None:
    """The North Galactic Pole (ICRS) maps to +z in Galactic coordinates."""
    op = cxf.frame_transition(cxastro.icrs, cxastro.galactic)
    ngp = cx.Point.from_(
        {
            "lon": u.Q(192.8594812065348, "deg"),
            "lat": u.Q(27.12825118085622, "deg"),
            "distance": u.Q(1.0, "kpc"),
        },
        cx.lonlat_sph3d,
    )
    out = cxfm.act(op, None, ngp).cconvert(cx.cart3d)
    got = np.array([_to_np(v, "kpc") for v in out.data.values()])
    np.testing.assert_allclose(got, [0.0, 0.0, 1.0], rtol=0, atol=1e-7)


def test_galactic_to_galactocentric_via_fallback_matches_astropy() -> None:
    """Galactic->Galactocentric (generic route through ICRS) matches Astropy."""
    apyc = pytest.importorskip("astropy.coordinates")

    gcf = cxastro.Galactocentric()
    op = cxf.frame_transition(cxastro.galactic, gcf)

    xyz, vxyz = (150, -220, 310), (30.0, -15.0, 22.0)
    out = cxfm.act(op, None, _coordinate(xyz, vxyz))
    got_q = np.array([_to_np(v, "pc") for v in out.point.data.values()])
    got_v = np.array([_to_np(v, "km/s") for v in out["velocity"].data.values()])

    exp_q, exp_v = _astropy_galactic_phase_space(
        xyz, vxyz, apyc.Galactic(), _as_astropy_galactocentric(gcf)
    )
    np.testing.assert_allclose(got_q, exp_q, rtol=0, atol=1e-6)
    np.testing.assert_allclose(got_v, exp_v, rtol=0, atol=1e-6)


def test_galactic_matrix_is_float64():
    """The Galactic rotation constant keeps float64 regardless of the x64 flag.

    A JAX-array constant would be silently truncated to float32 at import
    time under jax_enable_x64=False, discarding precision before use.
    """
    assert isinstance(ICRS_TO_GALACTIC_MATRIX, np.ndarray)
    assert ICRS_TO_GALACTIC_MATRIX.dtype == np.float64


def test_galactocentric_spherical_velocity_fibre():
    """The velocity kick handles non-Cartesian velocity fibres.

    Proper-motion-style (spherical) velocity data must traverse the
    ICRS->Galactocentric chain and agree with the same physics computed
    from a Cartesian velocity fibre.
    """
    usys = u.unitsystems.galactic
    op = cxf.frame_transition(cxastro.ICRS(), cxastro.Galactocentric())
    pt = cxv.Point.from_(
        {"lon": u.Q(30.0, "deg"), "lat": u.Q(10.0, "deg"), "distance": u.Q(1.0, "kpc")},
        cx.lonlat_sph3d,
    )
    vel_sph = cxv.Tangent(
        {
            "lon": u.Q(1e-12, "rad/s"),
            "lat": u.Q(0.0, "rad/s"),
            "distance": u.Q(10.0, "km/s"),
        },
        cx.lonlat_sph3d,
        cxr.coord_basis,
        cxr.vel,
    )
    out = cx.act(op, None, cxv.Coordinate(pt, vel=vel_sph), usys=usys)

    # reference: same input with a Cartesian velocity fibre
    vel_cart = cx.cconvert(vel_sph, cx.cart3d, at=pt.data, usys=usys)
    out_ref = cx.act(
        op,
        None,
        cxv.Coordinate(cx.cconvert(pt, cx.cart3d), vel=vel_cart),
        usys=usys,
    )
    out_v = cx.cconvert(
        out._data["vel"],
        cx.cart3d,
        at=cx.cconvert(out.point, out._data["vel"].chart).data,
        usys=usys,
    )
    for k in "xyz":
        a = u.ustrip("km/s", out_v.data[k])
        b = u.ustrip("km/s", out_ref._data["vel"].data[k])
        assert jnp.allclose(a, b, rtol=1e-5)
