"""Tests for astronomical frame transforms."""

__all__: tuple[str, ...] = ()

from collections.abc import Iterable

import equinox as eqx
import jax
import numpy as np
import plum
import pytest
from hypothesis import given

import quaxed.numpy as jnp
import unxt as u
import unxts.hypothesis as ust

import coordinax as cx
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import coordinax.vectors as cxv
import coordinaxs.astro as cxastro
import coordinaxs.hypothesis.astro as cxastrost
from coordinax.frames._src.base import is_same_frame
from coordinaxs.astro._src.galactic import ICRS_TO_GALACTIC_MATRIX

# Astropy is imported once, not re-checked inside five helper bodies -- but the
# skip stays per-test. A module-level `importorskip` would also skip the
# invariants below that never touch Astropy (the rotation-matrix properties,
# the NGP mapping, the dtype guard, and every round-trip property), which are
# exactly the ones worth still running in a minimal environment.
try:
    import astropy.coordinates as apyc
    import astropy.units as apyu

    HAS_ASTROPY = True
except ImportError:  # pragma: no cover - exercised only in minimal installs
    apyc = apyu = None
    HAS_ASTROPY = False

requires_astropy = pytest.mark.skipif(
    not HAS_ASTROPY, reason="astropy is not installed"
)

#: Bounded positions for the round-trip properties; written once, used four times.
POSITIONS_PC = ust.quantities(
    "pc", shape=(3,), elements={"min_value": -5e4, "max_value": 5e4}
)


def _to_np(x: object, unit: str) -> np.ndarray:
    assert isinstance(x, u.AbstractQuantity)
    return np.asarray(u.ustrip(unit, x), dtype=float)


def _as_astropy_galactocentric(frame: cxastro.Galactocentric):
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


def _astropy_xyz_pc(xyz_pc, *, frm, to) -> np.ndarray:
    """Transform a cartesian position between two astropy frames, in pc.

    One helper for both directions: the ICRS->GCF and GCF->ICRS references
    differed only in which frame went where.
    """
    x, y, z = xyz_pc
    sc = apyc.SkyCoord(
        x=x * apyu.pc,
        y=y * apyu.pc,
        z=z * apyu.pc,
        representation_type="cartesian",
        frame=frm,
    )
    out = sc.transform_to(to).cartesian
    return np.array(
        [out.x.to_value(apyu.pc), out.y.to_value(apyu.pc), out.z.to_value(apyu.pc)],
        dtype=float,
    )


@requires_astropy
@pytest.mark.parametrize("xyz_pc", [(0, 0, 0), (100, -20, 50), (-5000, 3200, 1200)])
def test_icrs_to_galactocentric_matches_astropy_positions(xyz_pc) -> None:
    """ICRS->Galactocentric position transforms match Astropy."""
    gcf = cxastro.Galactocentric()
    op = cxf.frame_transition(cxastro.ICRS(), gcf)

    got = cxfm.act(op, None, u.Q(jnp.asarray(xyz_pc), "pc")).ustrip("pc")
    expected = _astropy_xyz_pc(
        xyz_pc, frm=apyc.ICRS(), to=_as_astropy_galactocentric(gcf)
    )

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


@requires_astropy
@pytest.mark.parametrize(
    "xyz_pc", [(-8122, 0, 21), (-7800, 600, -200), (-9200, -500, 300)]
)
def test_galactocentric_to_icrs_matches_astropy_positions(xyz_pc) -> None:
    """Galactocentric->ICRS position transforms match Astropy."""
    gcf = cxastro.Galactocentric()
    op = cxf.frame_transition(gcf, cxastro.ICRS())

    got = cxfm.act(op, None, u.Q(jnp.asarray(xyz_pc), "pc")).ustrip("pc")
    expected = _astropy_xyz_pc(
        xyz_pc, frm=_as_astropy_galactocentric(gcf), to=apyc.ICRS()
    )

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


# ===================================================================
# Property-based tests


class TestFrameTransformProperties:
    """Hypothesis-driven property tests for ICRS <-> Galactocentric transforms."""

    @given(q=POSITIONS_PC)
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

    @given(q=POSITIONS_PC)
    def test_gcf_icrs_gcf_roundtrip(self, q: u.AbstractQuantity) -> None:
        """GCF → ICRS → GCF is the identity for arbitrary bounded positions."""
        icrs = cxastro.ICRS()
        gcf = cxastro.Galactocentric()

        fwd = cxf.frame_transition(gcf, icrs)
        bwd = cxf.frame_transition(icrs, gcf)

        back = cxfm.act(bwd, None, cxfm.act(fwd, None, q))
        np.testing.assert_allclose(back.ustrip("pc"), q.ustrip("pc"), rtol=0, atol=1e-6)

    @given(q=POSITIONS_PC)
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

    @requires_astropy
    @given(q=POSITIONS_PC)
    def test_icrs_to_gcf_matches_astropy_on_random_positions(
        self, q: u.AbstractQuantity
    ) -> None:
        """ICRS→GCF position matches Astropy for randomly generated positions."""
        gcf = cxastro.Galactocentric()
        op = cxf.frame_transition(cxastro.ICRS(), gcf)

        xyz = q.ustrip("pc")
        got = cxfm.act(op, None, q).ustrip("pc")
        expected = _astropy_xyz_pc(
            (xyz[0], xyz[1], xyz[2]),
            frm=apyc.ICRS(),
            to=_as_astropy_galactocentric(gcf),
        )
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


# ===================================================================
# Velocity (phase-space) transforms


def _astropy_icrs_to_gcf_phase_space(
    xyz_pc: Iterable[float], vxyz_kms: Iterable[float], frame: cxastro.Galactocentric
):
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


@requires_astropy
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


@requires_astropy
def test_custom_galcen_v_sun_velocities_match_astropy() -> None:
    """A non-default galcen_v_sun is honored and matches Astropy."""
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


@requires_astropy
@given(frame=cxastrost.galactocentric_frames())
def test_arbitrary_galactocentric_frames_match_astropy(frame) -> None:
    """ICRS->GCF matches Astropy for a frame with *all five* parameters drawn.

    The comparisons above pin `galcen_v_sun` and read `roll`, `z_sun` and
    `galcen` straight out of the default frame before mirroring them into
    Astropy, so a bug in any non-default parameter is structurally invisible
    to them. Here the whole frame is generated, and both the position and the
    velocity are checked -- the velocity because `z_sun` and `roll` enter the
    rotation while `galcen_v_sun` enters only the offset, so a position-only
    comparison would miss half of what the frame does.

    Agreement is ~1e-16 relative to the magnitude of the transformed vector;
    the tolerances below leave two orders of magnitude of headroom, and the
    `atol` is what covers a component that lands near zero while its
    siblings are of order kpc.
    """
    xyz_pc, vxyz_kms = (150.0, -220.0, 310.0), (30.0, -15.0, 22.0)
    op = cxf.frame_transition(cxastro.ICRS(), frame)

    out = cxfm.act(op, None, _coordinate(xyz_pc, vxyz_kms))
    got_q = np.array([_to_np(v, "pc") for v in out.point.data.values()])
    got_v = np.array([_to_np(v, "km/s") for v in out["velocity"].data.values()])

    exp_q, exp_v = _astropy_icrs_to_gcf_phase_space(xyz_pc, vxyz_kms, frame)
    np.testing.assert_allclose(got_q, exp_q, rtol=1e-13, atol=1e-9)
    np.testing.assert_allclose(got_v, exp_v, rtol=1e-13, atol=1e-9)


# ===================================================================
# Galactic frame


def _astropy_galactic_phase_space(xyz_pc, vxyz_kms, from_frame, to_frame):
    """Transform cartesian phase-space data between astropy frames."""
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


@requires_astropy
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


@requires_astropy
def test_galactic_to_galactocentric_via_fallback_matches_astropy() -> None:
    """Galactic->Galactocentric (generic route through ICRS) matches Astropy."""
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
        op, None, cxv.Coordinate(cx.cconvert(pt, cx.cart3d), vel=vel_cart), usys=usys
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


# ============================================================================
# The ICRS routing fallback needs base cases (#944)
#
# Frames declared at module scope, not inside tests: each class object is a
# fresh plum type, and rebuilding one per test would churn the dispatch cache.


class UnroutedFrame(cxastro.AbstractSpaceFrame):
    """An `AbstractSpaceFrame` with neither ICRS leg registered."""


class OneLegFrame(cxastro.AbstractSpaceFrame):
    """An `AbstractSpaceFrame` with only the *to*-ICRS leg registered."""


@plum.dispatch
def frame_transition(
    from_frame: OneLegFrame, to_frame: cxastro.ICRS, /
) -> cxfm.AbstractTransform:
    """Register the one leg `OneLegFrame` has."""
    del from_frame, to_frame
    return cxfm.Rotate.from_euler("z", u.Q(10, "deg"))


class TestAnUnroutedSpaceFrameSaysSoInsteadOfRecursing:
    """The `(AbstractSpaceFrame, AbstractSpaceFrame)` fallback routes via ICRS.

    `ICRS` is itself an `AbstractSpaceFrame`, so with no rule keyed on ICRS to
    stop the descent the fallback re-entered itself forever: `RecursionError`,
    reached without even writing a subclass because the abstract base was
    constructible. Both self-recursive calls always have ICRS on one side, so
    two base cases cover every route.
    """

    @pytest.mark.parametrize(
        ("frm", "to"),
        [
            (UnroutedFrame(), cxastro.icrs),
            (cxastro.icrs, UnroutedFrame()),
            (UnroutedFrame(), UnroutedFrame()),
            (UnroutedFrame(), cxastro.galactic),
            (cxastro.galactic, UnroutedFrame()),
            (UnroutedFrame(), cxastro.Galactocentric()),
        ],
        ids=["to-icrs", "from-icrs", "self", "to-galactic", "from-galactic", "to-gcf"],
    )
    def test_no_legs_registered(self, frm, to) -> None:
        with pytest.raises(cxf.FrameTransformError, match="No `frame_transition`"):
            cxf.frame_transition(frm, to)

    def test_one_leg_registered_still_goes_to_icrs(self) -> None:
        """The registered direction keeps working."""
        op = cxf.frame_transition(OneLegFrame(), cxastro.icrs)
        assert isinstance(op, cxfm.Rotate)

    @pytest.mark.parametrize(
        ("frm", "to"),
        [
            (cxastro.icrs, OneLegFrame()),
            (OneLegFrame(), OneLegFrame()),
            (cxastro.galactic, OneLegFrame()),
            (cxastro.Galactocentric(), OneLegFrame()),
        ],
        ids=["from-icrs", "self", "from-galactic", "from-gcf"],
    )
    def test_one_leg_registered_is_not_enough(self, frm, to) -> None:
        """Registering a single direction leaves the return leg blowing up."""
        with pytest.raises(cxf.FrameTransformError, match="registered from ICRS"):
            cxf.frame_transition(frm, to)

    def test_the_base_cases_do_not_shadow_the_working_paths(self) -> None:
        """The control: real ICRS legs are strictly more specific."""
        assert isinstance(
            cxf.frame_transition(cxastro.icrs, cxastro.icrs), cxfm.Identity
        )
        assert isinstance(
            cxf.frame_transition(cxastro.icrs, cxastro.galactic), cxfm.Rotate
        )
        assert isinstance(
            cxf.frame_transition(cxastro.galactic, cxastro.icrs), cxfm.Rotate
        )
        assert isinstance(
            cxf.frame_transition(cxastro.icrs, cxastro.Galactocentric()),
            cxfm.AbstractTransform,
        )
        assert isinstance(
            cxf.frame_transition(cxastro.galactic, cxastro.Galactocentric()),
            cxfm.AbstractTransform,
        )


def test_the_abstract_space_frame_cannot_be_built() -> None:
    """`AbstractSpaceFrame` is a dispatch category, not a frame (#954).

    It used to construct, which was the cheapest route into the ICRS-routing
    recursion above.
    """
    with pytest.raises(TypeError, match="Cannot instantiate abstract"):
        cxastro.AbstractSpaceFrame()


def test_the_concrete_space_frames_still_build() -> None:
    """The control: abstractness must not reach the subclasses."""
    assert isinstance(cxastro.ICRS(), cxastro.AbstractSpaceFrame)
    assert isinstance(cxastro.Galactic(), cxastro.AbstractSpaceFrame)
    assert isinstance(cxastro.Galactocentric(), cxastro.AbstractSpaceFrame)
    assert isinstance(UnroutedFrame(), cxastro.AbstractSpaceFrame)


class TestFrameTransitionUnderJit:
    """Regression for #963: a traced frame transition must not raise.

    ``frame_transition`` calls ``cxfm.simplify()`` internally with the default
    ``approx=True`` (fusing the ICRS bridge into one `Affine`), so a user who
    traces a frame transition could not opt out of the value checks -- they
    raised `jax.errors.TracerBoolConversionError` one layer below the public
    call. Now they decline to simplify, and the transition traces.
    """

    Q = u.Q([1.0, 2.0, 3.0], "kpc")

    #: gc -> gc(roll=10 deg), the #940 fall-through, computed eagerly.
    EXPECT_KPC = (1.00077259, 1.44822798, 3.30167987)

    @pytest.mark.parametrize(
        ("to_frame_type"),
        [cxastro.Galactocentric, cxastro.Galactic],
        ids=["ICRS->Galactocentric", "ICRS->Galactic"],
    )
    def test_the_transition_traces_and_agrees_with_eager(self, to_frame_type) -> None:
        a, b = cxastro.ICRS(), to_frame_type()

        def f(x, y):
            return cxf.frame_transition(x, y)(self.Q)

        got = jax.jit(f)(a, b)
        assert jnp.allclose(u.ustrip("kpc", got), u.ustrip("kpc", f(a, b)))

    def test_the_icrs_bridge_fall_through_traces(self) -> None:
        """The path #961's xfail exercises, reached without its fast-path fix.

        ``frame_transition(Galactocentric, Galactocentric)`` still raises in
        this tree, but at its own ``from_frame == to_frame`` check (#940/#961),
        not here -- so this drives the body that check falls through to. Once
        #961 lands, its ``test_the_whole_transition_under_jit`` xfail must lose
        its marker.
        """
        a = cxastro.Galactocentric()
        b = cxastro.Galactocentric(roll=u.Q(10, "deg"))
        icrs = cxastro.ICRS()

        def f(x, y):
            bridge = cxf.frame_transition(x, icrs) | cxf.frame_transition(icrs, y)
            return cxfm.simplify(bridge)(self.Q)

        got = jax.jit(f)(a, b)
class TestGalactocentricSelfTransition:
    """Regression for #940.

    ``from_frame == to_frame`` on a `Galactocentric` pair is a 0-d `jax.Array`,
    not a `bool`: fine eagerly, `TracerBoolConversionError` under `jit`. The
    check now goes through `is_same_frame`, which is concrete-only and
    structural.
    """

    #: gc -> gc(roll=10 deg) applied to Q([1, 2, 3], "kpc"), from before the fix.
    EXPECT_KPC = (1.00077259, 1.44822798, 3.30167987)

    def test_a_self_transition_is_the_identity(self) -> None:
        gcf = cxastro.Galactocentric()
        assert isinstance(cxf.frame_transition(gcf, gcf), cxfm.Identity)

    def test_equal_but_distinct_frames_are_the_identity(self) -> None:
        """A frame rebuilt from the same parameters is the same frame."""
        a, b = cxastro.Galactocentric(), cxastro.Galactocentric()
        assert a is not b
        assert isinstance(cxf.frame_transition(a, b), cxfm.Identity)

    @pytest.mark.parametrize(
        "jit", [jax.jit, eqx.filter_jit], ids=["jax.jit", "eqx.filter_jit"]
    )
    @pytest.mark.parametrize("same", [True, False], ids=["equal", "different"])
    def test_the_check_survives_tracing(self, jit, same) -> None:
        """Frames are pytrees, so they get passed as `jit` arguments.

        Under trace the answer is not statically knowable, so `is_same_frame` is
        `False` and the caller falls through to the general transform -- what it
        must never do is raise.
        """
        a = cxastro.Galactocentric()
        b = a if same else cxastro.Galactocentric(roll=u.Q(10, "deg"))
        out = jit(lambda x, y: jnp.asarray(is_same_frame(x, y)))(a, b)
        assert not bool(out)

    def test_different_frames_are_numerically_unchanged(self) -> None:
        """The non-equal case must not be touched by the fast-path change."""
        op = cxf.frame_transition(
            cxastro.Galactocentric(), cxastro.Galactocentric(roll=u.Q(10, "deg"))
        )
        got = op(u.Q([1.0, 2.0, 3.0], "kpc"))
        assert jnp.allclose(u.ustrip("kpc", got), jnp.asarray(self.EXPECT_KPC))

    @pytest.mark.xfail(
        raises=jax.errors.TracerBoolConversionError,
        strict=True,
        reason=(
            "the fast path is trace-safe now, but the fall-through still hits "
            "`simplify(Rotate)`'s `jnp.allclose` under trace (out of scope for "
            "#940); flip this to a plain test once that is fixed"
        ),
    )
    def test_the_whole_transition_under_jit(self) -> None:
        a, b = cxastro.Galactocentric(), cxastro.Galactocentric(roll=u.Q(10, "deg"))
        q = u.Q([1.0, 2.0, 3.0], "kpc")
        got = jax.jit(lambda x, y: cxf.frame_transition(x, y)(q))(a, b)
        assert jnp.allclose(u.ustrip("kpc", got), jnp.asarray(self.EXPECT_KPC))
