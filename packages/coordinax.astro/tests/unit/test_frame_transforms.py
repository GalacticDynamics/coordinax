"""Tests for astronomical frame transforms."""

__all__: tuple[str, ...] = ()

from collections.abc import Iterable

import numpy as np
import pytest
from hypothesis import given, settings

import quaxed.numpy as jnp
import unxt as u
import unxt_hypothesis as ust

import coordinax.astro as cxastro
import coordinax.frames as cxf
import coordinax.transforms as cxfm


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
    return apyc.Galactocentric(
        galcen_coord=galcen_coord,
        galcen_distance=u.ustrip("kpc", galcen["distance"]) * apyu.kpc,
        z_sun=u.ustrip("pc", frame.z_sun) * apyu.pc,
        roll=u.ustrip("deg", frame.roll) * apyu.deg,
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
        [
            out.x.to_value(apyu.pc),
            out.y.to_value(apyu.pc),
            out.z.to_value(apyu.pc),
        ],
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
        [
            out.x.to_value(apyu.pc),
            out.y.to_value(apyu.pc),
            out.z.to_value(apyu.pc),
        ],
        dtype=float,
    )


@pytest.mark.parametrize(
    "xyz_pc",
    [
        (0.0, 0.0, 0.0),
        (100.0, -20.0, 50.0),
        (-5000.0, 3200.0, 1200.0),
    ],
)
def test_icrs_to_galactocentric_matches_astropy_positions(xyz_pc) -> None:
    """ICRS->Galactocentric position transforms match Astropy."""
    gcf = cxastro.Galactocentric()
    op = cxf.frame_transition(cxastro.ICRS(), gcf)  # type: ignore[no-untyped-call]

    got = _to_np(cxfm.act(op, None, u.Q(jnp.asarray(xyz_pc), "pc")), "pc")
    expected = _astropy_icrs_to_gcf_xyz_pc(xyz_pc, gcf)

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    "xyz_pc",
    [
        (-8122.0, 0.0, 21.0),
        (-7800.0, 600.0, -200.0),
        (-9200.0, -500.0, 300.0),
    ],
)
def test_galactocentric_to_icrs_matches_astropy_positions(xyz_pc) -> None:
    """Galactocentric->ICRS position transforms match Astropy."""
    gcf = cxastro.Galactocentric()
    op = cxf.frame_transition(gcf, cxastro.ICRS())  # type: ignore[no-untyped-call]

    got = _to_np(cxfm.act(op, None, u.Q(jnp.asarray(xyz_pc), "pc")), "pc")
    expected = _astropy_gcf_to_icrs_xyz_pc(xyz_pc, gcf)

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-6)


def test_icrs_galactocentric_transitions_are_inverse_for_positions() -> None:
    """ICRS<->Galactocentric operators are inverses for position transforms."""
    icrs = cxastro.ICRS()  # type: ignore[no-untyped-call]
    gcf = cxastro.Galactocentric()

    fwd = cxf.frame_transition(icrs, gcf)
    bwd = cxf.frame_transition(gcf, icrs)

    q = u.Q(jnp.asarray([450.0, -100.0, 220.0]), "pc")
    back = cxfm.act(bwd, None, cxfm.act(fwd, None, q))

    np.testing.assert_allclose(_to_np(back, "pc"), _to_np(q, "pc"), rtol=0.0, atol=1e-6)


# ===================================================================
# Property-based tests


class TestFrameTransformProperties:
    """Hypothesis-driven property tests for ICRS <-> Galactocentric transforms."""

    @given(
        q=ust.quantities(
            "pc",
            shape=(3,),
            elements={"min_value": -5e4, "max_value": 5e4},
        )
    )
    @settings(deadline=None)
    def test_icrs_gcf_icrs_roundtrip(self, q: u.AbstractQuantity) -> None:
        """ICRS → GCF → ICRS is the identity for arbitrary bounded positions."""
        icrs = cxastro.ICRS()  # type: ignore[no-untyped-call]
        gcf = cxastro.Galactocentric()

        fwd = cxf.frame_transition(icrs, gcf)
        bwd = cxf.frame_transition(gcf, icrs)

        back = cxfm.act(bwd, None, cxfm.act(fwd, None, q))
        np.testing.assert_allclose(
            _to_np(back, "pc"), _to_np(q, "pc"), rtol=0.0, atol=1e-6
        )

    @given(
        q=ust.quantities(
            "pc",
            shape=(3,),
            elements={"min_value": -5e4, "max_value": 5e4},
        )
    )
    @settings(deadline=None)
    def test_gcf_icrs_gcf_roundtrip(self, q: u.AbstractQuantity) -> None:
        """GCF → ICRS → GCF is the identity for arbitrary bounded positions."""
        icrs = cxastro.ICRS()  # type: ignore[no-untyped-call]
        gcf = cxastro.Galactocentric()

        fwd = cxf.frame_transition(gcf, icrs)
        bwd = cxf.frame_transition(icrs, gcf)

        back = cxfm.act(bwd, None, cxfm.act(fwd, None, q))
        np.testing.assert_allclose(
            _to_np(back, "pc"), _to_np(q, "pc"), rtol=0.0, atol=1e-6
        )

    @given(
        q=ust.quantities(
            "pc",
            shape=(3,),
            elements={"min_value": -5e4, "max_value": 5e4},
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
        icrs = cxastro.ICRS()  # type: ignore[no-untyped-call]
        gcf = cxastro.Galactocentric()

        fwd = cxf.frame_transition(icrs, gcf)
        bwd = cxf.frame_transition(gcf, icrs)

        q_gcf = cxfm.act(fwd, None, q)
        via_inverse = cxfm.act(fwd.inverse, None, q_gcf)
        via_bwd = cxfm.act(bwd, None, q_gcf)

        np.testing.assert_allclose(
            _to_np(via_inverse, "pc"), _to_np(via_bwd, "pc"), rtol=0.0, atol=1e-6
        )

    @given(
        q=ust.quantities(
            "pc",
            shape=(3,),
            elements={"min_value": -5e4, "max_value": 5e4},
        )
    )
    @settings(deadline=None)
    def test_icrs_to_gcf_matches_astropy_on_random_positions(
        self, q: u.AbstractQuantity
    ) -> None:
        """ICRS→GCF position matches Astropy for randomly generated positions."""
        gcf = cxastro.Galactocentric()
        op = cxf.frame_transition(cxastro.ICRS(), gcf)  # type: ignore[no-untyped-call]

        xyz = _to_np(q, "pc")
        got = _to_np(cxfm.act(op, None, q), "pc")
        expected = _astropy_icrs_to_gcf_xyz_pc((xyz[0], xyz[1], xyz[2]), gcf)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-6)
