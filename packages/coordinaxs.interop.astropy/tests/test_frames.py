"""Frame interop: the guards and the shape of ``galcen_coord``.

Two defects, both in ``coordinaxs.interop.astropy._src.frames``: `plum.convert`
used to return a bare frame for an astropy frame carrying data, dropping it
silently, while the paired ``from_`` raised; and the cx -> astropy
Galactocentric direction attached a distance to ``galcen_coord``, which astropy
itself cannot compare or transform.
"""

__all__: tuple[str, ...] = ()

import astropy.coordinates as apyc
import astropy.units as apyu
import jax
import numpy as np
import plum
import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.frames as cxf
import coordinax.transforms as cxfm
import coordinaxs.astro as cxastro
import jax.tree_util as jtu
from plum import convert
import coordinaxs.interop.astropy  # noqa: F401


@pytest.mark.parametrize(
    ("apy_frame", "cx_frame_cls"),
    [
        (
            apyc.ICRS(ra=10 * apyu.deg, dec=20 * apyu.deg, distance=1 * apyu.kpc),
            cxastro.ICRS,
        ),
        (apyc.Galactic(l=1 * apyu.deg, b=2 * apyu.deg), cxastro.Galactic),
        (
            apyc.Galactocentric(x=1 * apyu.kpc, y=2 * apyu.kpc, z=3 * apyu.kpc),
            cxastro.Galactocentric,
        ),
    ],
    ids=["icrs", "galactic", "galactocentric"],
)
def test_convert_refuses_a_frame_carrying_data(apy_frame, cx_frame_cls):
    """`plum.convert` refuses data rather than dropping it, as ``from_`` does."""
    with pytest.raises(ValueError, match="must not have data"):
        plum.convert(apy_frame, cx_frame_cls)

    with pytest.raises(ValueError, match="must not have data"):
        cx_frame_cls.from_(apy_frame)


class TestGalactocentricGalcenCoord:
    """``galcen_coord`` is a direction; the distance is ``galcen_distance``."""

    @staticmethod
    def round_trip(apy_frame):
        return plum.convert(
            cxastro.Galactocentric.from_(apy_frame), apyc.Galactocentric
        )

    def test_galcen_coord_is_unit_spherical(self):
        a1 = self.round_trip(apyc.Galactocentric())
        assert isinstance(a1.galcen_coord.data, apyc.UnitSphericalRepresentation)

    def test_the_round_tripped_frame_is_equivalent_to_astropys_own(self):
        """A `SphericalRepresentation` here made this a `TypeError`."""
        a0 = apyc.Galactocentric()
        assert a0.is_equivalent_frame(self.round_trip(a0))

    def test_the_transform_is_unchanged(self):
        """Dropping the distance off ``galcen_coord`` does not move the answer.

        The bound is headroom, not observed error: on this machine the two
        transforms agree bit-for-bit. 1e-6 pc is ~1e6x above the double-
        precision floor for a ~9 kpc vector (~2e-12 pc), so a libm or Astropy
        version difference cannot trip it -- while still far below any real
        regression: stripping ``galcen_distance`` by mistake (rather than the
        distance off ``galcen_coord``) moves the point by ~7 kpc, and the
        bound catches a ``galcen_coord`` direction wrong by a quarter of a
        milliarcsecond.

        ``separation_3d`` compares directly rather than transforming only
        because the two frames are equivalent -- which the test above pins.
        """
        atol_pc = 1e-6
        a0 = apyc.Galactocentric()
        sc = apyc.SkyCoord(ra=90 * apyu.deg, dec=45 * apyu.deg, distance=1 * apyu.kpc)
        g0 = sc.transform_to(a0)
        g1 = sc.transform_to(self.round_trip(a0))
        assert np.allclose(
            g0.cartesian.xyz.to_value("pc"),
            g1.cartesian.xyz.to_value("pc"),
            rtol=0.0,
            atol=atol_pc,
        )
        assert g0.separation_3d(g1).to_value("pc") < atol_pc

    def test_the_frame_parameters_survive_the_round_trip(self):
        """`roll` comes back a float rather than a weak int, hence `allclose`."""
        a0 = apyc.Galactocentric(
            galcen_coord=apyc.ICRS(ra=1 * apyu.deg, dec=2 * apyu.deg),
            galcen_distance=8.0 * apyu.kpc,
            z_sun=15.0 * apyu.pc,
            roll=3 * apyu.deg,
        )
        a1 = self.round_trip(a0)
        assert np.allclose(a1.galcen_coord.ra.to_value("deg"), 1.0)
        assert np.allclose(a1.galcen_coord.dec.to_value("deg"), 2.0)
        assert np.allclose(a1.galcen_distance.to_value("kpc"), 8.0)
        assert np.allclose(a1.z_sun.to_value("pc"), 15.0)
        assert np.allclose(a1.roll.to_value("deg"), 3.0)


KMS = apyu.km / apyu.s

#: The four astropy frames with no coordinax counterpart. `FK5` appears twice
#: on purpose: a non-default equinox must not accidentally find a dispatch.
UNSUPPORTED = [
    pytest.param(apyc.FK5(), id="fk5"),
    pytest.param(apyc.FK5(equinox="J1975"), id="fk5-j1975"),
    pytest.param(apyc.FK4(), id="fk4"),
    pytest.param(
        apyc.AltAz(
            obstime="2020-01-01T00:00:00",
            location=apyc.EarthLocation(
                lat=-30.0 * apyu.deg, lon=-70.0 * apyu.deg, height=2000.0 * apyu.m
            ),
        ),
        id="altaz",
    ),
]


# ---------------------------------------------------------------------------
# Round trips


@pytest.mark.parametrize(
    ("apy", "cls"),
    [
        pytest.param(apyc.ICRS(), cxastro.ICRS, id="icrs"),
        pytest.param(apyc.Galactic(), cxastro.Galactic, id="galactic"),
    ],
)
def test_parameterless_frames_round_trip(apy, cls) -> None:
    """The two parameter-free frames survive a round trip through `from_`.

    `plum.convert` short-circuits these to the module singletons and never
    reaches the `from_` bodies, so the constructor is called directly.
    """
    frame = cls.from_(apy)
    assert isinstance(frame, cls)
    assert isinstance(plum.convert(apy, cls), cls)
    assert type(plum.convert(frame, type(apy))) is type(apy)
    assert type(plum.convert(frame, apyc.BaseCoordinateFrame)) is type(apy)


def test_astropy_galactocentric_round_trip() -> None:
    """Astropy -> coordinax -> Astropy preserves every frame parameter."""
    apy = apyc.Galactocentric()
    back = plum.convert(plum.convert(apy, cxastro.Galactocentric), apyc.Galactocentric)

    assert back.galcen_coord.ra.to_value("deg") == pytest.approx(
        apy.galcen_coord.ra.to_value("deg")
    )
    assert back.galcen_coord.dec.to_value("deg") == pytest.approx(
        apy.galcen_coord.dec.to_value("deg")
    )
    assert back.galcen_distance.to_value("kpc") == pytest.approx(
        apy.galcen_distance.to_value("kpc")
    )
    assert back.roll.to_value("deg") == pytest.approx(apy.roll.to_value("deg"))
    assert back.z_sun.to_value("pc") == pytest.approx(apy.z_sun.to_value("pc"))
    for comp in ("d_x", "d_y", "d_z"):
        assert getattr(back.galcen_v_sun, comp).to_value(KMS) == pytest.approx(
            getattr(apy.galcen_v_sun, comp).to_value(KMS)
        )


def test_round_trip_does_not_invent_a_galcen_coord_distance() -> None:
    """The round trip must not give ``galcen_coord`` a physical distance.

    Was `xfail(reason="#947", strict=True)`; #965 fixed it on main, so the
    rebase turned it into an `XPASS(strict)`. Now a live regression test.

    Astropy's default ``galcen_coord`` is a direction: its spherical distance
    is the dimensionless 1.0 that `UnitSphericalRepresentation` carries. The
    coordinax->astropy conversion builds it from a full
    `SphericalRepresentation`, so it comes back as ``galcen_distance``,
    which makes the two frames compare unequal and changes what
    ``galcen_coord`` means.
    """
    apy = apyc.Galactocentric()
    assert isinstance(apy.galcen_coord.data, apyc.UnitSphericalRepresentation)

    back = plum.convert(plum.convert(apy, cxastro.Galactocentric), apyc.Galactocentric)
    assert isinstance(back.galcen_coord.data, apyc.UnitSphericalRepresentation)


def test_coordinax_galactocentric_round_trip() -> None:
    """Coordinax -> Astropy -> coordinax preserves every frame parameter."""
    frame = cxastro.Galactocentric()
    back = plum.convert(
        plum.convert(frame, apyc.Galactocentric), cxastro.Galactocentric
    )

    for key, unit in (("lon", "deg"), ("lat", "deg"), ("distance", "kpc")):
        assert back.galcen[key].ustrip(unit) == pytest.approx(
            frame.galcen[key].ustrip(unit)
        )
    assert back.roll.ustrip("deg") == pytest.approx(frame.roll.ustrip("deg"))
    assert back.z_sun.ustrip("pc") == pytest.approx(frame.z_sun.ustrip("pc"))
    for key in "xyz":
        assert back.galcen_v_sun.data[key].ustrip("km/s") == pytest.approx(
            frame.galcen_v_sun.data[key].ustrip("km/s")
        )


# ---------------------------------------------------------------------------
# Unsupported sources


@pytest.mark.parametrize("frame", UNSUPPORTED)
def test_unsupported_astropy_frame_is_refused_by_convert(frame) -> None:
    """`plum.convert` has no route from these astropy frames."""
    with pytest.raises(TypeError):
        plum.convert(frame, cxastro.AbstractSpaceFrame)


@pytest.mark.xfail(
    reason="#968: KeyError('AbstractReferenceFrame') masks the "
    "NotFoundLookupError whenever runtime typechecking is on",
    strict=True,
)
@pytest.mark.parametrize("frame", UNSUPPORTED)
def test_unsupported_astropy_frame_is_refused_by_from_(frame) -> None:
    """`from_` has no dispatch for these astropy frames.

    What is raised instead is `KeyError`. The first implementation registered
    on `AbstractReferenceFrame.from_` is a jaxtyping wrapper whenever
    ``COORDINAX_ENABLE_RUNTIME_TYPECHECKING`` is set -- which is how the suite
    itself runs -- so ``plum.Function._f.__globals__`` is jaxtyping's module
    namespace. plum resolves a method's owner by looking its class name up in
    exactly that namespace, so building the "could not be resolved" message
    dies with ``KeyError('AbstractReferenceFrame')`` and the caller never sees
    which argument was wrong. The `plum.convert` direction above is
    unaffected, which is why only this half is marked.
    """
    with pytest.raises(plum.NotFoundLookupError):
        cxastro.Galactocentric.from_(frame)


@pytest.mark.parametrize(
    ("frame", "cls"),
    [
        pytest.param(
            apyc.ICRS(ra=10.0 * apyu.deg, dec=20.0 * apyu.deg), cxastro.ICRS, id="icrs"
        ),
        pytest.param(
            apyc.Galactic(l=10.0 * apyu.deg, b=20.0 * apyu.deg),
            cxastro.Galactic,
            id="galactic",
        ),
        pytest.param(
            apyc.Galactocentric(
                apyc.CartesianRepresentation(
                    x=1.0 * apyu.kpc, y=2.0 * apyu.kpc, z=3.0 * apyu.kpc
                )
            ),
            cxastro.Galactocentric,
            id="galactocentric",
        ),
    ],
)
def test_astropy_frame_carrying_data_is_refused(frame, cls) -> None:
    """A frame is a frame; one carrying coordinates is not convertible.

    Astropy conflates the two -- a `BaseCoordinateFrame` may or may not hold
    data -- while a coordinax frame never does, so dropping the data silently
    would lose a caller's coordinates without a word. Each of the three
    conversions has its own guard, hence all three are checked.
    """
    assert frame.has_data  # guard the premise
    with pytest.raises(ValueError, match="must not have data"):
        cls.from_(frame)


# ---------------------------------------------------------------------------
# Contradictory galcen_coord distance


def test_contradictory_galcen_distance_follows_astropy() -> None:
    """``galcen_distance`` wins over a distance carried by ``galcen_coord``.

    Astropy resolves the contradiction that way -- the Sun sits at
    ``-galcen_distance`` along x whatever ``galcen_coord.distance`` says --
    and the conversion has to agree, or a frame built this way would mean two
    different things in the two libraries. Both halves are asserted, so the
    guard fails if Astropy ever changes its mind.
    """
    galcen_coord = apyc.SkyCoord(
        ra=266.4051 * apyu.deg,
        dec=-28.936175 * apyu.deg,
        distance=3.0 * apyu.kpc,  # contradicts galcen_distance below
        frame="icrs",
    )
    apy = apyc.Galactocentric(
        galcen_coord=galcen_coord, galcen_distance=8.122 * apyu.kpc
    )

    # Astropy itself: the Sun sits galcen_distance from the origin, not 3 kpc.
    sun = apyc.SkyCoord(
        x=0 * apyu.pc,
        y=0 * apyu.pc,
        z=0 * apyu.pc,
        representation_type="cartesian",
        frame=apyc.ICRS(),
    )
    sun_distance = sun.transform_to(apy).cartesian.norm().to_value("kpc")
    assert sun_distance == pytest.approx(8.122)

    # ... and the converted coordinax frame carries galcen_distance too.
    frame = plum.convert(apy, cxastro.Galactocentric)
    assert frame.galcen["distance"].ustrip("kpc") == pytest.approx(8.122)


# ---------------------------------------------------------------------------
# Batching and vmap


def test_converted_frame_transforms_batched_input() -> None:
    """A converted frame handles a ``(200,)`` batch, elementwise."""
    frame = plum.convert(apyc.Galactocentric(), cxastro.Galactocentric)
    op = cxf.frame_transition(cxastro.ICRS(), frame)

    xyz = np.linspace(-5e3, 5e3, 600).reshape(200, 3)
    batched = cxfm.act(op, None, u.Q(jnp.asarray(xyz), "pc")).ustrip("pc")
    assert batched.shape == (200, 3)

    one_by_one = np.stack(
        [
            np.asarray(cxfm.act(op, None, u.Q(jnp.asarray(row), "pc")).ustrip("pc"))
            for row in xyz[:5]
        ]
    )
    np.testing.assert_allclose(
        np.asarray(batched)[:5], one_by_one, rtol=1e-12, atol=1e-9
    )


def test_converted_frame_works_under_vmap() -> None:
    """`jax.vmap` over a converted frame matches the batched transform."""
    frame = plum.convert(apyc.Galactocentric(), cxastro.Galactocentric)
    op = cxf.frame_transition(cxastro.ICRS(), frame)

    xyz = jnp.asarray(np.linspace(-5e3, 5e3, 600).reshape(200, 3))
    mapped = jax.vmap(lambda v: cxfm.act(op, None, u.Q(v, "pc")).ustrip("pc"))(xyz)
    batched = cxfm.act(op, None, u.Q(xyz, "pc")).ustrip("pc")

    assert mapped.shape == (200, 3)
    np.testing.assert_allclose(
        np.asarray(mapped), np.asarray(batched), rtol=1e-12, atol=1e-9
    )


# ---------------------------------------------------------------------------
# dtype pass-through


def test_float32_z_sun_is_not_promoted() -> None:
    """A float32 astropy parameter stays float32 on the coordinax side.

    Silently widening to float64 would hide a caller's deliberate choice of
    precision, and under ``jax_enable_x64=False`` the reverse -- a forced
    float64 -- is what actually happens, so the direction matters.
    """
    apy = apyc.Galactocentric(z_sun=np.float32(20.8) * apyu.pc)
    assert apy.z_sun.dtype == np.float32  # guard the premise

    frame = plum.convert(apy, cxastro.Galactocentric)
    assert frame.z_sun.dtype == np.float32
# Each coordinax frame paired with its one supported Astropy frame class.
FRAME_PAIRS = [
    (cxastro.ICRS(), apyc.ICRS),
    (cxastro.Galactic(), apyc.Galactic),
    (cxastro.Galactocentric(), apyc.Galactocentric),
]

# Astropy classes used as conversion *targets* that are unsupported for at
# least one coordinax frame. Distinct from `UNSUPPORTED` above, which is the
# astropy->coordinax direction; the per-pair skip drops supported combinations.
UNSUPPORTED_TARGETS = [apyc.FK5, apyc.FK4, apyc.Galactic, apyc.AltAz]


@pytest.mark.parametrize(("cx_frame", "apy_cls"), FRAME_PAIRS)
def test_roundtrip_exact_class(
    cx_frame: cxastro.AbstractSpaceFrame, apy_cls: type[apyc.BaseCoordinateFrame]
) -> None:
    """Conversion to the exact Astropy class round-trips."""
    apy_frame = convert(cx_frame, apy_cls)
    assert isinstance(apy_frame, apy_cls)
    assert type(apy_frame) is apy_cls

    back = convert(apy_frame, type(cx_frame))
    assert isinstance(back, type(cx_frame))
    # Compare leaves, not the frames: Galactocentric's round trip promotes
    # `roll` from a weak int to a float, so the pytrees are not `==`.
    assert jtu.tree_leaves(back) == pytest.approx(jtu.tree_leaves(cx_frame))


@pytest.mark.parametrize(("cx_frame", "apy_cls"), FRAME_PAIRS)
@pytest.mark.parametrize("target", UNSUPPORTED_TARGETS)
def test_unsupported_target_raises(
    cx_frame: cxastro.AbstractSpaceFrame,
    apy_cls: type[apyc.BaseCoordinateFrame],
    target: type[apyc.BaseCoordinateFrame],
) -> None:
    """Unsupported Astropy targets raise instead of silently mis-converting.

    Registering a conversion on `astropy.coordinates.BaseCoordinateFrame`
    claimed every Astropy frame class, so e.g. ``convert(ICRS(), apyc.Galactic)``
    quietly returned an ICRS frame -- a ~60 degree error.
    """
    if target is apy_cls:  # this one is genuinely supported
        pytest.skip(f"{target.__name__} is the supported target")

    with pytest.raises(TypeError, match="Cannot convert"):
        convert(cx_frame, target)
