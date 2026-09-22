"""Hypothesis strategies for astronomical reference frames."""

__all__ = ("galactocentric_frames",)


from typing import cast

from hypothesis import strategies as st

import quaxed.numpy as jnp
import unxt as u
import unxts.hypothesis as ust

import coordinax.charts as cxc
import coordinax.vectors as cxv
import coordinaxs.astro as cxastro

#: Galactic-centre distance, in kpc. Bounded well away from zero so the
#: ``asin(z_sun / galcen_distance)`` tilt is always a real angle and the frame
#: never degenerates onto its own origin.
DISTANCE_KPC = ust.quantities(
    "kpc", dtype=jnp.float64, elements={"min_value": 1.0, "max_value": 20.0}
)

#: Sun's height above the midplane, in pc. |z_sun| stays below the 1 kpc floor
#: on ``DISTANCE_KPC``, which is what keeps the tilt real.
Z_SUN_PC = ust.quantities(
    "pc", dtype=jnp.float64, elements={"min_value": -500.0, "max_value": 500.0}
)

#: Solar velocity components, in km/s.
V_SUN_KMS = ust.quantities(
    "km/s",
    dtype=jnp.float64,
    shape=(3,),
    elements={"min_value": -500.0, "max_value": 500.0},
)


def _angles(lo: float, hi: float) -> st.SearchStrategy[u.Angle]:
    """Scalar `unxt.Angle` in degrees, bounded to ``[lo, hi]``."""
    out = ust.quantities(
        "deg",
        quantity_cls=u.Angle,
        dtype=jnp.float64,
        elements={"min_value": lo, "max_value": hi},
    )
    return cast("st.SearchStrategy[u.Angle]", out)


@st.composite
def galactocentric_frames(draw: st.DrawFn, /) -> cxastro.Galactocentric:
    """Strategy for generating Galactocentric instances.

    All five frame parameters are drawn: the longitude, latitude and distance
    of the Galactic centre, the roll angle, the Sun's height above the
    midplane, and the solar velocity. The ranges are physically sensible --
    a Galactic-centre distance of 1-20 kpc, |z_sun| < 500 pc (so the
    ``asin(z_sun / galcen_distance)`` tilt is always real), latitudes off the
    poles, and solar velocities under 500 km/s -- because the point is to
    exercise non-default parameters, not to generate degenerate frames.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.

    Returns
    -------
    coordinaxs.astro.Galactocentric
        A strategy that generates Galactocentric instances.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinaxs.hypothesis.astro as cxastrost
    >>> import coordinaxs.astro as cxastro

    >>> @given(frame=cxastrost.galactocentric_frames())
    ... def test_galactocentric(frame):
    ...     assert isinstance(frame, cxastro.Galactocentric)
    ...     assert frame.galcen["distance"].ustrip("kpc") >= 1.0

    """
    galcen = cxv.Point.from_(
        {
            "lon": draw(_angles(0.0, 360.0)),
            "lat": draw(_angles(-89.0, 89.0)),
            "distance": draw(DISTANCE_KPC),
        },
        cxc.lonlat_sph3d,
    )
    return cxastro.Galactocentric(
        galcen=galcen,
        roll=draw(_angles(-180.0, 180.0)),
        z_sun=draw(Z_SUN_PC),
        galcen_v_sun=cxv.Tangent.from_(draw(V_SUN_KMS)),
    )


# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxastro.Galactocentric, lambda _: galactocentric_frames())
