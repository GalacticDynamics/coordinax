"""`component_domains` for the intrinsic hyper-spherical charts.

Mirrors the 3D table in `coordinax._src.charts.register_domains`, minus the
radius: these charts are on the unit sphere. The theta-phi swap between the
physics and mathematics conventions is the same trap, so the pairs are kept
adjacent here too.
"""

__all__: tuple[str, ...] = ()

import plum

from .chart import (
    CircularOneSphere,
    LonCosLatSphericalTwoSphere,
    LonLatSphericalTwoSphere,
    MathSphericalTwoSphere,
    SphericalTwoSphere,
)
from coordinax._src.charts.domains import AZIMUTH, FREE, LATITUDE, POLAR, Interval


@plum.dispatch
def component_domains(chart: CircularOneSphere, /) -> dict[str, Interval]:
    return {"phi": AZIMUTH}


@plum.dispatch
def component_domains(chart: SphericalTwoSphere, /) -> dict[str, Interval]:
    """Physics convention: theta is the colatitude, phi the azimuth."""
    return {"theta": POLAR, "phi": AZIMUTH}


@plum.dispatch
def component_domains(chart: MathSphericalTwoSphere, /) -> dict[str, Interval]:
    """Swap theta and phi, as in `MathSpherical3D`."""
    return {"theta": AZIMUTH, "phi": POLAR}


@plum.dispatch
def component_domains(chart: LonLatSphericalTwoSphere, /) -> dict[str, Interval]:
    return {"lon": AZIMUTH, "lat": LATITUDE}


@plum.dispatch
def component_domains(chart: LonCosLatSphericalTwoSphere, /) -> dict[str, Interval]:
    """``lon_coslat`` is unbounded for the reason given on `LonCosLatSpherical3D`."""
    return {"lon_coslat": FREE, "lat": LATITUDE}
