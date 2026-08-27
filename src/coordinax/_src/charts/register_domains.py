"""`component_domains` for the built-in R^n charts.

The intervals themselves live in `.domains`; this is only the chart-type ->
interval table. It is one table rather than a method per chart because the
`Spherical3D` / `MathSpherical3D` theta-phi swap is the whole reason the
lookup is dispatched, and that is legible only when the two sit side by side.

The two-sphere charts are registered next to their definitions, in
`coordinax._src.spherical.register_domains`.
"""

__all__: tuple[str, ...] = ()

import plum

from .d2 import Polar2D
from .d3 import (
    Cylindrical3D,
    LonCosLatSpherical3D,
    LonLatSpherical3D,
    MathSpherical3D,
    Spherical3D,
)
from .domains import AZIMUTH, FREE, LATITUDE, POLAR, RADIAL, Interval
from coordinax._src.base import AbstractChart
from coordinax._src.product.chart import CartesianProductChart


@plum.dispatch
def component_domains(chart: AbstractChart, /) -> dict[str, Interval]:
    """Unconstrained by default: a chart is free unless it says otherwise.

    Cartesian-like charts land here and want exactly this.
    """
    return dict.fromkeys(chart.components, FREE)


# ---------------------------------------------------------------------------
# Radial


@plum.dispatch
def component_domains(chart: Polar2D, /) -> dict[str, Interval]:
    """`Polar2D` names its *azimuth* ``theta``, unlike `Spherical3D`."""
    return {"r": RADIAL, "theta": AZIMUTH}


@plum.dispatch
def component_domains(chart: Cylindrical3D, /) -> dict[str, Interval]:
    return {"rho": RADIAL, "phi": AZIMUTH, "z": FREE}


# ---------------------------------------------------------------------------
# Spherical, in both conventions


@plum.dispatch
def component_domains(chart: Spherical3D, /) -> dict[str, Interval]:
    """Physics convention: theta is the colatitude, phi the azimuth."""
    return {"r": RADIAL, "theta": POLAR, "phi": AZIMUTH}


@plum.dispatch
def component_domains(chart: MathSpherical3D, /) -> dict[str, Interval]:
    """Mathematics convention: theta and phi swap roles."""
    return {"r": RADIAL, "theta": AZIMUTH, "phi": POLAR}


# ---------------------------------------------------------------------------
# Longitude / latitude


@plum.dispatch
def component_domains(chart: LonLatSpherical3D, /) -> dict[str, Interval]:
    return {"lon": AZIMUTH, "lat": LATITUDE, "distance": RADIAL}


@plum.dispatch
def component_domains(chart: LonCosLatSpherical3D, /) -> dict[str, Interval]:
    """``lon_coslat`` is ``lon * cos(lat)``, so the azimuth bounds do not apply.

    Its range depends on the latitude it is paired with, which a per-component
    interval cannot express; `lat` is bounded, and core enforces exactly that.
    """
    return {"lon_coslat": FREE, "lat": LATITUDE, "distance": RADIAL}


# ---------------------------------------------------------------------------
# Composite charts delegate to what they are built from


@plum.dispatch
def component_domains(chart: CartesianProductChart, /) -> dict[str, Interval]:
    """Namespaced union of the factors' domains.

    Keeps a product chart correct for free: constrain the factor once and every
    product containing it follows.
    """
    out: dict[str, Interval] = {}
    for name, factor in zip(chart.factor_names, chart.factors, strict=True):
        for comp, interval in component_domains(factor).items():
            out[f"{name}.{comp}"] = interval
    return out
