"""`component_domains` for the built-in R^n charts.

The intervals themselves live in `.domains`; this is only the chart-type ->
interval table. It is one table rather than a method per chart because the
`Spherical3D` / `MathSpherical3D` theta-phi swap is the whole reason the
lookup is dispatched, and that is legible only when the two sit side by side.

The two-sphere charts are registered next to their definitions, in
`coordinax._src.spherical.register_domains`: that package imports this one, so
their registrations cannot live here without inverting the dependency.
"""

__all__: tuple[str, ...] = ()

import math

import numpy as np
import plum

import unxt as u

from .d2 import Polar2D
from .d3 import (
    Cylindrical3D,
    LonCosLatSpherical3D,
    LonLatSpherical3D,
    MathSpherical3D,
    ProlateSpheroidal3D,
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
# Parameterized


@plum.dispatch
def component_domains(chart: ProlateSpheroidal3D, /) -> dict[str, Interval]:
    """Bounds set by the focal length: ``mu >= Delta^2``, ``|nu| <= Delta^2``.

    The only chart whose domain depends on its own parameters rather than on
    its type alone. Dispatch passes the instance, so reading `Delta` here is
    what makes that expressible -- and it is the same `Delta` that
    `ProlateSpheroidal3D.check_data` compares against, so there is no second
    copy of the bound to drift.

    Both bounds are closed, matching what `check_data` enforces; no margin,
    because none has been measured for the focal ring at ``mu == Delta^2``.

    Unlike every other chart's domain, this one needs a concrete `Delta` and
    so cannot be read under `jax.jit`. Nothing on the construction path calls
    it -- `check_data` compares quantities directly, and stays traceable.
    """
    # `Delta` is a length and positive, so `Delta**2` is always an area -- but
    # not always a finite one: a `Delta` near the float ceiling squares to
    # infinity, which is not a bound. Such a chart declares nothing for `mu`
    # and `nu`, the same answer as any other chart with no bound to state.
    with np.errstate(over="ignore"):
        delta_sq = chart.Delta**2
    unit = str(delta_sq.unit)
    bound = float(u.ustrip(delta_sq.unit, delta_sq))
    if not math.isfinite(bound):
        return {"mu": FREE, "nu": FREE, "phi": AZIMUTH}

    return {
        "mu": Interval(unit, min=bound),
        "nu": Interval(unit, min=-bound, max=bound),
        "phi": AZIMUTH,
    }


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
