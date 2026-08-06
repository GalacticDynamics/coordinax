"""Per-component coordinate domains for chart strategies.

A chart's ``coord_dimensions`` says what *dimension* each component carries. It
does not say what values are legal: nothing in ``('length', 'angle', 'angle')``
records that a spherical radius is positive or that a colatitude stops at pi.
Generating from dimensions alone therefore produces coordinates that are
dimensionally well-formed and geometrically nonsense -- measured on 300 draws
of `Spherical3D`, r <= 0 in 60% and theta outside ``(0, pi)`` in 94%.

Nor can the gap be closed by looking at component *names*:

    Spherical3D      ('r', 'theta', 'phi')   ('length', 'angle', 'angle')
    MathSpherical3D  ('r', 'theta', 'phi')   ('length', 'angle', 'angle')

Identical names, identical dimensions, opposite meanings -- `Spherical3D` reads
theta as the colatitude and phi as the azimuth, `MathSpherical3D` the other way
round. `Polar2D` calls its azimuth ``theta`` as well. A lookup keyed on name or
dimension cannot distinguish charts that need opposite bounds, so the domains
are dispatched on the chart *type*.
"""

__all__ = ("Interval", "component_domains")

import math
from dataclasses import dataclass

from typing import Any, cast

import plum
import unxt as u

import coordinax.charts as cxc


@dataclass(frozen=True)
class Interval:
    """Legal values for one coordinate, in a stated canonical unit.

    An unconstrained component is ``Interval()`` -- no unit, no bounds.

    Parameters
    ----------
    unit
        Unit the bounds are written in. Bounds are converted into whichever
        unit a strategy actually draws, so that stating ``(0, pi) rad`` still
        constrains a draw made in ``cycle``.
    min, max
        Bounds, or `None` for unbounded on that side.
    margin
        How far to stay clear of each finite bound, in ``unit``. A non-zero
        margin is what "open interval" means here -- there is no separate
        exclusive flag, because a bound excluded by zero distance is no use.

        Strict inequality alone is not enough. ``theta = 1e-30 rad`` satisfies
        ``theta > 0`` and is still numerically *at* the pole: the Jacobian
        there is singular to working precision. The margin is the difference
        between mathematically legal and numerically usable, and it is why
        filtering with ``assume`` cannot substitute for this -- the rejection
        region is not measure-zero.

    """

    unit: str | None = None
    min: float | None = None
    max: float | None = None
    margin: float = 0.0

    def bounds_in(self, unit: Any, /) -> tuple[float | None, float | None]:
        """Return ``(lo, hi)`` expressed in *unit*, margins already applied."""
        if self.unit is None:
            return self.min, self.max

        def to_unit(v: float | None, shift: float) -> float | None:
            if v is None:
                return None
            return float(u.ustrip(unit, u.Q(v + shift, self.unit)))

        return to_unit(self.min, self.margin), to_unit(self.max, -self.margin)


#: Strictly positive radius. Unbounded above; `magnitude` caps it in practice.
RADIAL = Interval("m", min=0.0, margin=1e-3)

#: Polar / colatitude angle, open at both poles where the chart degenerates.
POLAR = Interval("rad", min=0.0, max=math.pi, margin=0.05)

#: Azimuth. Closed: no singularity at either end, they are the same ray.
AZIMUTH = Interval("rad", min=-math.pi, max=math.pi)

#: Latitude, open at the poles for the same reason as `POLAR`.
LATITUDE = Interval("rad", min=-math.pi / 2, max=math.pi / 2, margin=0.05)

#: Explicitly unconstrained.
FREE = Interval()


@plum.dispatch.abstract
def component_domains(chart: Any, /) -> dict[str, Interval]:
    """Return the legal interval for each of *chart*'s components.

    Dispatched on the chart type, because names and dimensions do not
    determine the domain (see the module docstring).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> from coordinaxs.hypothesis.charts import component_domains

    `Spherical3D` reads theta as the colatitude:

    >>> component_domains(cxc.sph3d)["theta"].max
    3.14159...

    `MathSpherical3D` reads it as the azimuth, despite the identical name:

    >>> component_domains(cxc.math_sph3d)["theta"].min
    -3.14159...

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
def component_domains(chart: cxc.AbstractChart, /) -> dict[str, Interval]:
    """Unconstrained by default: a chart is free unless it says otherwise.

    Cartesian-like charts land here and want exactly this.
    """
    return dict.fromkeys(chart.components, FREE)


# ---------------------------------------------------------------------------
# Radial


# `Radial1D` deliberately has no overload. Its docstring calls it "semantically
# equivalent to Cart1D but uses `r` instead of `x`", and `pt_map` bears that
# out: cart1d -> radial1d carries the sign through, so x = -5 m gives r = -5 m
# and round-trips. Constraining it to r > 0 on the strength of the *name* would
# silently halve the chart's domain -- the same trap as reading `theta` off a
# name, in the other direction. It falls through to the unconstrained default.


@plum.dispatch
def component_domains(chart: cxc.Polar2D, /) -> dict[str, Interval]:
    """`Polar2D` names its *azimuth* ``theta``, unlike `Spherical3D`."""
    return {"r": RADIAL, "theta": AZIMUTH}


@plum.dispatch
def component_domains(chart: cxc.Cylindrical3D, /) -> dict[str, Interval]:
    return {"rho": RADIAL, "phi": AZIMUTH, "z": FREE}


# ---------------------------------------------------------------------------
# Spherical, in both conventions


@plum.dispatch
def component_domains(chart: cxc.Spherical3D, /) -> dict[str, Interval]:
    """Physics convention: theta is the colatitude, phi the azimuth."""
    return {"r": RADIAL, "theta": POLAR, "phi": AZIMUTH}


@plum.dispatch
def component_domains(chart: cxc.MathSpherical3D, /) -> dict[str, Interval]:
    """Mathematics convention: theta and phi swap roles."""
    return {"r": RADIAL, "theta": AZIMUTH, "phi": POLAR}


@plum.dispatch
def component_domains(chart: cxc.SphericalTwoSphere, /) -> dict[str, Interval]:
    return {"theta": POLAR, "phi": AZIMUTH}


@plum.dispatch
def component_domains(chart: cxc.MathSphericalTwoSphere, /) -> dict[str, Interval]:
    """Swap theta and phi, as in `MathSpherical3D`."""
    return {"theta": AZIMUTH, "phi": POLAR}


# ---------------------------------------------------------------------------
# Longitude / latitude


@plum.dispatch
def component_domains(chart: cxc.LonLatSpherical3D, /) -> dict[str, Interval]:
    return {"lon": AZIMUTH, "lat": LATITUDE, "distance": RADIAL}


@plum.dispatch
def component_domains(chart: cxc.LonLatSphericalTwoSphere, /) -> dict[str, Interval]:
    return {"lon": AZIMUTH, "lat": LATITUDE}


@plum.dispatch
def component_domains(chart: cxc.CircularOneSphere, /) -> dict[str, Interval]:
    return {"phi": AZIMUTH}


# ---------------------------------------------------------------------------
# Composite charts delegate to what they are built from


@plum.dispatch
def component_domains(chart: cxc.CartesianProductChart, /) -> dict[str, Interval]:
    """Namespaced union of the factors' domains.

    Keeps a product chart correct for free: constrain the factor once and every
    product containing it follows.
    """
    out: dict[str, Interval] = {}
    for name, factor in zip(chart.factor_names, chart.factors, strict=True):
        factor_domains = cast("dict[str, Interval]", component_domains(factor))
        for comp, interval in factor_domains.items():
            out[f"{name}.{comp}"] = interval
    return out
