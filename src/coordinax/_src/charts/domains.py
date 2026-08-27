"""Per-component coordinate domains.

A chart's `coord_dimensions` says what *dimension* each component carries. It
does not say what values are legal: nothing in ``('length', 'angle', 'angle')``
records that a spherical radius is positive or that a colatitude stops at pi.

Nor can the gap be closed by looking at component *names*:

    Spherical3D      ('r', 'theta', 'phi')   ('length', 'angle', 'angle')
    MathSpherical3D  ('r', 'theta', 'phi')   ('length', 'angle', 'angle')

Identical names, identical dimensions, opposite meanings -- `Spherical3D` reads
theta as the colatitude and phi as the azimuth, `MathSpherical3D` the other way
round. `Polar2D` calls its azimuth ``theta`` as well. A lookup keyed on name or
dimension cannot distinguish charts that need opposite bounds, so the domains
are dispatched on the chart *type* -- see `.register_domains`.

This module holds the `Interval` type and the intervals themselves; it imports
no chart, so `.checks` can read its bounds from here without a cycle.

Why this lives in core
----------------------
This completes the split argued in #740: **dimension belongs in the container,
topology belongs in the domain.** #752 made the container side authoritative --
a chart's declared dimensions decide its components' types. The topology side
was declared twice, once by `coordinax._src.charts.checks` on the construction
path and once by ``coordinaxs.hypothesis`` for its strategies, with nothing
linking them; #772 pinned the two equal with a test rather than merging them.
They are now one declaration, and both sides read it.

What core enforces is still a subset of what is declared here. `POLAR` and
`LATITUDE` are checked at construction; `AZIMUTH` and `RADIAL` are not -- an
azimuth outside ``[-pi, pi]`` is a legal coordinate on a different sheet, not
an error. Declaring them anyway is what lets a generator produce points in the
chart's fundamental domain.
"""

__all__ = ("Interval", "component_domains")

import math
from dataclasses import dataclass

from typing import Any, Final

import plum

import unxt as u


@dataclass(frozen=True)
class Interval:
    """Legal values for one coordinate, in a stated canonical unit.

    An unconstrained component is ``Interval()`` -- no unit, no bounds.

    Parameters
    ----------
    unit
        Unit the bounds are written in. Bounds are converted into whichever
        unit a caller actually works in, so that stating ``(0, pi) rad`` still
        constrains a value read in ``cycle``.
    min, max
        Bounds, or `None` for unbounded on that side.
    margin
        How far to stay clear of each finite bound, in ``unit``. A non-zero
        margin is what "open interval" means here -- there is no separate
        exclusive flag, because a bound excluded by zero distance is no use.

        Strict inequality alone is not enough. ``theta = 1e-30 rad`` satisfies
        ``theta > 0`` and is still numerically *at* the pole: the Jacobian
        there is singular to working precision. The margin is the difference
        between mathematically legal and numerically usable.

        It is advisory. Core's construction-time checks admit the closed
        interval -- see `endpoints`, which ignores the margin; the margin is
        for callers *choosing* a value, such as a generator.

    Examples
    --------
    >>> import math
    >>> import unxt as u
    >>> from coordinax.charts import Interval

    >>> Interval("rad", min=0.0, max=math.pi).bounds_in(u.unit("deg"))
    (0.0, 180.0)

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

    def endpoints(self) -> tuple[Any, Any]:
        """Return the bounds as quantities, **without** the margin.

        The construction-path checks in `coordinax._src.charts.checks` compare
        against these, so the closed interval is what a chart admits and the
        margin stays advisory.

        >>> import math
        >>> from coordinax.charts import Interval
        >>> lo, hi = Interval("rad", min=0.0, max=math.pi).endpoints()
        >>> hi
        Angle(3.14159265, 'rad')

        """
        if self.unit is None:
            return None, None
        mk = u.Angle if u.dimension_of(u.unit(self.unit)) == _ANGLE else u.Q
        lo = None if self.min is None else mk(self.min, self.unit)
        hi = None if self.max is None else mk(self.max, self.unit)
        return lo, hi


_ANGLE: Final = u.dimension("angle")

#: Strictly positive radius. Unbounded above; `magnitude` caps it in practice.
RADIAL: Final = Interval("m", min=0.0, margin=1e-3)

#: Polar / colatitude angle, open at both poles where the chart degenerates.
POLAR: Final = Interval("rad", min=0.0, max=math.pi, margin=0.05)

#: Azimuth. Closed: no singularity at either end, they are the same ray.
AZIMUTH: Final = Interval("rad", min=-math.pi, max=math.pi)

#: Latitude, open at the poles for the same reason as `POLAR`.
LATITUDE: Final = Interval("rad", min=-math.pi / 2, max=math.pi / 2, margin=0.05)

#: Explicitly unconstrained.
FREE: Final = Interval()

#: The two intervals core enforces at construction, as quantity pairs. Built
#: once: `check_data` runs on the construction path of every chart, so it must
#: not pay for building a bound per call.
POLAR_ENDPOINTS: Final = POLAR.endpoints()
LATITUDE_ENDPOINTS: Final = LATITUDE.endpoints()


@plum.dispatch.abstract
def component_domains(chart: Any, /) -> dict[str, Interval]:
    """Return the legal interval for each of *chart*'s components.

    Dispatched on the chart type, because names and dimensions do not
    determine the domain (see the module docstring). The implementations live
    in `coordinax._src.charts.register_domains`, and in the corresponding
    module of each package that defines charts of its own.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> from coordinax.charts import component_domains

    `Spherical3D` reads theta as the colatitude:

    >>> component_domains(cxc.sph3d)["theta"].max
    3.14159...

    `MathSpherical3D` reads it as the azimuth, despite the identical name:

    >>> component_domains(cxc.math_sph3d)["theta"].min
    -3.14159...

    A chart says nothing about its components unless it has something to say:

    >>> component_domains(cxc.cart3d)["x"]
    Interval(unit=None, min=None, max=None, margin=0.0)

    """
    raise NotImplementedError  # pragma: no cover
