r"""Which built-in chart pairs are affine relabellings of one another.

Split from `affinity`, which holds the predicate itself: the declarations
need the concrete chart classes, and the predicate must be importable before
they exist. Each pair is a claim that $\partial^2\psi \equiv 0$, pinned by a
test that measures the curvature rather than trusting the list.
"""

__all__: tuple[str, ...] = ()

import plum

from .d1 import Cart1D, Radial1D
from .d3 import LonLatSpherical3D, MathSpherical3D, Spherical3D
from coordinax._src.base.charts import AbstractChart

# One parameterisation, three spellings. Declared pairwise because no base
# class separates them from `LonCosLat*`, which shares their ancestry and is
# emphatically not affine. The 2-sphere family is declared over in
# `_src/spherical`, where its charts are defined -- which is also how a
# downstream package declares its own.
_SPH3D = (Spherical3D, MathSpherical3D, LonLatSpherical3D)


@plum.dispatch.multi(
    *[(a, b) for a in _SPH3D for b in _SPH3D if a is not b],
    (Cart1D, Radial1D),
    (Radial1D, Cart1D),
)
def is_affine_transition(
    from_chart: AbstractChart,  # type: ignore[type-arg]
    to_chart: AbstractChart,  # type: ignore[type-arg]
    /,
) -> bool:
    r"""Report a relabelling of one parameterisation as affine.

    Constant Jacobian, so no curvature term. In one dimension the radius *is*
    the coordinate, so `cart1d` and `radial1d` are the same statement twice.

    Pinned by a test that measures $\partial^2\psi(v, v)$ over several
    velocity directions rather than trusting this list -- a wrong entry here
    would silently drop that term.
    """
    del from_chart, to_chart
    return True
