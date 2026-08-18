"""Utility functions for charts."""

__all__: tuple[str, ...] = ()

from typing import cast

import plum

import coordinaxs.api.charts as cxcapi
from coordinax._src.base import AbstractChart, AbstractManifold
from coordinax._src.charts.d0 import Cart0D
from coordinax._src.charts.d1 import Cart1D, Radial1D, Time1D
from coordinax._src.charts.d2 import Cart2D, Polar2D
from coordinax._src.charts.d3 import (
    AbstractSpherical3D,
    Cart3D,
    Cylindrical3D,
    ProlateSpheroidal3D,
)
from coordinax._src.euclidean import R0, R1, R2, R3, EuclideanManifold
from coordinax._src.null import no_manifold
from coordinaxs.api.custom_types import CDict

# ===================================================================
# Guess Manifolds


@plum.dispatch
def guess_manifold(obj: AbstractManifold, /) -> AbstractManifold:
    """Return the manifold of a manifold.

    >>> import coordinax.manifolds as cxm
    >>> M = cxm.Rn(3)
    >>> cxm.guess_manifold(M) is M
    True

    """
    return obj


@plum.dispatch
def guess_manifold(_: type[AbstractChart], /) -> AbstractManifold:
    """Return `no_manifold` for a chart class with no rule of its own.

    Reached only where the class genuinely does not fix a manifold: `CartND`
    carries its dimension per instance, and `PoincarePolar6D` has no manifold
    even as an instance. Every other concrete chart class declares a rule, and
    `TestGuessManifoldOnChartClasses` fails if one stops doing so --
    the fallback is silent by design, so nothing else would notice.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.CartND)
    NoManifold()

    """
    return no_manifold


@plum.dispatch
def guess_manifold(chart: AbstractChart, /) -> AbstractManifold:
    """Infer manifold from a chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.Cart3D)
    Rn(3)

    """
    return chart.M


@plum.dispatch
def guess_manifold(_: type[Cart0D], /) -> EuclideanManifold:
    """Infer manifold from a chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.Cart0D)
    Rn(0)

    """
    return R0


@plum.dispatch
def guess_manifold(_: type[Cart1D | Radial1D | Time1D], /) -> EuclideanManifold:
    """Infer manifold from a chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.Cart1D)
    Rn(1)

    `Time1D` is 1-dimensional too, and its instances already report `Rn(1)`:

    >>> cxm.guess_manifold(cxc.Time1D)
    Rn(1)
    >>> cxc.Time1D().M
    Rn(1)

    """
    return R1


@plum.dispatch
def guess_manifold(_: type[Cart2D | Polar2D], /) -> EuclideanManifold:
    """Infer manifold from a chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.Cart2D)
    Rn(2)

    """
    return R2


@plum.dispatch
def guess_manifold(
    _: type[Cart3D | Cylindrical3D | AbstractSpherical3D | ProlateSpheroidal3D], /
) -> EuclideanManifold:
    """Infer manifold from a chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.Cart3D)
    Rn(3)

    """
    return R3


@plum.dispatch
def guess_manifold(obj: CDict, /) -> AbstractManifold:
    """Infer manifold from a mapping.

    Redispatches based on the inferred chart.

    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold({"x": 1, "y": 2, "z": 3})
    Rn(3)

    """
    chart = cast("AbstractChart", cxcapi.guess_chart(obj))
    return chart.M
