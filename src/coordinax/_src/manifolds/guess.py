"""Utility functions for charts."""

__all__: tuple[str, ...] = ()


import plum

import coordinax.api.charts as cxcapi
from coordinax._src.base_charts import AbstractChart
from coordinax._src.base_manifold import AbstractManifold
from coordinax._src.base_topo import AbstractTopologicalManifold
from coordinax._src.charts.d0 import Cart0D
from coordinax._src.charts.d1 import Cart1D, Radial1D
from coordinax._src.charts.d2 import Cart2D, Polar2D
from coordinax._src.charts.d3 import (
    AbstractSpherical3D,
    Cart3D,
    Cylindrical3D,
    ProlateSpheroidal3D,
)
from coordinax._src.custom_types import CDict
from coordinax._src.euclidean import (
    EuclideanManifold,
    euclidean0d,
    euclidean1d,
    euclidean2d,
    euclidean3d,
)
from coordinax._src.no_manifold import no_manifold

# ===================================================================
# Guess Manifolds


@plum.dispatch
def guess_manifold(obj: AbstractTopologicalManifold, /) -> AbstractTopologicalManifold:
    """Return the manifold of a manifold.

    >>> import coordinax.manifolds as cxm
    >>> M = cxm.EuclideanManifold(3)
    >>> cxm.guess_manifold(M) is M
    True

    """
    return obj


@plum.dispatch
def guess_manifold(_: type[AbstractChart], /) -> AbstractTopologicalManifold:
    """Infer manifold from a chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.Cart3D)
    Rn(3)

    """
    return no_manifold


@plum.dispatch
def guess_manifold(chart: AbstractChart, /) -> AbstractTopologicalManifold:
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
    return euclidean0d


@plum.dispatch
def guess_manifold(_: type[Cart1D | Radial1D], /) -> EuclideanManifold:
    """Infer manifold from a chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.Cart1D)
    Rn(1)

    """
    return euclidean1d


@plum.dispatch
def guess_manifold(_: type[Cart2D | Polar2D], /) -> EuclideanManifold:
    """Infer manifold from a chart class.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold(cxc.Cart2D)
    Rn(2)

    """
    return euclidean2d


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
    return euclidean3d


@plum.dispatch
def guess_manifold(obj: CDict, /) -> AbstractManifold:
    """Infer manifold from a mapping.

    Redispatches based on the inferred chart.

    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold({"x": 1, "y": 2, "z": 3})
    Rn(3)

    """
    chart = cxcapi.guess_chart(obj)
    return chart.M
