"""Manifolds in coordinax."""

__all__: tuple[str, ...] = ()


from typing import cast

import plum

import coordinax.api.manifolds as cxmapi
import coordinax.charts as cxc
from .base import AbstractManifold
from coordinax.internal.custom_types import CDict


@plum.dispatch
def guess_manifold(obj: AbstractManifold, /) -> AbstractManifold:
    """Return the manifold of a manifold.

    >>> import coordinax.manifolds as cxm
    >>> M = cxm.EuclideanManifold(3)
    >>> cxm.guess_manifold(M) is M
    True

    """
    return obj


@plum.dispatch
def guess_manifold(obj: CDict, /) -> AbstractManifold:
    """Infer manifold from a mapping.

    Redispatches based on the inferred chart.

    >>> import coordinax.manifolds as cxm
    >>> cxm.guess_manifold({"x": 1, "y": 2, "z": 3})
    EuclideanManifold(ndim=3)

    """
    chart = cxc.guess_chart(obj)
    out = cxmapi.guess_manifold(chart)
    return cast("AbstractManifold", out)
