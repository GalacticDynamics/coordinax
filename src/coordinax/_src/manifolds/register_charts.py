"""Manifold definitions and manifold inference helpers."""

__all__: tuple[str, ...] = ()


from collections.abc import Callable
from typing import Any, Final

import plum
import wadler_lindig as wl

import coordinax.charts as cxc
from .base import AbstractAtlas, AbstractManifold

#####################################################################
# Point Transition Map


_ATLAS_MSG: Final[Callable[[AbstractAtlas, cxc.AbstractChart[Any, Any]], str]] = (
    lambda a, c: (
        f"Atlas {a} does not support chart {wl.pformat(c, include_params=False)}"
    )
)


@plum.dispatch
def pt_map(
    x: Any,
    atlas: AbstractAtlas,
    chart_from: cxc.AbstractChart,
    chart_to: cxc.AbstractChart,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Transition map for points, checking the atlas.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> atlas = cxm.EuclideanAtlas(2)

    >>> x = {"x": 1.0, "y": 1.0}
    >>> cxc.pt_map(x, atlas, cxc.cart2d, cxc.polar2d)
    {'r': Array(1.41421356, dtype=float64, ...),
     'theta': Array(0.78539816, dtype=float64, ...)}

    >>> try: cxc.pt_map(x, atlas, cxc.cart2d, cxc.sph2)
    ... except ValueError as e: print(e)
    Atlas EuclideanAtlas(ndim=2) does not support chart SphericalTwoSphere()

    """
    # Check the atlas supports the charts
    if not atlas.has_chart(chart_to):  # Check compatibility
        raise ValueError(_ATLAS_MSG(atlas, chart_to))
    if not atlas.has_chart(chart_from):  # Check compatibility
        raise ValueError(_ATLAS_MSG(atlas, chart_from))

    # If charts are supported, delegate to ptm
    return cxc.pt_map(x, chart_from, chart_to, *args, **kwargs)


# default route
@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def pt_map(
    x: Any,
    manifold: AbstractManifold,
    chart_from: cxc.AbstractChart,
    chart_to: cxc.AbstractChart,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Transition map for points, checking the manifold's atlas.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> M = cxm.EuclideanManifold(2)

    >>> x = {"x": 1.0, "y": 1.0}
    >>> cxc.pt_map(x, M, cxc.cart2d, cxc.polar2d)
    {'r': Array(1.41421356, dtype=float64, ...),
     'theta': Array(0.78539816, dtype=float64, ...)}

    >>> try: cxc.pt_map(x, M, cxc.cart2d, cxc.sph2)
    ... except ValueError as e: print(e)
    Atlas EuclideanAtlas(ndim=2) does not support chart SphericalTwoSphere()

    """
    # Redispatch to the atlas
    return cxc.pt_map(x, manifold.atlas, chart_from, chart_to, *args, **kwargs)


# ===================================================================
# Special cases
