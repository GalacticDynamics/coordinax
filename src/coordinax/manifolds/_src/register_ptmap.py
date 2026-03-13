"""Manifold definitions and manifold inference helpers."""

__all__: tuple[str, ...] = ()


from collections.abc import Callable
from typing import Any, Final

import plum
import wadler_lindig as wl  # type: ignore[import-untyped]

import coordinax.charts as cxc
from .base import AbstractAtlas, AbstractManifold

_ATLAS_MSG: Final[Callable[[AbstractAtlas, cxc.AbstractChart[Any, Any]], str]] = (
    lambda a, c: (
        f"Atlas {a} does not support chart {wl.pformat(c, include_params=False)}"
    )
)


@plum.dispatch
def point_transition_map(
    atlas: AbstractAtlas,
    chart_to: cxc.AbstractChart,  # type: ignore[type-arg]
    chart_from: cxc.AbstractChart,  # type: ignore[type-arg]
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Transition map for points, checking the atlas.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> atlas = cxm.EuclideanAtlas(2)

    >>> x = {"x": 1.0, "y": 1.0}
    >>> cxc.point_transition_map(atlas, cxc.polar2d, cxc.cart2d, x)
    {'r': Array(1.41421356, dtype=float64, ...),
     'theta': Array(0.78539816, dtype=float64, ...)}

    >>> try: cxc.point_transition_map(atlas, cxc.sph2, cxc.cart2d, x)
    ... except ValueError as e: print(e)
    Atlas EuclideanAtlas(ndim=2) does not support chart SphericalTwoSphere()

    """
    if not atlas.supports(chart_to):  # Check compatibility
        raise ValueError(_ATLAS_MSG(atlas, chart_to))
    if not atlas.supports(chart_from):  # Check compatibility
        raise ValueError(_ATLAS_MSG(atlas, chart_from))
    return cxc.point_transition_map(chart_to, chart_from, *args, **kwargs)


@plum.dispatch
def point_transition_map(
    manifold: AbstractManifold,
    chart_to: cxc.AbstractChart,
    chart_from: cxc.AbstractChart,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Transition map for points, checking the manifold's atlas.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> M = cxm.EuclideanManifold(2)

    >>> x = {"x": 1.0, "y": 1.0}
    >>> cxc.point_transition_map(M, cxc.polar2d, cxc.cart2d, x)
    {'r': Array(1.41421356, dtype=float64, ...),
     'theta': Array(0.78539816, dtype=float64, ...)}

    >>> try: cxc.point_transition_map(M, cxc.sph2, cxc.cart2d, x)
    ... except ValueError as e: print(e)
    Atlas EuclideanAtlas(ndim=2) does not support chart SphericalTwoSphere()

    """
    return cxc.point_transition_map(
        manifold.atlas, chart_to, chart_from, *args, **kwargs
    )
