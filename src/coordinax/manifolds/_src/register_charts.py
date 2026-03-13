"""Point-roled transformations."""

__all__: tuple[str, ...] = ()


from collections.abc import Callable
from typing import Any

import plum

import coordinax.charts as cxc
from .base import AbstractAtlas, AbstractManifold


@plum.dispatch
def point_transition_map(
    atlas: AbstractAtlas,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Check atlas support and perform the point transformation."""
    if not atlas.supports(from_chart):
        msg = f"Atlas does not support source chart: {from_chart}"
        raise ValueError(msg)
    if not atlas.supports(to_chart):
        msg = f"Atlas does not support target chart: {to_chart}"
        raise ValueError(msg)

    return cxc.point_transition_map(to_chart, from_chart, *args, **kwargs)


@plum.dispatch
def point_transition_map(
    manifold: AbstractManifold,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Check manifold's atlas support and perform the point transformation."""
    return cxc.point_transition_map(
        manifold.atlas, to_chart, from_chart, *args, **kwargs
    )
