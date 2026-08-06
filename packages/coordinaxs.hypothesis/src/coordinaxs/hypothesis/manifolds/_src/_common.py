"""Shared helpers for the atlas and manifold strategies.

Lives here rather than in either module because ``manifold`` imports ``atlas``,
so the helper cannot sit in ``manifold`` without a cycle.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from typing import Any, cast

import coordinax.charts as cxc

from coordinaxs.hypothesis.utils import get_all_subclasses


@ft.cache
def matching_chart_classes_for_ndim(
    ndim: int, /
) -> tuple[type[cxc.AbstractChart[Any, Any, Any]], ...]:
    """Return zero-arg chart classes whose default instance has ``ndim``.

    Cached: this instantiates every concrete chart class, and both the
    ``CustomAtlas`` strategy and the ``CustomManifold`` ndim predicate call it
    on every draw.
    """
    classes: list[type[cxc.AbstractChart[Any, Any, Any]]] = []
    for cls in get_all_subclasses(cxc.AbstractChart, exclude_abstract=True):
        cls = cast("type[cxc.AbstractChart[Any, Any, Any]]", cls)
        # One construction serves both questions: a `TypeError` means the class
        # needs arguments, and otherwise the instance carries the ndim to check.
        try:
            chart = cls()
        except TypeError:
            continue
        if chart.ndim == ndim:
            classes.append(cls)
    return tuple(classes)
