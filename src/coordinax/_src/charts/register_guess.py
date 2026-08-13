"""Utility functions for charts."""

__all__: tuple[str, ...] = ()


from jaxtyping import ArrayLike, Shaped
from typing import Any, cast

import plum

import unxt as u

import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from .d1 import cart1d
from .d2 import cart2d
from .d3 import Spherical3D, cart3d
from .dn import cartnd
from coordinax._src.base import (
    NON_ABC_CHART_CLASSES,
    AbstractChart,
    AbstractFixedComponentsChart,
)
from coordinaxs.api.custom_types import CDict

# ===================================================================
# Guess Chart Classes

ChartCls = type[AbstractFixedComponentsChart[Any, Any, Any]]

CANONICAL_CHART_CLASSES: dict[frozenset[str], ChartCls] = {}
"""The chart to infer when a component-name set matches several charts.

Component names do not always identify a chart: `Spherical3D` and
`MathSpherical3D` both carry ``("r", "theta", "phi")`` but disagree about which
of the two angles is polar. Which one a bare name set means is a choice, so it
is declared here rather than left to the order charts happen to be scanned in.

Populated by `register_canonical_chart`, called from the module that defines
the chart -- the same late-registration pattern as the ``register_*`` modules,
which is what lets a chart from any package declare itself without
`coordinax._src.charts` having to import that package.
"""


def register_canonical_chart(cls: ChartCls, /) -> ChartCls:
    """Declare `cls` as the chart to infer for its component names.

    Redeclaring the same class is a no-op, so a module may be imported twice.
    Declaring a *different* class for names already claimed raises: silently
    overwriting would put the inferred chart back at the mercy of import order,
    which is what this registry exists to take it out of.

    Returns `cls`, so it can also be used as a decorator.
    """
    keys = frozenset(cls._components)
    claimed = CANONICAL_CHART_CLASSES.get(keys)
    if claimed is not None and claimed is not cls:
        msg = (
            f"Components {sorted(keys)} are already declared canonical for "
            f"{claimed.__name__}; {cls.__name__} cannot also claim them."
        )
        raise ValueError(msg)
    CANONICAL_CHART_CLASSES[keys] = cls
    return cls


register_canonical_chart(Spherical3D)  # over `MathSpherical3D`


# TODO: speed this up. The problem is that caching the results breaks something,
# causing functions in other modules to fail type(x) is type(y) checks.
def guess_chart_cls(obj: frozenset[str]) -> type[AbstractChart[Any, Any, Any]]:
    """Infer a chart class from the keys of a component dictionary.

    This only works on charts with fixed components.

    Every match is collected rather than the first one returned, because
    `NON_ABC_CHART_CLASSES` is a `weakref.WeakSet`: classes hash by `id`, so it
    iterates in a different order in every process. Where several charts share
    a name set, `CANONICAL_CHART_CLASSES` decides between them.

    """
    matches = [
        chart_cls
        for chart_cls in NON_ABC_CHART_CLASSES
        if issubclass(chart_cls, AbstractFixedComponentsChart)
        and frozenset(chart_cls._components) == obj
    ]

    # `obj` is a frozenset, whose iteration order varies between processes;
    # sort it so the message reads the same everywhere.
    keys = sorted(obj)

    if not matches:
        msg = f"Cannot infer representation from keys {keys}"
        raise ValueError(msg)

    if len(matches) == 1:
        return matches[0]

    canonical = CANONICAL_CHART_CLASSES.get(obj)
    if canonical is None:
        # Returning any one of these would be a coin flip between conventions.
        names = ", ".join(sorted(cls.__name__ for cls in matches))
        msg = (
            f"Keys {keys} match several charts ({names}) and none of them is "
            "canonical; declare the intended one with "
            "`register_canonical_chart`, or pass the chart explicitly."
        )
        raise ValueError(msg)
    return canonical


# ===================================================================
# Guess Charts


@plum.dispatch
def guess_chart(obj: frozenset[str], /) -> AbstractChart:
    """Infer a chart from the keys of a component dictionary.

    Note that many charts may share the same component names (e.g., Spherical3D
    and MathSpherical3D both use 'r', 'theta', 'phi'). These are completely
    indistinguishable from component names alone, so `CANONICAL_CHART_CLASSES`
    names the one to infer -- the physics convention in both current cases.
    Pass the chart explicitly when the convention matters.

    >>> import coordinax.charts as cxc
    >>> d = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> chart = cxc.guess_chart(d)
    >>> chart
    Cart3D(M=Rn(3))

    """
    # Infer the chart from the keys
    chart_cls = guess_chart_cls(obj)
    # Guess the manifold from the chart class, if possible. This is needed to
    # instantiate the chart class, which requires a manifold argument.
    M = cxmapi.guess_manifold(chart_cls)

    # Instantiate the chart class.
    return chart_cls(M=M)  # ty: ignore[unknown-argument]


@plum.dispatch
def guess_chart(obj: CDict, /) -> AbstractChart:
    """Infer a chart from the keys of a component dictionary.

    Note that many charts may share the same component names (e.g., Spherical3D
    and MathSpherical3D both use 'r', 'theta', 'phi'). These are completely
    indistinguishable from component names alone, so `CANONICAL_CHART_CLASSES`
    names the one to infer -- the physics convention in both current cases.
    Pass the chart explicitly when the convention matters.

    >>> import coordinax.charts as cxc
    >>> d = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> chart = cxc.guess_chart(d)
    >>> chart
    Cart3D(M=Rn(3))

    """
    out = cxcapi.guess_chart(frozenset(obj.keys()))
    return cast("AbstractChart", out)


@plum.dispatch
def guess_chart(
    _: Shaped[ArrayLike, "*batch 1"] | Shaped[u.AbstractQuantity, "*batch 1"], /
) -> AbstractChart:
    """Infer a 1D Cartesian chart from last dimension of a value / quantity.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> q = u.Q([1.0], "m")
    >>> cxc.guess_chart(q)
    Cart1D(M=Rn(1))

    """
    return cart1d  # ty: ignore[invalid-return-type]


@plum.dispatch
def guess_chart(
    _: Shaped[ArrayLike, "*batch 2"] | Shaped[u.AbstractQuantity, "*batch 2"], /
) -> AbstractChart:
    """Infer a 2D Cartesian chart from last dimension of a value / quantity.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> q = u.Q([1.0, 2.0], "m")
    >>> cxc.guess_chart(q)
    Cart2D(M=Rn(2))

    """
    return cart2d  # ty: ignore[invalid-return-type]


@plum.dispatch
def guess_chart(
    _: Shaped[ArrayLike, "*batch 3"] | Shaped[u.AbstractQuantity, "*batch 3"], /
) -> AbstractChart:
    """Infer a 3D Cartesian chart from last dimension of a value / quantity.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> cxc.guess_chart(q)
    Cart3D(M=Rn(3))

    """
    return cart3d  # ty: ignore[invalid-return-type]


@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def guess_chart(
    _: Shaped[ArrayLike, "*batch N"] | Shaped[u.AbstractQuantity, "*batch N"], /
) -> AbstractChart:
    """Infer a N-dimensional Cartesian chart from last dimension of a value / quantity.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> q = u.Q([1.0, 2.0, 3.0, 4.0], "m")
    >>> cxc.guess_chart(q)
    CartND(M=Rn(True))

    """
    return cartnd  # ty: ignore[invalid-return-type]
