"""Utilities."""

__all__ = ()

import functools as ft
import inspect

from typing import (
    Any,
    Final,
    TypeVar,
    _GenericAlias,  # ty: ignore[unresolved-import]
    get_origin,
)

import equinox as eqx
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import plum
import unxt as u
import unxts.hypothesis as ust
from hypothesis.extra.array_api import make_strategies_namespace

from .dtypes import honoured_dtypes
from .meta import Metadata
from .wrap import AbstractNotIntrospectable

T = TypeVar("T")

xps = make_strategies_namespace(jnp)

#: Default dtypes for array and quantity annotations that pin none themselves.
SCALAR_DTYPES: Final = honoured_dtypes(xps.scalar_dtypes())

# Fallback unit strategy for quantity annotations that pin no dimension.
# ``ust.units()`` re-runs astropy's ``UnitBase.compose()`` on every draw, which is
# uncached and costs ~0.25s for exotic dimensions (e.g. "molar heat capacity") --
# enough to trip ``HealthCheck.too_slow`` on its own. When the annotation tells us
# nothing about the dimension any valid unit will do, so sample the canonical unit
# of each named dimension instead.
ANY_UNITS: Final = st.sampled_from(
    [u.unit(u.dimension(name)._unit) for name in ust.DIMENSION_NAMES]  # ty: ignore[unresolved-attribute]
)


@ft.lru_cache(maxsize=256)
def cached_strategy_for_annotation(
    ann_type: type | AbstractNotIntrospectable[Any],
) -> st.SearchStrategy:
    """Cache strategy_for_annotation calls with empty metadata.

    This is a performance optimization - strategy_for_annotation is expensive
    and is often called with the same annotation types repeatedly.
    """
    return strategy_for_annotation(ann_type, meta=Metadata())


# -----------------------------------------------


@plum.dispatch
def strategy_for_annotation(ann: type, /, *, meta: Metadata) -> st.SearchStrategy:
    """Generate a strategy for a type annotation (base case).

    We ignore the Metadata here.
    """
    return st.from_type(ann)


@plum.dispatch
def strategy_for_annotation(
    ann: _GenericAlias, /, *, meta: Metadata
) -> st.SearchStrategy:
    """Generate a strategy for a type annotation (base case).

    We ignore the Metadata here.
    """
    return strategy_for_annotation(get_origin(ann), meta=meta)


@plum.dispatch
def strategy_for_annotation(
    ann: type[jax.Array], /, *, meta: Metadata
) -> st.SearchStrategy:
    strategy = xps.arrays(
        dtype=meta.get("dtype", SCALAR_DTYPES),
        shape=meta.get("shape", xps.array_shapes()),
    )

    # Apply validators if present
    for validator in meta.get("validators", []):
        strategy = strategy.filter(validator)

    return strategy


@plum.dispatch
def strategy_for_annotation(
    ann: type[u.AbstractQuantity], /, *, meta: Metadata
) -> st.SearchStrategy:
    # Get the units/dimensions for the quantity. Determining the dimension from a
    # bare (non-parametrized) quantity type can raise ``ValueError`` (or, under
    # tracing, ``EquinoxTracetimeError``); fall back to arbitrary units.
    try:
        dim = u.dimension_of(ann)
    except (eqx.EquinoxTracetimeError, ValueError):
        dim = ANY_UNITS

    # Which concrete quantity type(s) to draw, as (class, static_value) pairs.
    # `static_value=True` is not optional -- `StaticQuantity` requires a
    # `StaticValue`. Unwrap parametrized annotations like
    # `StaticQuantity[PhysicalType('length')]` first.
    origin = ann.__origin__ if hasattr(ann, "__origin__") else ann
    if not inspect.isclass(origin):
        kinds = [(u.Q, False)]
    elif issubclass(origin, u.StaticQuantity):
        kinds = [(u.StaticQuantity, True)]  # the annotation pins static
    elif issubclass(u.StaticQuantity, origin):
        # An abstract base (e.g. `AbstractQuantity`) admits either kind, so it
        # must draw both. Defaulting to `u.Q` here silently strips *all* static
        # coverage from every field annotated with it -- and for a chart field
        # that is worse than thin coverage: a dynamic value inside a static
        # chart is a live array that JAX cannot see.
        kinds = [(u.Q, False), (u.StaticQuantity, True)]
    else:
        kinds = [(u.Q, False)]  # the annotation pins dynamic

    # Build quantity strategy
    strategy = st.one_of(
        [
            ust.quantities(
                unit=dim,
                quantity_cls=quantity_cls,
                dtype=meta.get("dtype", SCALAR_DTYPES),
                shape=meta.get("shape", xps.array_shapes()),
                static_value=static_value,
            )
            for quantity_cls, static_value in kinds
        ]
    )

    # Apply validators if present
    for validator in meta.get("validators", []):
        strategy = strategy.filter(validator)

    return strategy
