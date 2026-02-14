"""Utilities."""

__all__ = ()

import functools as ft
import inspect

from typing import (  # type: ignore[attr-defined]
    Any,
    TypeVar,
    _GenericAlias,
    get_origin,
)

import equinox as eqx
import hypothesis.strategies as st
import jax
import plum

import unxt as u
import unxt_hypothesis as ust

from .meta import Metadata
from .wrap import AbstractNotIntrospectable
from coordinax_hypothesis._src.namespace import xps

T = TypeVar("T")


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
        dtype=meta.get("dtype", xps.scalar_dtypes()),
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
    # Get the units/dimensions for the quantity
    try:
        dim = u.dimension_of(ann)
    except eqx.EquinoxTracetimeError:
        dim = ust.units()

    # Determine the quantity class and whether to use static values
    # Check if ann is a subclass of StaticQuantity or if it's a parametrized
    # type with StaticQuantity as the underlying type
    quantity_cls = u.Q  # Default
    static_value = False
    if inspect.isclass(ann):
        if issubclass(ann, u.StaticQuantity):
            quantity_cls = u.StaticQuantity
            static_value = True  # StaticQuantity requires StaticValue

    # For generic types like StaticQuantity[PhysicalType('length')]
    elif (
        hasattr(ann, "__origin__")
        and inspect.isclass(ann.__origin__)
        and issubclass(ann.__origin__, u.StaticQuantity)
    ):
        quantity_cls = u.StaticQuantity
        static_value = True  # StaticQuantity requires StaticValue

    # Build quantity strategy
    strategy = ust.quantities(
        unit=dim,
        quantity_cls=quantity_cls,
        dtype=meta.get("dtype", xps.scalar_dtypes()),
        shape=meta.get("shape", xps.array_shapes()),
        static_value=static_value,
    )

    # Apply validators if present
    for validator in meta.get("validators", []):
        strategy = strategy.filter(validator)

    return strategy
