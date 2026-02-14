"""Jaxtyping Utilities."""

__all__ = ()

import functools as ft
import inspect
from dataclasses import dataclass

import jaxtyping
from typing import Any, Final, final

import hypothesis.strategies as st
import plum

from .meta import Metadata
from .wrap import (
    RECOGNIZE_NONINTROSPECTABLE,
    AbstractNotIntrospectable,
    T,
    wrap_if_not_inspectable,
)
from coordinax_hypothesis._src.namespace import xps

# =====================================================================
# Strategy generation for jaxtyping annotations


@final
@dataclass(frozen=True, slots=True)
class JaxtypingNotIntrospectable(AbstractNotIntrospectable[T]):
    """Wrapper for jaxtyping annotations that are not introspectable."""

    ann: T


# We need to special-case jaxtyping-decorated annotations, since
# <class 'jaxtyping.Shaped'> is uncheckable at runtime.
RECOGNIZE_NONINTROSPECTABLE.append(
    (
        lambda x: inspect.isclass(x) and issubclass(x, jaxtyping.AbstractArray),
        JaxtypingNotIntrospectable,
    )
)


@plum.dispatch
def strategy_for_annotation(
    ann: JaxtypingNotIntrospectable, /, *, meta: Metadata
) -> st.SearchStrategy:  # type: ignore[type-arg]
    """Unwrap and parse."""
    if not (
        inspect.isclass(ann.ann) and issubclass(ann.ann, jaxtyping.AbstractArray)
    ) and not (
        hasattr(ann.ann, "__class__")
        and issubclass(ann.ann.__class__, jaxtyping.AbstractArray)
    ):
        msg = f"Unknown jaxtyping annotation: {ann.ann}"
        raise NotImplementedError(msg)

    typ = ann.ann.array_type
    meta |= parse_jaxtyping_annotation(ann.ann)

    # Re-dispatch with unwrapped type and parsed metadata
    return strategy_for_annotation(wrap_if_not_inspectable(typ), meta=meta)


# =====================================================================


def is_variadic_dim(dim: Any) -> bool:
    """Check if a dimension is variadic (matches 0+ axes)."""
    dim_type = type(dim).__name__
    return dim_type in ("_NamedVariadicDim", "object")  # object is ellipsis


@ft.lru_cache(maxsize=256)
def make_size_strategy(
    dim: Any, /, *, as_list: bool = False
) -> st.SearchStrategy[int | list[int]]:
    """Create a strategy for a single dimension.

    Parameters
    ----------
    dim : Any
        A jaxtyping dimension object (_FixedDim, _NamedDim, etc.)
    as_list : bool
        If True, wrap single values in a list (for variadic flattening)

    Returns
    -------
    st.SearchStrategy[int | list[int]]
        Strategy generating sizes for this dimension

    """
    dim_type = type(dim).__name__

    if is_variadic_dim(dim):
        # Variadic: can match 0 or more axes
        return st.lists(st.integers(min_value=1, max_value=10), min_size=0, max_size=3)

    if dim_type == "_FixedDim":
        # Fixed dimension with optional broadcasting
        size = dim.size
        if dim.broadcastable:
            base_strategy = st.sampled_from([1, size])
        else:
            base_strategy = st.just(size)
        return st.builds(lambda x: [x], base_strategy) if as_list else base_strategy

    if dim_type == "_NamedDim":
        # Named dimension: variable size with optional broadcasting
        size_strategy = st.integers(min_value=1, max_value=10)
        if dim.broadcastable:
            size_strategy = st.one_of(st.just(1), size_strategy)
        return st.builds(lambda x: [x], size_strategy) if as_list else size_strategy

    msg = f"Unknown dimension type: {dim_type}"
    raise NotImplementedError(msg)


def parse_jaxtyping_shape(
    dims: tuple[Any, ...], /
) -> st.SearchStrategy[tuple[int, ...]]:
    """Parse jaxtyping dimension info into a hypothesis shape strategy.

    Parameters
    ----------
    dims : tuple[Any, ...]
        The dims tuple from a jaxtyping annotation (e.g., `Shaped[Array, "3 4"].dims`)

    Returns
    -------
    st.SearchStrategy[tuple[int, ...]]
        A strategy that generates shapes satisfying the dimension constraints.

    Examples
    --------
    >>> from jaxtyping import Shaped
    >>> import jaxtyping
    >>> ann = Shaped[jaxtyping.Array, "3 4"]
    >>> strategy = parse_jaxtyping_shape(ann.dims)

    """
    # Scalar case: empty tuple
    if not dims:
        return st.just(())

    # Check if any dimensions are variadic
    has_variadic = any(is_variadic_dim(dim) for dim in dims)

    if has_variadic:
        # Variadic case: dimensions can expand, so we generate lists and flatten
        strategies = [make_size_strategy(dim, as_list=True) for dim in dims]
        return st.tuples(*strategies).map(
            lambda dims_list: tuple(size for sizes in dims_list for size in sizes)
        )

    # Non-variadic case: simpler direct tuple generation
    strategies = [make_size_strategy(dim, as_list=False) for dim in dims]
    return st.tuples(*strategies)


JAXTYPING_DTYPE_TO_STRATEGY: Final[dict[Any, st.SearchStrategy[Any]]] = {
    jaxtyping.Shaped: xps.scalar_dtypes(),
    jaxtyping.Bool: xps.boolean_dtypes(),
    jaxtyping.Key: xps.unsigned_integer_dtypes(),
    jaxtyping.Num: xps.numeric_dtypes(),
    jaxtyping.Inexact: st.one_of(xps.floating_dtypes(), xps.complex_dtypes()),
    jaxtyping.Float: xps.floating_dtypes(),
    jaxtyping.Complex: xps.complex_dtypes(),
    jaxtyping.Integer: xps.integer_dtypes(),
    jaxtyping.Int: xps.integer_dtypes(),
    jaxtyping.UInt: xps.unsigned_integer_dtypes(),
    jaxtyping.Real: st.one_of(xps.floating_dtypes(), xps.integer_dtypes()),
}


def parse_jaxtyping_dtype(ann: jaxtyping.AbstractArray, /) -> st.SearchStrategy[Any]:
    # Process the dtype annotation. Shaped doesn't list out the full dtype
    # sets, so we need to specify the strategy, otherwise just select from
    # the dtype enumeration.
    return JAXTYPING_DTYPE_TO_STRATEGY.get(ann.dtype, st.sampled_from(ann.dtypes))


def parse_jaxtyping_annotation(ann: jaxtyping.AbstractArray) -> "Metadata":
    """Create Metadata from a jaxtyping annotation.

    Parameters
    ----------
    ann : jaxtyping.AbstractArray
        A jaxtyping annotation like `Shaped[Array, "3 4"]`

    Returns
    -------
    Metadata
        Information needed to generate arrays matching the annotation.

    """
    # Parse the dtype from the annotation into a strategy
    dtype = parse_jaxtyping_dtype(ann)

    # Parse the shape dimensions into a strategy
    shape: st.SearchStrategy[tuple[int, ...]] = parse_jaxtyping_shape(ann.dims)

    return Metadata(dtype=dtype, shape=shape)
