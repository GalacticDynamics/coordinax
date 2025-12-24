"""Utilities."""
from is_annotated import isannotated

__all__ = ("get_all_subclasses", "draw_if_strategy", "build_init_kwargs_strategy")

import functools as ft
import inspect
import warnings
from dataclasses import dataclass

import jaxtyping
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any, Generic, TypedDict, TypeVar

import beartype
import equinox as eqx
import hypothesis.strategies as st
import jax.numpy as jaxnp
import plum
import unxt_hypothesis as ust
from hypothesis.extra.array_api import make_strategies_namespace

import unxt as u

import coordinax as cx

T = TypeVar("T")

xps: SimpleNamespace = make_strategies_namespace(jaxnp)

BeartypeValidator = beartype.vale.Is[lambda x: x].__class__

@ft.cache
def get_all_subclasses(
    base_class: type,
    /,
    *,
    filter: type | tuple[type, ...] = object,
    exclude_abstract: bool = True,
    exclude: tuple[type, ...] = (),
) -> tuple[type, ...]:
    """Build a set of all subclasses of a given base class.

    Parameters
    ----------
    base_class : type
        The base class to find subclasses of.
    filter : tuple[type, ...] | None, optional
        A tuple of classes to limit the subclasses to, by default `None`.
    exclude_abstract : bool, optional
        Whether to exclude abstract subclasses, by default True.
    exclude : tuple[type, ...], optional
        Specific classes (covariant) to exclude, by default ().

    Returns
    -------
    tuple[type, ...]
        A tuple of all subclasses of the base class.

    """
    subclasses = []

    # Normalize filter to a tuple
    filter_tuple = filter if isinstance(filter, tuple) else (filter,)

    def recurse(cls: type, /) -> None:
        for subclass in cls.__subclasses__():
            # Skip if in exclude list
            if any(issubclass(subclass, ex) for ex in exclude):
                continue
            # Check if subclass matches ALL filters (not just ANY)
            if all(issubclass(subclass, f) for f in filter_tuple) and not (
                exclude_abstract
                and (
                    inspect.isabstract(subclass)
                    or subclass.__name__.startswith("Abstract")
                )
            ):
                # Harden against multiple coordinax copies, which can happy in
                # editable installs for uv-workspaces.
                if hasattr(cx.r, subclass.__name__):
                    subclass = getattr(cx.r, subclass.__name__)  # noqa: PLW2901

                subclasses.append(subclass)
            # Always recurse to find deeper subclasses
            recurse(subclass)

    recurse(base_class)

    if not subclasses:
        warnings.warn(
            f"No subclasses found for base class {base_class} "
            f"with filter={filter} "
            f"and exclude_abstract={exclude_abstract}.",
            category=UserWarning,
            stacklevel=2,
        )

    return tuple(subclasses)


def draw_if_strategy(draw: st.DrawFn, v: T | st.SearchStrategy[T], /) -> T:
    """Draw a value if a strategy is given, else return the value."""
    return draw(v) if isinstance(v, st.SearchStrategy) else v


# =============================================================================
# Argument introspection and strategy generation


def _param_filter(name: str, param: inspect.Parameter, /) -> bool:
    """Filter function to identify parameters."""
    return (
        name != "self"
        # and param.default is inspect.Parameter.empty
        and param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )


def get_init_params(cls: type, /) -> Mapping[str, inspect.Parameter]:
    """Get ``__init__`` parameters for a class (excluding 'self').

    Parameters
    ----------
    cls : type
        The class to inspect.

    Returns
    -------
    Mapping[str, inspect.Parameter]
        Dictionary mapping parameter names to their Parameter objects.
        Only includes parameters without defaults.

    """
    try:
        sig = inspect.signature(cls.__init__)  # type: ignore[misc]
    except (ValueError, TypeError):
        return {}

    return {n: p for n, p in sig.parameters.items() if _param_filter(n, p)}


# -----------------------------------------------
# Jaxtyping parsing


def _is_variadic_dim(dim: Any) -> bool:
    """Check if a dimension is variadic (matches 0+ axes)."""
    dim_type = type(dim).__name__
    return dim_type in ("_NamedVariadicDim", "object")  # object is ellipsis


def _make_size_strategy(
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

    if _is_variadic_dim(dim):
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
    >>> shape = strategy.example()  # doctest: +SKIP
    >>> shape  # doctest: +SKIP
    (3, 4)

    """
    # Scalar case: empty tuple
    if not dims:
        return st.just(())

    # Check if any dimensions are variadic
    has_variadic = any(_is_variadic_dim(dim) for dim in dims)

    if has_variadic:
        # Variadic case: dimensions can expand, so we generate lists and flatten
        strategies = [_make_size_strategy(dim, as_list=True) for dim in dims]
        return st.tuples(*strategies).map(
            lambda dims_list: tuple(size for sizes in dims_list for size in sizes)
        )

    # Non-variadic case: simpler direct tuple generation
    strategies = [_make_size_strategy(dim, as_list=False) for dim in dims]
    return st.tuples(*strategies)


def parse_jaxtyping_dtype(ann: jaxtyping.AbstractArray) -> st.SearchStrategy[Any]:
    # Process the dtype annotation. Shaped doesn't list out the full dtype
    # sets, so we need to specify the strategy, otherwise just select from
    # the dtype enumeration.
    dtype: st.SearchStrategy
    match ann.dtype:
        case jaxtyping.Shaped:
            dtype = xps.scalar_dtypes()
        case jaxtyping.Bool:
            dtype = xps.boolean_dtypes()
        case jaxtyping.Key:  # TODO: confirm
            dtype = xps.unsigned_integer_dtypes()
        case jaxtyping.Num:
            dtype = xps.numeric_dtypes()
        case jaxtyping.Inexact:
            dtype = xps.inexact_dtypes()
        case jaxtyping.Float:
            dtype = xps.floating_dtypes()
        case jaxtyping.Complex:
            dtype = xps.complex_dtypes()
        case jaxtyping.Integer | jaxtyping.Int:
            dtype = xps.integer_dtypes()
        case jaxtyping.UInt:
            dtype = xps.unsigned_integer_dtypes()
        case jaxtyping.Real:
            # Real should only include float and integer dtypes, not bool
            dtype = st.one_of(xps.floating_dtypes(), xps.integer_dtypes())
        case _:
            dtype = st.sampled_from(ann.dtypes)

    return dtype


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


# -----------------------------------------------


class Metadata(TypedDict, total=False):  # closed=False
    """Holds shape and dtype information for strategy generation."""

    dtype: st.SearchStrategy[Any]
    shape: st.SearchStrategy[tuple[int, ...]]
    validators: list[Callable[[Any], bool]]  # Beartype validator functions


@dataclass(frozen=True, slots=True)
class Wrapper(Generic[T]):
    """Wrapper to indicate a jaxtyping annotation."""

    ann: T


def wrap_if_not_inspectable(ann: T, /) -> T | Wrapper[T]:
    """Wrap a jaxtyping annotation in a wrapper class.

    We need to special-case jaxtyping-decorated annotations, since
    <class 'jaxtyping.Shaped'> is uncheckable at runtime.
    """
    if (
        inspect.isclass(ann) and issubclass(ann, jaxtyping.AbstractArray)
    ) or isannotated(ann):
        return Wrapper(ann)

    return ann


@plum.dispatch
def strategy_for_annotation(ann: type, /, *, meta: Metadata) -> st.SearchStrategy:
    """Generate a strategy for a type annotation (base case).

    We ignore the Metadata here.
    """
    return st.from_type(ann)


@plum.dispatch
def strategy_for_annotation(ann: Wrapper, /, *, meta: Metadata) -> st.SearchStrategy:
    """Unwrap and parse."""
    if isannotated(ann.ann):
        # Unpack Annotated type
        typ = ann.ann.__origin__
        # Extract metadata from annotations
        for md in ann.ann.__metadata__:
            if isinstance(md, dict):
                meta |= md
            elif isinstance(md, BeartypeValidator):
                # Initialize validators list if not present
                if "validators" not in meta:
                    meta["validators"] = []
                meta["validators"].append(md.is_valid)

    elif issubclass(ann.ann, jaxtyping.AbstractArray):
        typ = ann.ann.array_type
        meta |= parse_jaxtyping_annotation(ann.ann)
    else:
        msg = f"Unknown annotation: {ann.ann}"
        raise NotImplementedError(msg)

    # Re-dispatch with unwrapped type and parsed metadata
    return strategy_for_annotation(wrap_if_not_inspectable(typ), meta=meta)


@plum.dispatch
def strategy_for_annotation(
    ann: type[jaxtyping.Array], /, *, meta: Metadata
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
    # Build quantity strategy
    strategy = ust.quantities(
        unit=dim,
        dtype=meta.get("dtype", xps.scalar_dtypes()),
        shape=meta.get("shape", xps.array_shapes()),
        static_value=issubclass(ann, u.StaticQuantity),
    )

    # Apply validators if present
    for validator in meta.get("validators", []):
        strategy = strategy.filter(validator)

    return strategy


@plum.dispatch
def build_init_kwargs_strategy(cls: type, *, dim: int | None) -> st.SearchStrategy:
    """Build a strategy that generates valid __init__ kwargs for a class.

    Parameters
    ----------
    cls : type
        The class to generate kwargs for.

    Returns
    -------
    st.SearchStrategy[dict[str, Any]]
        A strategy that generates dictionaries of keyword arguments
        suitable for passing to cls.__init__().

    """
    required_params = get_init_params(cls)

    if not required_params:
        # No required parameters
        return st.just({})

    # Build a strategy for each parameter
    strategies = {}
    for k, param in required_params.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            msg = f"Parameter '{k}' of {cls} has no type annotation"
            raise ValueError(msg)

        # Generate strategy for this parameter's annotation
        strategies[k] = strategy_for_annotation(
            wrap_if_not_inspectable(ann), meta=Metadata()
        )

    # Combine all parameter strategies into a single kwargs dict strategy
    return st.fixed_dictionaries(strategies)
