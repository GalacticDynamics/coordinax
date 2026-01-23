"""Utilities."""

__all__ = ("get_all_subclasses", "draw_if_strategy")

import dataclasses
import functools as ft
import inspect
import sys
import warnings
from dataclasses import dataclass

import jaxtyping
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import (
    Any,
    Final,
    Generic,
    TypedDict,
    TypeVar,
    _GenericAlias,
    final,
    get_origin,
)

import beartype
import equinox as eqx
import hypothesis.strategies as st
import jax.numpy as jaxnp
import plum
from hypothesis.extra.array_api import make_strategies_namespace
from is_annotated import isannotated

import unxt as u
import unxt_hypothesis as ust

import coordinax.charts as cxc

T = TypeVar("T")

xps: SimpleNamespace = make_strategies_namespace(jaxnp)

BeartypeValidator = beartype.vale.Is[lambda x: x].__class__


def _canonicalize_coordinax_class(cls: type) -> type:
    """Resolve a coordinax class to its canonical version.

    In editable installs with uv-workspaces, the same class can exist as
    multiple Python objects due to import path duplication. This function
    returns the canonical version by looking it up via its __qualname__ in
    the coordinax.charts module.

    This is particularly needed for slotted dataclasses (equinox modules)
    which seem more prone to this duplication.

    Parameters
    ----------
    cls : type
        The class to canonicalize.

    Returns
    -------
    type
        The canonical version of the class from coordinax.charts, or the
        original class if it can't be resolved.

    """
    module = getattr(cls, "__module__", "")

    # Only process coordinax classes
    if not module.startswith("coordinax"):
        return cls

    # Check if it's a slotted dataclass (equinox Module or has __slots__)
    # These are the ones that tend to get duplicated
    is_slotted = (
        hasattr(cls, "__slots__")
        or (dataclasses.is_dataclass(cls) and getattr(cls, "__dataclass_fields__", {}))
        or issubclass(cls, eqx.Module)
    )

    if not is_slotted:
        return cls

    # Try to resolve via coordinax.charts using __qualname__
    qualname = cls.__qualname__

    # Handle nested classes (e.g., "Outer.Inner")
    parts = qualname.split(".")

    # Try to find the class in cxc (coordinax.charts)
    try:
        resolved = cxc
        for part in parts:
            resolved = getattr(resolved, part)
        if isinstance(resolved, type) and issubclass(resolved, cls.__bases__[0] if cls.__bases__ else object):
            return resolved
    except AttributeError:
        pass

    # If not in cxc, try the module directly from sys.modules
    # Use the public module path if available
    public_module = module.replace("._src.", ".")
    if public_module in sys.modules:
        try:
            mod = sys.modules[public_module]
            resolved = mod
            for part in parts:
                resolved = getattr(resolved, part)
            if isinstance(resolved, type):
                return resolved
        except AttributeError:
            pass

    return cls


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
    # Use a dict keyed by (module, qualname) to deduplicate classes that appear
    # multiple times due to import path issues in editable installs.
    seen: dict[tuple[str, str], type] = {}

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
                # Canonicalize the class to handle duplicates from editable
                # installs in uv-workspaces.
                canonical = _canonicalize_coordinax_class(subclass)

                # Deduplicate by (module, qualname) - only keep first seen
                key = (canonical.__module__, canonical.__qualname__)
                if key not in seen:
                    seen[key] = canonical

            # Always recurse to find deeper subclasses
            recurse(subclass)

    recurse(base_class)

    subclasses = list(seen.values())

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


@ft.lru_cache(maxsize=256)
def get_init_params(cls: type, /) -> Mapping[str, inspect.Parameter]:
    """Get ``__init__`` parameters for a class (excluding 'self').

    This function is cached for performance.

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


@ft.lru_cache(maxsize=256)
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
    >>> shape = strategy.example()
    >>> shape
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


# -----------------------------------------------


@final
class Metadata(TypedDict, total=False):  # closed=False
    """Holds shape and dtype information for strategy generation."""

    dtype: st.SearchStrategy[Any]
    shape: st.SearchStrategy[tuple[int, ...]]
    validators: list[Callable[[Any], bool]]  # Beartype validator functions


@final
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


# ===================================================================
# Strategy generation for type annotations


@ft.lru_cache(maxsize=256)
def cached_strategy_for_annotation(ann_type: type | Wrapper[Any]) -> st.SearchStrategy:
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
def strategy_for_annotation(ann: Wrapper, /, *, meta: Metadata) -> st.SearchStrategy:  # type: ignore[type-arg]
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

    elif inspect.isclass(ann.ann) and issubclass(ann.ann, jaxtyping.AbstractArray):
        typ = ann.ann.array_type
        meta |= parse_jaxtyping_annotation(ann.ann)
    elif hasattr(ann.ann, "__class__") and issubclass(
        ann.ann.__class__, jaxtyping.AbstractArray
    ):
        # Handle jaxtyping instances (e.g., Real[StaticQuantity["length"], ""])
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
