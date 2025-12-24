"""Hypothesis strategies for coordinax representations."""

__all__ = (
    "can_point_transform",
    "chart_classes",
    "charts",
    "charts_like",
    "chart_time_chain",
    "product_charts",
)


import functools as ft

from typing import Any, Final
from typing_extensions import get_annotations

import hypothesis.strategies as st
import plum
from hypothesis import assume

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
from .utils import (
    Metadata,
    build_init_kwargs_strategy,
    draw_if_strategy,
    get_all_subclasses,
    strategy_for_annotation,
    wrap_if_not_inspectable,
)


# Cache build_init_kwargs_strategy since it's called repeatedly for the same classes
# Strategies are immutable and deterministic, so this is safe
@ft.lru_cache(maxsize=128)
def cached_build_init_kwargs_strategy(
    cls: type, *, dim: int | None
) -> st.SearchStrategy:
    """Return cached wrapper around build_init_kwargs_strategy."""
    return build_init_kwargs_strategy(cls, dim=dim)


# =============================================================================
# Main strategies


def can_point_transform(
    to_chart: cxc.AbstractChart[Any, Any], from_chart: cxc.AbstractChart[Any, Any], /
) -> bool:
    """Return True if ``point_transform`` can convert between the two reps."""
    if type(to_chart) is type(from_chart):
        return True
    try:
        _ = to_chart.cartesian
        _ = from_chart.cartesian
    except (NotImplementedError, ValueError):
        return False
    return True


@st.composite  # type: ignore[untyped-decorator]
def chart_classes(
    draw: st.DrawFn,
    /,
    filter: type
    | tuple[type, ...]
    | st.SearchStrategy[type | tuple[type, ...]] = object,
    *,
    exclude_abstract: bool | st.SearchStrategy[bool] = True,
    exclude: tuple[type, ...] = (),
) -> type[cxc.AbstractChart[Any, Any]]:
    """Strategy to draw representation classes.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    filter
        A class or tuple of classes to limit the representations to, by default
        `object`.  Can be a single type or a tuple of types.
    exclude_abstract
        Whether to exclude abstract subclasses, by default True.
    exclude
        Specific classes to exclude, by default
        ``(coordinax.charts.Abstract0D, coordinax.charts.TwoSphere)``.

    Returns
    -------
    type[AbstractChart]
        A representation class.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    >>> # Draw any representation class
    >>> rep_class_strategy = cxst.chart_classes()
    >>> # Draw 1D velocity representation classes only
    >>> vel_1d_chart_class_strategy = cxst.chart_classes(
    ...     filter=cxc.Abstract1D)

    """
    classes = get_all_subclasses(
        cxc.AbstractChart,
        filter=draw_if_strategy(draw, filter),
        exclude_abstract=draw_if_strategy(draw, exclude_abstract),
        exclude=exclude,
    )
    return draw(st.sampled_from(classes))


# ===================================================================


@st.composite  # type: ignore[untyped-decorator]
def charts(
    draw: st.DrawFn,
    /,
    filter: type | tuple[type, ...] | st.SearchStrategy[type | tuple[type, ...]] = (),
    *,
    exclude: tuple[type, ...] = (cxc.Abstract0D, cxc.TwoSphere),
    dimensionality: (int | None | st.SearchStrategy[int | None]) = None,
) -> cxc.AbstractChart[Any, Any]:
    """Strategy to draw representation instances.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    filter
        A class or tuple of classes to limit the representations to, by default
        `()` (no additional filter). Can be a single type or a tuple of types.

        For example: - `coordinax.charts.Abstract0D` to limit to 0D representations.
        - `coordinax.charts.Abstract1D` to limit to 1D representations.  -
        `coordinax.charts.Abstract2D` to limit to 2D representations.  -
        `coordinax.charts.Abstract3D` to limit to 3D representations.

        In combination, this can be used to draw representations that satisfy
        multiple criteria, e.g., `filter=(coordinax.charts.Abstract3D,
        coordinax.charts.AbstractSpherical3D)`.  Just note that some 3-D
        representations, e.g. SpaceTimeCT[Cart2D] are not subclasses of
        `coordinax.charts.Abstract3D` and will be excluded unless explicitly added.
    exclude
        Specific classes to exclude, by default ().
    dimensionality
        Dimensionality constraint for the representation. Can be: - `None`: No
        constraint - An integer: Exact dimensionality match - A strategy: Draw
        dimensionality from strategy (e.g.,
          `st.integers(min_value=1, max_value=2)`)

    Returns
    -------
    AbstractChart
        An instance of a representation class.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst
    >>> import hypothesis.strategies as st

    >>> # Draw any representation instance (dimensionality > 0 by default)
    >>> rep_strategy = cxst.charts()
    >>> rep = rep_strategy.example()
    >>> isinstance(rep, cxc.AbstractChart)
    True
    >>> rep.ndim > 0
    True

    >>> # Draw representations with exact dimensionality
    >>> exact_2d_strategy = cxst.charts(dimensionality=2)
    >>> exact_2d_chart = exact_2d_strategy.example()
    >>> exact_2d_chart.ndim == 2
    True

    >>> # Include 0-dimensional representations
    >>> all_dim_strategy = cxst.charts(dimensionality=None, exclude=())
    >>> all_dim_rep = all_dim_strategy.example()
    >>> isinstance(all_dim_rep, cxc.AbstractChart)
    True

    >>> # Use a strategy to draw dimensionality
    >>> strategy_dim = cxst.charts(
    ...     dimensionality=st.integers(min_value=1, max_value=2))
    >>> strategy_dim_rep = strategy_dim.example()
    >>> 1 <= strategy_dim_rep.ndim <= 2
    True

    """
    # Handle dimensionality parameter
    dimensionality = draw_if_strategy(draw, dimensionality)
    if isinstance(dimensionality, int) and dimensionality != 2:
        exclude = (*exclude, cxc.EmbeddedManifold)
    # Exclude all dimensional flags except the target
    if isinstance(dimensionality, int) and dimensionality in cxc.DIMENSIONAL_FLAGS:
        exclude = exclude + tuple(
            flag for i, flag in cxc.DIMENSIONAL_FLAGS.items() if i != dimensionality
        )

    # Draw the representation class
    chart_cls = draw(
        chart_classes(
            filter=draw_if_strategy(draw, filter),
            exclude_abstract=True,
            exclude=exclude,
        )
    )

    # Build and draw kwargs for required parameters
    kwargs_strategy = cached_build_init_kwargs_strategy(chart_cls, dim=dimensionality)
    kwargs = draw(kwargs_strategy)

    # Create the instance
    chart = chart_cls(**kwargs)

    # Filter by dimensionality if specified
    if dimensionality is not None:
        assume(chart.ndim == dimensionality)

    return chart


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cxc.SpaceTimeCT],  # type: ignore[type-arg]
    /,
    *,
    dim: int | None,
) -> st.SearchStrategy:
    """Specialized strategy for SpaceTimeCT classes.

    Parameters
    ----------
    cls : type[cxc.SpaceTimeCT]
        The SpaceTimeCT class.
    dim : int | None
        The required dimensionality for the spatial_chart, or None for any
        dimensionality.

    Returns
    -------
    st.SearchStrategy[dict[str, Any]]
        A strategy that generates dictionaries with 'spatial_chart' key.
        The 'c' parameter is optional and uses the default value.

    """
    # Generate spatial_chart: any AbstractChart except SpaceTimeCT itself
    # If dimensionality is specified, use it; otherwise allow any dimensionality > 0
    spatial_chart_strategy = charts(
        exclude=(cxc.SpaceTimeCT,),
        dimensionality=dim if dim is None else dim - 1,
    )
    # Generate 'c' parameter: either use default or draw from annotation
    c = st.one_of(
        st.just(cls.__dataclass_fields__["c"].default),
        strategy_for_annotation(
            wrap_if_not_inspectable(get_annotations(cls)["c"]), meta=Metadata()
        ),
    )
    return st.fixed_dictionaries({"spatial_chart": spatial_chart_strategy, "c": c})


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cxc.SpaceTimeEuclidean],  # type: ignore[type-arg]
    /,
    *,
    dim: int | None,
) -> st.SearchStrategy:
    """Specialized strategy for SpaceTimeEuclidean classes."""
    spatial_chart_strategy = charts(
        exclude=(cxc.SpaceTimeEuclidean,),
        dimensionality=dim if dim is None else dim - 1,
    )
    c = st.one_of(
        st.just(cls.__dataclass_fields__["c"].default),
        strategy_for_annotation(
            wrap_if_not_inspectable(get_annotations(cls)["c"]), meta=Metadata()
        ),
    )
    return st.fixed_dictionaries({"spatial_chart": spatial_chart_strategy, "c": c})


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cxc.EmbeddedManifold],  # type: ignore[type-arg]
    /,
    *,
    dim: int | None,
) -> st.SearchStrategy:
    """Specialized strategy for EmbeddedManifold.

    Currently supports TwoSphere embedded in Cart3D with a length scale ``R``.
    """
    del cls, dim
    R = st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ).map(lambda v: u.Q(v, "km"))
    params = st.fixed_dictionaries({"R": R})
    return st.fixed_dictionaries(
        {
            "intrinsic_chart": st.just(cxc.twosphere),
            "ambient_chart": st.just(cxc.cart3d),
            "params": params,
        }
    )


@st.composite  # type: ignore[untyped-decorator]
def charts_like(
    draw: st.DrawFn,
    /,
    representation: cxc.AbstractChart[Any, Any]
    | st.SearchStrategy[cxc.AbstractChart[Any, Any]],
) -> cxc.AbstractChart[Any, Any]:
    """Generate representations similar to the provided one.

    This strategy inspects the provided representation to determine its flags
    (e.g., Abstract1D, Abstract2D, Abstract3D, AbstractSpherical3D, etc.) and
    dimensionality, then generates new representations matching those criteria.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    representation
        The template representation to match, or a strategy that generates one.

    Returns
    -------
    AbstractChart
        A new representation instance with the same flags and dimensionality
        as the template.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst
    >>> from hypothesis import given

    >>> # Generate representations
    >>> @given(chart=cxst.charts_like(r.cart3d))
    ... def test_3d(rep):
    ...     assert isinstance(rep, r.Abstract3D)
    ...     assert rep.ndim == 3

    >>> # Generate representations like a 2D representation
    >>> @given(chart=cxst.charts_like(r.polar2d))
    ... def test_2d(rep):
    ...     assert isinstance(rep, r.Abstract2D)
    ...     assert rep.ndim == 2

    """
    # Draw the template representation if it's a strategy
    template = draw_if_strategy(draw, representation)

    # Extract flags by looking through the MRO for AbstractDimensionalFlag subclasses
    flags = tuple(
        base
        for base in type(template).mro()
        if (
            issubclass(base, cxc.AbstractDimensionalFlag)
            and base is not cxc.AbstractDimensionalFlag
        )
    )

    # If no flags found, just use the base type
    if not flags:
        flags = (type(template),)

    # Generate a new representation with the same flags and dimensionality.
    # Keep the template's class available even if it's excluded by default.
    exclude = tuple(
        ex for ex in (cxc.Abstract0D, cxc.TwoSphere) if not isinstance(template, ex)
    )
    chart = draw(charts(filter=flags, dimensionality=template.ndim, exclude=exclude))
    assume(can_point_transform(chart, template))
    assume(can_point_transform(template, chart))
    return chart


MAX_TIME_CHAIN_ITERS: Final = 50
MAX_TIME_CHAIN_ITER_MSG: Final = (
    f"Exceeded maximum iterations ({MAX_TIME_CHAIN_ITERS}) while building "
    "time antiderivative chain. This likely indicates a bug in the "
    "representation's time_antiderivative property."
)


@st.composite  # type: ignore[untyped-decorator]
def chart_time_chain(
    draw: st.DrawFn,
    role: type[cx.roles.AbstractRole],
    chart: cxc.AbstractChart[Any, Any] | st.SearchStrategy[cxc.AbstractChart[Any, Any]],
    /,
) -> tuple[cxc.AbstractChart[Any, Any], ...]:
    """Generate a chain of representations following time antiderivative pattern.

    Given a representation (position, velocity, or acceleration), this strategy
    returns a tuple containing representations that match the flags of each time
    antiderivative up to and including a position representation. Each element
    in the chain is generated using `charts_like()` to match the flags
    of the corresponding time antiderivative.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    role
        The role flag for the starting representation (e.g., `cx.roles.Pos`,
        `cx.roles.Vel`, or `cx.roles.Acc`).
    chart
        The starting chart or a strategy that generates one.

    Returns
    -------
    tuple[AbstractChart, ...]
        A tuple of representations matching the time antiderivative chain.
        Each representation matches the flags of the corresponding time
        antiderivative but may be a different instance.
        - If input is position: (pos_rep,)
        - If input is velocity: (vel_rep, pos_rep)
        - If input is acceleration: (acc_rep, vel_rep, pos_rep)

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    >>> # Given an acceleration, get (acc, vel, pos) chain
    >>> @given(chain=cxst.chart_time_chain(cx.roles.Acc, cxc.cart3d))
    ... def test_chain(chain):
    ...     acc_rep, vel_rep, pos_rep = chain
    ...     assert isinstance(acc_rep, cxc.AbstractChart)
    ...     assert isinstance(vel_rep, cxc.AbstractChart)
    ...     assert isinstance(pos_rep, cxc.AbstractChart)

    """
    # Draw the starting representation if it's a strategy
    start_chart = draw_if_strategy(draw, chart)

    # Build the chain by following time_antiderivative until we reach position
    chain = [start_chart]
    current_role = role

    # Keep getting time antiderivatives until we reach a position representation
    # Safety: limit iterations to prevent infinite loops (no representation
    # should have more than a few time antiderivatives)
    i = 0
    while current_role is not cx.roles.Pos:
        i += 1
        if i > MAX_TIME_CHAIN_ITERS:
            raise RuntimeError(MAX_TIME_CHAIN_ITER_MSG)

        # Get a representation like the time antiderivative
        current = draw(charts_like(representation=chain[-1]))

        # Store and move to next role
        chain.append(current)
        current_role = current_role.antiderivative()

    return tuple(chain)


@st.composite  # type: ignore[untyped-decorator]
def product_charts(
    draw: st.DrawFn,
    /,
    factor_charts: tuple[cxc.AbstractChart[Any, Any], ...]
    | st.SearchStrategy[tuple[cxc.AbstractChart[Any, Any], ...]]
    | None = None,
    factor_names: tuple[str, ...] | st.SearchStrategy[tuple[str, ...]] | None = None,
    *,
    min_factors: int = 1,
    max_factors: int = 3,
) -> cxc.AbstractCartesianProductChart[Any, Any]:
    """Generate Cartesian product chart instances.

    Generates product charts with a weighted distribution:
    - 25% chance: specialized flat-key products (concrete subclasses of
      `AbstractFlatCartesianProductChart` like `SpaceTimeCT`, `SpaceTimeEuclidean`)
    - 75% chance: general `CartesianProductChart` with namespaced keys

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    factor_charts
        Fixed tuple of factor chart instances, or strategy generating one.
        If None, generates random charts with count between min_factors and max_factors.
        When provided, only generates `CartesianProductChart` (not specializations).
    factor_names
        Fixed tuple of factor names, or strategy generating one.
        If None, generates names like "f0", "f1", etc.
        When provided, only generates `CartesianProductChart` (not specializations).
    min_factors
        Minimum number of factors when generating random factor_charts (default: 1).
    max_factors
        Maximum number of factors when generating random factor_charts (default: 3).

    Returns
    -------
    AbstractCartesianProductChart
        A product chart instance.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc
    >>> import coordinax_hypothesis as cxst

    Generate any product chart:

    >>> @given(chart=cxst.product_charts())
    ... def test_product_chart(chart):
    ...     assert isinstance(chart, cxc.AbstractCartesianProductChart)
    ...     assert len(chart.factors) >= 1

    Generate product with specific factors:

    >>> @given(chart=cxst.product_charts(
    ...     factor_charts=(cxc.cart3d, cxc.cart3d),
    ...     factor_names=("q", "p")
    ... ))
    ... def test_phase_space(chart):
    ...     assert chart.ndim == 6
    ...     assert len(chart.factors) == 2
    ...     assert isinstance(chart, cxc.CartesianProductChart)

    """
    # If factors or names are specified, only generate CartesianProductChart
    if factor_charts is not None or factor_names is not None:
        if factor_charts is None:
            # Generate random number of factors
            n_factors = draw(st.integers(min_value=min_factors, max_value=max_factors))
            factor_exclude = (
                cxc.Abstract0D,
                cxc.AbstractCartesianProductChart,
                cxc.EmbeddedManifold,
            )
            factors = tuple(
                draw(charts(exclude=factor_exclude)) for _ in range(n_factors)
            )
        else:
            factors = draw_if_strategy(draw, factor_charts)

        if factor_names is None:
            names = tuple(f"f{i}" for i in range(len(factors)))
        else:
            names = draw_if_strategy(draw, factor_names)

        assume(len(factors) == len(names))
        return cxc.CartesianProductChart(factors, names)

    # 25% chance to generate a specialized flat-key product
    if draw(st.integers(min_value=1, max_value=100)) <= 25:
        # Get all concrete subclasses of AbstractFlatCartesianProductChart
        flat_product_classes = get_all_subclasses(
            cxc.AbstractFlatCartesianProductChart,
            exclude_abstract=True,
            exclude=(),
        )

        if not flat_product_classes:
            # Fallback to CartesianProductChart if no specializations found
            pass
        else:
            # Pick a random specialization
            flat_cls = draw(st.sampled_from(flat_product_classes))

            # Build kwargs using the existing strategy infrastructure
            kwargs_strategy = cached_build_init_kwargs_strategy(flat_cls, dim=None)
            kwargs = draw(kwargs_strategy)

            return flat_cls(**kwargs)

    # 75% chance (or fallback): generate CartesianProductChart
    n_factors = draw(st.integers(min_value=min_factors, max_value=max_factors))
    factor_exclude = (
        cxc.Abstract0D,
        cxc.AbstractCartesianProductChart,
        cxc.EmbeddedManifold,
    )
    factors = tuple(draw(charts(exclude=factor_exclude)) for _ in range(n_factors))
    names = tuple(f"f{i}" for i in range(len(factors)))

    return cxc.CartesianProductChart(factors, names)


# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxc.AbstractChart, lambda _: charts())

for flag_cls in get_all_subclasses(cxc.AbstractDimensionalFlag, exclude_abstract=False):
    # Skip representation base classes
    if issubclass(flag_cls, cxc.AbstractChart):
        continue

    st.register_type_strategy(flag_cls, lambda typ: charts(typ))
