"""Hypothesis strategies for coordinax representations."""

__all__: tuple[str, ...] = ("cartesian_product_factors",)

import functools as ft
from typing import Any

import hypothesis.strategies as st
import plum
import unxt as u
import unxt_hypothesis as ust
from hypothesis import assume

import coordinax.charts as cxc

from .charts import charts
from coordinax.hypothesis.utils import (
    draw_if_strategy,
    get_all_subclasses,
    strip_return_annotation,
)

##############################################################################
# Cartesian Products


def _factor_dims(draw: st.DrawFn, /, *, n_factors: int, target_dim: int) -> list[int]:
    """Generate factor dimensions that sum to `target_dim`.

    Parameters
    ----------
    draw : `hypothesis.strategies.DrawFn`
        Draw function.
    n_factors : int
        Number of factor charts.
    target_dim : int
        Target total dimensionality of the factor charts.

    Returns
    -------
    list[int]
        A list of factor dimensions that sum to `target_dim`.

    """
    if n_factors == 1:
        factor_dims = [target_dim]
    else:
        # Draw n_factors-1 random splits; the last factor's dim is determined by
        # what remains, ensuring all factors sum exactly to `target_dim`.
        remaining = target_dim
        factor_dims = []
        for i in range(n_factors - 1):
            # Reserve at least 1 dimension for each factor still to be assigned
            # (the current one plus the remaining ones after it), so no factor
            # ends up with 0 dimensions.
            max_for_this = remaining - (n_factors - i - 1)
            # If the budget is already tight (max_for_this < 1), assign 1 to
            # avoid an invalid range; otherwise draw uniformly.
            factor_dim = 1 if max_for_this < 1 else draw(st.integers(1, max_for_this))
            factor_dims.append(factor_dim)
            remaining -= factor_dim
        factor_dims.append(remaining)  # Last factor gets the remaining budget

    return factor_dims


@st.composite
def cartesian_product_factors(
    draw: st.DrawFn,
    /,
    *,
    ndim: int | None = None,
    min_factors: int = 1,
    max_factors: int = 4,
) -> tuple[cxc.AbstractChart[Any, Any], ...]:
    """Generate factors and names for a product chart with target dimensionality.

    Parameters
    ----------
    draw
        Hypothesis draw function.
    ndim
        Target total dimensionality. If None, generates random dim between 2-6.
    min_factors
        Minimum number of factors.
    max_factors
        Maximum number of factors.

    Returns
    -------
    tuple[tuple[AbstractChart, ...], tuple[str, ...]]
        A tuple of (factors, names) where factors are chart instances
        and names are corresponding factor names.

    """
    # Confirm the dimensionality
    if min_factors < 1:
        raise ValueError("min_factors must be at least 1")
    if max_factors < min_factors:
        raise ValueError("max_factors must be >= to min_factors")

    # Determine target dimensionality. If not specified, pick a random dim.
    # Single-factor products can validly be 1D, while multi-factor products
    # must have at least 2 total dimensions.
    if ndim is not None:
        target_dim = ndim
    else:
        min_ndim = 1 if max_factors == 1 else max(2, min_factors)
        max_ndim = min(6, max_factors)
        target_dim = draw(st.integers(min_ndim, max_ndim))

    # Determine number of factors (at least min_factors, at most max_factors)
    n_factors = draw(st.integers(min_factors, min(target_dim, max_factors)))

    # Generate factor dimensions that sum to target_dim
    factor_dims = _factor_dims(draw, n_factors=n_factors, target_dim=target_dim)

    # Generate the actual chart factors
    factors = []
    for factor_dim in factor_dims:
        # Exclude abstract charts, product charts (to avoid infinite recursion),
        # and charts with unresolvable TypeVars Late import to avoid circular
        # import (core.py imports from this module)
        chart = draw(charts(filter=cxc.AbstractFixedComponentsChart, ndim=factor_dim))  # ty: ignore[missing-argument]
        factors.append(chart)

    return tuple(factors)


@plum.dispatch
@strip_return_annotation
@st.composite
def charts(  # noqa: F811
    draw: st.DrawFn,
    chart_cls: type[cxc.CartesianProductChart],
    /,
    *,
    factor_charts: tuple[cxc.AbstractChart, ...] | st.SearchStrategy | None = None,
    factor_names: tuple[str, ...] | st.SearchStrategy | None = None,
    ndim: int | None = None,
    min_factors: int = 1,
    max_factors: int = 3,
) -> cxc.CartesianProductChart:
    """Generate a CartesianProductChart with specified or random factors and names.

    Parameters
    ----------
    draw : `hypothesis.strategies.DrawFn`
        Draw function.
    chart_cls : type[CartesianProductChart] | None
        The chart class. Must be ``coordinax.charts.CartesianProductChart`` class.
    factor_charts : tuple[AbstractChart,] | SearchStrategy[thereof] | None
        Either a fixed tuple of factor chart instances, or a strategy that
        generates such tuples. If provided, only generates a
        `coordinax.charts.CartesianProductChart` with these factors (no
        specialized product charts). If None, generates random factor charts
        with count between min_factors and max_factors.
    factor_names : tuple[str,] | SearchStrategy[tuple[str,] | None] | None
        Either a fixed tuple of factor names, or a strategy that generates such
        tuples. If provided, only generates a
        `coordinax.charts.CartesianProductChart` with these names (no
        specialized product charts). If `None`, generates default names like
        "f0", "f1", etc.
    ndim : int | None
        Target total dimensionality for the product chart. If specified,
        generates factors whose dimensions sum to this value. Only used when
        `factor_charts` is `None`.
    min_factors : int
        Minimum number of factors when generating random `factor_charts`.
        Ignored if `factor_charts` is provided.
    max_factors : int
        Maximum number of factors when generating random `factor_charts`.
        Ignored if `factor_charts` is provided.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc
    >>> import coordinax.hypothesis.main as cxst

    >>> @given(chart=cxst.charts(cxc.CartesianProductChart))
    ... def test_product_chart(chart):
    ...     assert isinstance(chart, cxc.CartesianProductChart)

    """
    # If factors or names are specified, only generate CartesianProductChart
    if factor_charts is not None or factor_names is not None:
        # Generate the factor_charts. These can be generated, or user-provided.
        if factor_charts is not None:  # user-provided
            factors = draw_if_strategy(draw, factor_charts)
        else:
            # Generate factors with specified or random dimensionality
            factors = draw(
                cartesian_product_factors(  # ty: ignore[missing-argument]
                    ndim=ndim, min_factors=min_factors, max_factors=max_factors
                )
            )

        # Build `factor_names`
        maybe_names = draw_if_strategy(draw, factor_names)
        names = (
            tuple(f"f{i}" for i in range(len(factors)))
            if maybe_names is None
            else maybe_names
        )

        assume(len(factors) == len(names))

    else:
        factors = draw(
            cartesian_product_factors(  # ty: ignore[missing-argument]
                ndim=ndim, min_factors=min_factors, max_factors=max_factors
            )
        )
        names = tuple(f"f{i}" for i in range(len(factors)))

    return cxc.CartesianProductChart(factors, names)


@plum.dispatch
@ft.lru_cache(maxsize=128, typed=True)
@strip_return_annotation
@st.composite
def chart_init_kwargs(
    draw: st.DrawFn,
    chart_class: type[cxc.CartesianProductChart],
    /,
    *,
    ndim: int | None = None,
) -> dict[str, Any]:
    """Specialized strategy for CartesianProductChart.

    Parameters
    ----------
    chart_class : type[cxc.CartesianProductChart]
        The `coordinax.charts.CartesianProductChart` class.
    ndim : int | None
        The required total dimensionality for the product, or `None` for any
        dimensionality. The strategy will generate factors whose dimensions sum
        to this value.

    """
    del chart_class

    factors = draw(cartesian_product_factors(ndim=ndim))  # ty: ignore[missing-argument]
    names = tuple(f"f{i}" for i in range(len(factors)))

    return {"factors": factors, "factor_names": names}


##############################################################################
# Specialized Cartesian Product Charts


@plum.dispatch
@strip_return_annotation
@st.composite
def charts(  # noqa: F811
    draw: st.DrawFn, chart_cls: type[cxc.SpaceTimeCT], /, *, ndim: int | None = None
) -> Any:
    """Generate SpaceTimeCT charts with random spatial factors and `c` values.

    Parameters
    ----------
    draw
        Hypothesis draw function.
    chart_cls
        Class of the chart.
    ndim
        Target total dimensionality (including time). If specified, generates
        spatial charts with ndim-1 dimensions. If `None`, generates 3-D spatial charts.

    Returns
    -------
    SpaceTimeCT
        A `SpaceTimeCT` instance with a random spatial chart and `c` value.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc
    >>> import coordinax.hypothesis.main as cxst

    >>> @given(chart=cxst.charts(cxc.SpaceTimeCT))
    ... def test_spacetime_chart(chart):
    ...     assert isinstance(chart, cxc.SpaceTimeCT)

    """
    # Draw a spatial chart with one less dimension than the target, since
    # SpaceTimeCT has 1 time dimension plus the spatial chart's dimensions.
    ndim = 4 if ndim is None else ndim
    spatial_chart = draw(  # exclude=() allows 0-D
        charts(filter=cxc.AbstractFixedComponentsChart, exclude=(), ndim=ndim - 1)  # ty: ignore[missing-argument]
    )

    # Have a 10% chance to generate `c` different than the default value
    if draw(st.integers(1, 100)) <= 10:
        c = draw(
            ust.quantities(  # ty: ignore[missing-argument]
                unit="km/s",
                quantity_cls=u.StaticQuantity,
                shape=(),
                static_value=True,
                elements=st.floats(min_value=1e6, max_value=5e6, width=32),
            )
        )
        c = c.uconvert(draw(ust.units("speed")))  # ty: ignore[missing-argument]
    else:
        c = cxc.SpaceTimeCT.__dataclass_fields__["c"].default

    return cxc.SpaceTimeCT(spatial_chart=spatial_chart, c=c)


##############################################################################
# General product chart strategy


@plum.dispatch
@strip_return_annotation
@st.composite
def charts(  # noqa: F811
    draw: st.DrawFn,
    chart_cls: type[cxc.AbstractCartesianProductChart],
    /,
    *,
    factor_charts: tuple[cxc.AbstractChart, ...] | st.SearchStrategy | None = None,
    factor_names: tuple[str, ...] | st.SearchStrategy | None = None,
    ndim: int | None = None,
    min_factors: int = 1,
    max_factors: int = 3,
) -> Any:
    """Generate Cartesian product chart instances.

    Generates product charts with a weighted distribution:
    - 25% chance: specialized flat-key products (concrete subclasses of
      `AbstractFlatCartesianProductChart` like `SpaceTimeCT`)
    - 75% chance: general `CartesianProductChart` with namespaced keys

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    factor_charts
        Fixed tuple of factor chart instances, or strategy generating one.  If
        None, generates random charts with count between min_factors and
        max_factors.  When provided, only generates `CartesianProductChart` (not
        specializations).
    factor_names
        Fixed tuple of factor names, or strategy generating one.  If `None`,
        generates names like "f0", "f1", etc.  When provided, only generates
        `CartesianProductChart` (not specializations).
    ndim
        Target total dimensionality for the product chart. If specified,
        generates factors whose dimensions sum to this value. Only used when
        factor_charts is None. If `None`, generates random dimensionality.
    min_factors
        Minimum number of factors when generating random factor_charts.
    max_factors
        Maximum number of factors when generating random factor_charts.

    Returns
    -------
    AbstractCartesianProductChart
        A product chart instance.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc
    >>> import coordinax.hypothesis.main as cxst

    Generate any product chart:

    >>> @given(chart=cxst.charts(cxc.AbstractCartesianProductChart))
    ... def test_product_chart(chart):
    ...     assert isinstance(chart, cxc.AbstractCartesianProductChart)
    ...     assert len(chart.factors) >= 1

    Generate product with specific factors:

    >>> @given(chart=cxst.charts(
    ...     cxc.AbstractCartesianProductChart,
    ...     factor_charts=(cxc.cart3d, cxc.cart3d),
    ...     factor_names=("q", "p")
    ... ))
    ... def test_phase_space(chart):
    ...     assert chart.ndim == 6
    ...     assert len(chart.factors) == 2
    ...     assert isinstance(chart, cxc.CartesianProductChart)

    Generate product with specific total dimensionality:

    >>> @given(chart=cxst.charts(cxc.AbstractCartesianProductChart, ndim=6))
    ... def test_6d_product(chart):
    ...     assert chart.ndim == 6
    ...     assert isinstance(chart, cxc.AbstractCartesianProductChart)

    """
    # If factor_charts/names are not provided we have a 25% chance to generate a
    # specialized product chart.
    if (factor_charts is None and factor_names is None) and draw(
        st.integers(1, 100)
    ) <= 25:
        # 25% chance to generate a specialized product chart
        # Get all concrete subclasses of AbstractFlatCartesianProductChart
        flat_product_classes = get_all_subclasses(
            cxc.AbstractFlatCartesianProductChart, exclude_abstract=True, exclude=()
        )
        # Pick a random specialization
        flat_cls = draw(st.sampled_from(flat_product_classes))

        chart = draw(charts(flat_cls, ndim=ndim))

    else:
        chart = draw(
            charts(
                cxc.CartesianProductChart,
                factor_charts=factor_charts,
                factor_names=factor_names,
                ndim=ndim,
                min_factors=min_factors,
                max_factors=max_factors,
            )
        )

    return chart
