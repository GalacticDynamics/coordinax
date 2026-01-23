"""Hypothesis strategies for coordinax representations."""

__all__ = ("charts", "product_charts")


from typing import Any

import hypothesis.strategies as st
from hypothesis import assume

import coordinax.charts as cxc
from .chart_kwargs import cached_build_init_kwargs_strategy, chart_init_kwargs
from .classes import chart_classes
from coordinax_hypothesis._src.utils import draw_if_strategy, get_all_subclasses


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
    # TODO: support product charts
    exclude = (*exclude, cxc.CartesianProductChart)

    # Handle dimensionality parameter
    dimensionality = draw_if_strategy(draw, dimensionality)
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
    kwargs = draw(chart_init_kwargs(chart_cls, dimensionality=dimensionality))

    # Create the instance
    chart = chart_cls(**kwargs)

    # Filter by dimensionality if specified
    if dimensionality is not None:
        assume(chart.ndim == dimensionality)

    return chart


@st.composite  # type: ignore[untyped-decorator]
def product_charts(
    draw: st.DrawFn,
    /,
    factor_charts: tuple[cxc.AbstractChart[Any, Any], ...]
    | st.SearchStrategy[tuple[cxc.AbstractChart[Any, Any], ...]]
    | None = None,
    factor_names: tuple[str, ...] | st.SearchStrategy[tuple[str, ...]] | None = None,
    *,
    ndim: int | None = None,
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
    ndim
        Target total dimensionality for the product chart. If specified, generates
        factors whose dimensions sum to this value. Only used when factor_charts
        is None. If None, generates random dimensionality.
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

    Generate product with specific total dimensionality:

    >>> @given(chart=cxst.product_charts(ndim=6))
    ... def test_6d_product(chart):
    ...     assert chart.ndim == 6
    ...     assert isinstance(chart, cxc.AbstractCartesianProductChart)

    """
    # If factors or names are specified, only generate CartesianProductChart
    if factor_charts is not None or factor_names is not None:
        if factor_names is not None:
            names = draw_if_strategy(draw, factor_names)

        if factor_charts is None:
            # Generate factors with specified or random dimensionality
            factors, names = draw(
                _generate_product_factors(
                    ndim=ndim,
                    min_factors=min_factors,
                    max_factors=max_factors,
                )
            )
        else:
            factors = draw_if_strategy(draw, factor_charts)
            names = tuple(f"f{i}" for i in range(len(factors)))

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
            # Note: ndim constraint doesn't apply to specialized classes
            kwargs_strategy = cached_build_init_kwargs_strategy(flat_cls, dim=None)
            kwargs = draw(kwargs_strategy)

            return flat_cls(**kwargs)

    # 75% chance (or fallback): generate CartesianProductChart
    factors, names = draw(
        _generate_product_factors(
            ndim=ndim,
            min_factors=min_factors,
            max_factors=max_factors,
        )
    )

    return cxc.CartesianProductChart(factors, names)
