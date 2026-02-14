"""Hypothesis strategies for coordinax representations."""

__all__ = ("product_charts",)

from typing import Any

import hypothesis.strategies as st
import plum
from hypothesis import assume

import coordinax.charts as cxc
from .chart_kwargs import cached_build_init_kwargs_strategy
from coordinax_hypothesis.utils import draw_if_strategy, get_all_subclasses

# =============================================================================
# Helper for generating product chart factors


@st.composite
def _generate_product_factors(
    draw: st.DrawFn,
    /,
    *,
    ndim: int | None = None,
    min_factors: int = 1,
    max_factors: int = 4,
) -> tuple[tuple[cxc.AbstractChart[Any, Any], ...], tuple[str, ...]]:
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
    # Determine target dimensionality
    target_dim = (
        ndim
        if ndim is not None
        else draw(st.integers(max(2, min_factors), max(6, min_factors)))
    )

    # Determine number of factors (at least min_factors, at most max_factors)
    n_factors = draw(
        st.integers(min_value=min_factors, max_value=min(target_dim, max_factors))
    )

    # Generate factor dimensions that sum to target_dim
    if n_factors == 1:
        factor_dims = [target_dim]
    else:
        # Generate n_factors-1 random splits and derive dimensions
        remaining = target_dim
        factor_dims = []
        for i in range(n_factors - 1):
            # Each factor needs at least 1 dimension
            max_for_this = remaining - (n_factors - i - 1)
            if max_for_this < 1:
                factor_dim = 1
            else:
                factor_dim = draw(st.integers(min_value=1, max_value=max_for_this))
            factor_dims.append(factor_dim)
            remaining -= factor_dim
        factor_dims.append(remaining)  # Last factor gets what's left

    # Generate the actual chart factors
    factors = []
    for factor_dim in factor_dims:
        # Exclude abstract charts, product charts (to avoid recursion), and
        # charts with unresolvable TypeVars Late import to avoid circular import
        # (core.py imports from this module)
        from coordinax_hypothesis.charts import charts  # noqa: PLC0415

        chart = draw(charts(filter=cxc.AbstractFixedComponentsChart, ndim=factor_dim))
        factors.append(chart)

    # Generate names for each factor
    names = tuple(f"f{i}" for i in range(len(factors)))

    return tuple(factors), names


# =============================================================================


@st.composite
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
    >>> import coordinax_hypothesis.core as cxst

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
            factors, gen_names = draw(
                _generate_product_factors(
                    ndim=ndim, min_factors=min_factors, max_factors=max_factors
                )
            )
            if factor_names is None:
                names = gen_names
        else:
            factors = draw_if_strategy(draw, factor_charts)
            if factor_names is None:
                names = tuple(f"f{i}" for i in range(len(factors)))

        assume(len(factors) == len(names))
        return cxc.CartesianProductChart(factors, names)

    # 25% chance to generate a specialized flat-key product
    if draw(st.integers(min_value=1, max_value=100)) <= 25:
        # Get all concrete subclasses of AbstractFlatCartesianProductChart
        flat_product_classes = get_all_subclasses(
            cxc.AbstractFlatCartesianProductChart, exclude_abstract=True, exclude=()
        )

        # Filter by ndim if specified
        if ndim is not None:
            flat_product_classes = [
                cls for cls in flat_product_classes if cls().ndim == ndim
            ]

        # Fallback to CartesianProductChart if no matching specializations.
        if not flat_product_classes:
            pass
        else:
            # Pick a random specialization
            flat_cls = draw(st.sampled_from(flat_product_classes))

            # Build kwargs using the existing strategy infrastructure
            kwargs_strategy = cached_build_init_kwargs_strategy(flat_cls, dim=None)
            kwargs = draw(kwargs_strategy)

            return flat_cls(**kwargs)

    # 75% chance (or fallback): generate CartesianProductChart
    factors, names = draw(
        _generate_product_factors(
            ndim=ndim, min_factors=min_factors, max_factors=max_factors
        )
    )

    return cxc.CartesianProductChart(factors, names)


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cxc.CartesianProductChart], /, *, dim: int | None
) -> st.SearchStrategy:
    """Specialized strategy for CartesianProductChart.

    Parameters
    ----------
    cls : type[cxc.CartesianProductChart]
        The CartesianProductChart class.
    dim : int | None
        The required total dimensionality for the product, or None for any
        dimensionality. The strategy will generate factors whose dimensions
        sum to this value.

    Returns
    -------
    st.SearchStrategy[dict[str, Any]]
        A strategy that generates dictionaries with 'factors' and 'names' keys.
        The factors are a tuple of charts from AbstractFixedComponentsChart
        with collective dimension equal to dim.

    """
    del cls

    @st.composite
    def _kwargs_strategy(draw: st.DrawFn) -> dict[str, Any]:
        """Generate kwargs dict for CartesianProductChart."""
        factors, names = draw(_generate_product_factors(ndim=dim))
        return {"factors": factors, "factor_names": names}

    return _kwargs_strategy()
