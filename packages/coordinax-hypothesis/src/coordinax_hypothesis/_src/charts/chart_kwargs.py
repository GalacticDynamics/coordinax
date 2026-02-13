"""Hypothesis strategies for coordinax representations."""

__all__ = ("chart_init_kwargs", "build_init_kwargs_strategy")


import functools as ft
import inspect

from typing import Any

import hypothesis.strategies as st
import plum

import unxt as u

import coordinax.charts as cxc
import coordinax.embeddings as cxe
from coordinax_hypothesis._src.utils import (
    cached_strategy_for_annotation,
    draw_if_strategy,
    get_init_params,
    wrap_if_not_inspectable,
)


@st.composite
def chart_init_kwargs(
    draw: st.DrawFn,
    /,
    chart_class: type[cxc.AbstractChart] | st.SearchStrategy[type[cxc.AbstractChart]],  # type: ignore[type-arg]
    *,
    ndim: (int | None | st.SearchStrategy[int | None]) = None,
    dimensionality: (int | None | st.SearchStrategy[int | None]) = None,
) -> dict[str, Any]:
    """Strategy to draw initialization kwargs for a chart class.

    This strategy generates valid keyword arguments that can be used to
    instantiate a given chart class. It inspects the chart class's
    initialization signature and generates appropriate values for all
    required parameters.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    chart_class
        The chart class to generate init kwargs for, or a strategy that
        generates one. Must be a subclass of `AbstractChart`.
    ndim
        Optional `chart.ndim` constraint (currently unused, reserved for future
        functionality). By default None.
    dimensionality
        Deprecated alias for `ndim`.

    Returns
    -------
    dict[str, Any]
        A dictionary of keyword arguments suitable for instantiating the
        chart class.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax_hypothesis as cxst
    >>> from hypothesis import given

    Generate init kwargs for a specific chart class:

    >>> @given(kwargs=cxst.chart_init_kwargs(cxc.SpaceTimeCT))
    ... def test_spacetime_kwargs(kwargs):
    ...     # kwargs will contain 'spatial_chart' and 'c' keys
    ...     assert 'spatial_chart' in kwargs
    ...     chart = cxc.SpaceTimeCT(**kwargs)
    ...     assert isinstance(chart, cxc.SpaceTimeCT)

    Use with chart_classes strategy:

    >>> @given(
    ...     chart_cls=cxst.chart_classes(filter=cxc.Abstract3D),
    ...     kwargs=cxst.chart_init_kwargs(
    ...         cxst.chart_classes(filter=cxc.Abstract3D)
    ...     )
    ... )
    ... def test_3d_chart_construction(chart_cls, kwargs):
    ...     # Construct chart from generated kwargs
    ...     chart = chart_cls(**kwargs)
    ...     assert chart.ndim == 3

    """
    chart_class = draw_if_strategy(draw, chart_class)
    if ndim is None and dimensionality is not None:
        ndim = dimensionality
    ndim = draw_if_strategy(draw, ndim)
    kwargs_strategy = cached_build_init_kwargs_strategy(chart_class, dim=ndim)
    return draw(kwargs_strategy)


# ============================================================================


# Cache build_init_kwargs_strategy since it's called repeatedly for the same classes
# Strategies are immutable and deterministic, so this is safe
@ft.lru_cache(maxsize=128)
def cached_build_init_kwargs_strategy(
    cls: type[cxc.AbstractChart],  # type: ignore[type-arg]
    *,
    dim: int | None,
) -> st.SearchStrategy:
    """Return cached wrapper around build_init_kwargs_strategy."""
    return build_init_kwargs_strategy(cls, dim=dim)


# -------------------------------------------------------------------


@plum.dispatch.abstract
def build_init_kwargs_strategy(cls: type, /, *, dim: int | None) -> st.SearchStrategy:
    pass


@plum.dispatch(precedence=-1)
def build_init_kwargs_strategy(
    cls: type[cxc.AbstractChart],  # type: ignore[type-arg]
    /,
    *,
    dim: int | None,
) -> st.SearchStrategy:
    """Build a strategy that generates valid __init__ kwargs for a class.

    Parameters
    ----------
    cls : type
        The class to generate kwargs for.
    dim : int | None
        Optional dimensionality constraint for the generated arguments.

    Returns
    -------
    st.SearchStrategy[dict[str, Any]]
        A strategy that generates dictionaries of keyword arguments
        suitable for passing to cls.__init__().

    """
    # Get a dictionary of all the required parameters for cls.__init__
    required_params = get_init_params(cls)

    # If there are no required parameters, return empty dict strategy
    if not required_params:
        # No required parameters
        return st.just({})

    # Build a strategy for each parameter.
    strategies = {}
    for k, param in required_params.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            msg = f"Parameter '{k}' of {cls} has no type annotation"
            raise ValueError(msg)

        # Generate strategy for this parameter's annotation
        # Use cached version for performance
        wrapped_ann = wrap_if_not_inspectable(ann)
        strategies[k] = cached_strategy_for_annotation(wrapped_ann)

    # Combine all parameter strategies into a single kwargs dict strategy
    return st.fixed_dictionaries(strategies)


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cxe.EmbeddedManifold],  # type: ignore[type-arg]
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
    target_dim = ndim if ndim is not None else draw(st.integers(2, 6))

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
        # charts with unresolvable TypeVars
        # Late import to avoid circular import (core.py imports from this module)
        from coordinax_hypothesis._src.charts.core import charts  # noqa: PLC0415

        chart = draw(charts(filter=cxc.AbstractFixedComponentsChart, ndim=factor_dim))
        factors.append(chart)

    # Generate names for each factor
    names = tuple(f"f{i}" for i in range(len(factors)))

    return tuple(factors), names


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cxc.CartesianProductChart],  # type: ignore[type-arg]
    /,
    *,
    dim: int | None,
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
