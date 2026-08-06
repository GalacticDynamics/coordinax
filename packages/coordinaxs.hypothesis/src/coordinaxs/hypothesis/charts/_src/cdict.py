"""Hypothesis strategies for CDict objects.

A CDict is a mapping from component-name strings to quantity-like array values.
This module provides strategies for generating valid CDict objects that match
chart component schemas.

"""

__all__ = ("cdicts",)

from collections.abc import Mapping
from typing import Any, TypeAlias

import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
import plum
import unxt as u
import unxts.hypothesis as ust
from hypothesis.extra.array_api import make_strategies_namespace

import coordinax.charts as cxc

from .domains import FREE, Interval, component_domains
from coordinaxs.hypothesis.utils import (
    CDict,
    Shape,
    draw_if_strategy,
    strip_return_annotation,
)

#: Cap on |value| in each component's canonical unit.
#:
#: Without one, draws span the representable range -- measured 1e-5 to 3e38 m --
#: which is dimensionally fine but useless to any test that compares numbers:
#: at 1.8e19 m a float32 ULP is ~2e12, so agreement to any absolute tolerance is
#: meaningless. Pass ``magnitude=None`` to draw across the full range.
DEFAULT_MAGNITUDE = 1e3

#: How a caller may specify element values: a strategy, a mapping of keyword
#: arguments for `hypothesis.strategies.floats`, or nothing at all.
#:
#: Spelled as a real object rather than a string: these signatures are
#: introspected at runtime by beartype and plum, which resolve a string
#: annotation in the *caller's* namespace, where ``st`` is not bound.
Elements: TypeAlias = st.SearchStrategy[float] | Mapping[str, Any] | None


def _canonical_unit(interval: Interval, dim: Any, /) -> Any:
    """Return the unit the magnitude cap is measured in.

    Returns `None` when the dimension has no SI representative -- exotic
    dimensions such as ``length / time**0.5`` resolve to
    ``PhysicalType('unknown')``, which is not in the registry. The caller then
    caps in the drawn unit instead, which still bounds the magnitude even
    though the bound is not physically canonical.
    """
    if interval.unit is not None:
        return u.unit(interval.unit)
    if dim is None:
        return None
    try:
        return u.unitsystems.si[dim]
    except KeyError:
        return None


def _snap_inward(value: float | None, width: int, /, *, up: bool) -> float | None:
    """Round *value* to a float of *width* bits, moving into the interval.

    Hypothesis rejects bounds that are not exactly representable at the
    requested width, and rounding the wrong way would widen the domain back
    over the singularity it was drawn to avoid -- so lower bounds round up and
    upper bounds round down.
    """
    if value is None:
        return None
    dtype = np.float32 if width == 32 else np.float64
    snapped = float(dtype(value))
    if up and snapped < value:
        snapped = float(np.nextafter(dtype(snapped), dtype(np.inf)))
    elif not up and snapped > value:
        snapped = float(np.nextafter(dtype(snapped), dtype(-np.inf)))
    return snapped


@st.composite
def _component_quantities(
    draw: st.DrawFn,
    *,
    interval: Interval,
    dim: Any,
    unit: Any,
    dtype: Any,
    shape: Shape,
    elements: Elements,
    magnitude: float | tuple[float, float] | None,
) -> Any:
    """One component, bounded by its domain and the magnitude cap."""
    if unit is None:  # dimensionless / no dimension recorded
        return draw(
            ust.quantities(unit=dim, dtype=dtype, shape=shape, elements=elements)
        )

    lo, hi = interval.bounds_in(unit)

    if magnitude is not None:
        floor, cap = (
            (0.0, magnitude) if isinstance(magnitude, int | float) else magnitude
        )
        canon = _canonical_unit(interval, dim)
        if canon is not None:
            floor = float(u.ustrip(unit, u.Q(floor, canon)))
            cap = float(u.ustrip(unit, u.Q(cap, canon)))
        lo = -cap if lo is None else max(lo, -cap)
        hi = cap if hi is None else min(hi, cap)

        # A magnitude *floor* only means something where the coordinate
        # degenerates at zero -- a radius, not a Cartesian offset, for which
        # x = 0 is an ordinary value. Applying it everywhere would carve a hole
        # out of the middle of every free axis.
        if floor > 0 and interval.min == 0.0 and interval.margin > 0:
            lo = floor if lo is None else max(lo, floor)

    width = 32 if dtype is jnp.float32 else 64
    lo = _snap_inward(lo, width, up=True)
    hi = _snap_inward(hi, width, up=False)

    elements = _bounded_elements(elements, lo=lo, hi=hi, width=width)

    return draw(ust.quantities(unit=unit, dtype=dtype, shape=shape, elements=elements))


def _bounded_elements(
    elements: Elements,
    /,
    *,
    lo: float | None,
    hi: float | None,
    width: int,
) -> Elements:
    """Confine *elements* to ``[lo, hi]``, however it was supplied.

    Three cases, and the middle one is why this is worth separating:

    - Nothing given: build the strategy straight from the bounds.
    - Kwargs for `hypothesis.strategies.floats`: intersect the bounds
      *exactly*. No filtering, so no rejection, no `filter_too_much`, and no
      distorted distribution.
    - An opaque strategy: it can only be filtered. Filtering rather than
      overriding means an out-of-domain range fails loudly instead of silently
      yielding coordinates the chart cannot represent.
    """
    if elements is None:
        return st.floats(
            min_value=lo,
            max_value=hi,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            width=width,
        )

    if isinstance(elements, Mapping):
        merged: dict[str, Any] = {str(k): v for k, v in elements.items()}
        if lo is not None:
            merged["min_value"] = (
                lo if merged.get("min_value") is None else max(merged["min_value"], lo)
            )
        if hi is not None:
            merged["max_value"] = (
                hi if merged.get("max_value") is None else min(merged["max_value"], hi)
            )
        return merged

    if lo is None and hi is None:
        return elements

    def _in_domain(v: float) -> bool:
        return (lo is None or v >= lo) and (hi is None or v <= hi)

    return elements.filter(_in_domain)


# Create array API strategies namespace for JAX
xps = make_strategies_namespace(jnp)


@plum.dispatch.abstract
def cdicts(*args: Any, **kwargs: Any) -> CDict:
    """Generate a valid CDict matching chart components and role constraints.

    A CDict is a mapping from component-name strings to quantity-like values,
    constrained by:
    - Keys must exactly match `chart.components`
    - For physical tangent roles (Pos, Vel, PhysAcc), all component values must
      have the same physical dimension (length, length/time, or length/time²)
    - For Point role, component dimensions follow `chart.coord_dimensions`

    This is the abstract dispatcher; see the concrete overloads below for the
    accepted arguments (``chart``, ``dtype``, ``shape``, ``elements``).

    Returns
    -------
    dict[str, Any]
        A mapping from component names to quantity-like values

    Examples
    --------
    >>> import coordinaxs.hypothesis.main as cxst
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc

    Generate CDict for Cartesian chart:

    >>> @given(p=cxst.cdicts(cxc.cart3d))
    ... def test_cdict(p):
    ...     assert set(p.keys()) == {"x", "y", "z"}
    ...     assert all(isinstance(v, u.Q) for v in p.values())

    Generate CDict with chart as a strategy (draws chart first, then builds CDict):

    >>> @given(p=cxst.cdicts(cxst.charts(filter=cxc.Abstract3D)))
    ... def test_any_3d_chart_cdict(p):
    ...     assert len(p) == 3  # 3D charts have 3 components

    This can also be called without specifying a chart strategy, in which case
    it defaults to drawing from all charts:

    >>> @given(p=cxst.cdicts())
    ... def test_any_chart_cdict(p):
    ...     assert isinstance(p, dict)

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
@strip_return_annotation
@st.composite
def cdicts(
    draw: st.DrawFn,
    chart: st.SearchStrategy = st.deferred(lambda: cxc.charts()),  # ty: ignore[unresolved-attribute]
    /,
    **kwargs: Any,
) -> CDict:
    """Draw a CDict from a strategy that generates charts.

    >>> import coordinaxs.hypothesis.main as cxst
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc

    >>> @given(p=cxst.cdicts(cxst.charts(filter=cxc.Abstract3D)))
    ... def test_any_3d_chart_cdict(p):
    ...     assert len(p) == 3  # 3D charts have 3 components

    This can also be called without specifying a chart strategy, in which case
    it defaults to drawing from all charts:

    >>> @given(p=cxst.cdicts())
    ... def test_any_chart_cdict(p):
    ...     assert isinstance(p, dict)

    """
    # Draw chart
    chart = draw(chart)

    # Redispatch to the more specific implementation
    return draw(cdicts(chart, **kwargs))


@plum.dispatch
@strip_return_annotation
@st.composite
def cdicts(
    draw: st.DrawFn,
    chart: cxc.AbstractChart,
    /,
    *,
    dtype: Any | st.SearchStrategy = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] = (),
    elements: Elements = None,
    magnitude: float | tuple[float, float] | None = DEFAULT_MAGNITUDE,
) -> CDict:
    """Generate a valid CDict matching chart components and role constraints.

    Parameters
    ----------
    draw
        The Hypothesis draw function (provided automatically).
    chart
        The chart instance defining the component schema.
    dtype
        Data type for array components (default: ``jnp.float32``).
    shape
        Shape for array components; int, tuple of ints, or a strategy. Default
        is scalar (``shape=()``).
    elements
        Strategy for generating individual float values. If ``None``, one is
        built from the component's domain. If given, it is *intersected* with
        the domain rather than overriding it, so an out-of-domain range fails
        loudly instead of yielding invalid coordinates.
    magnitude
        Bound on ``|value|`` in each component's canonical unit. A scalar is
        the upper cap; a ``(floor, cap)`` pair also keeps radial coordinates
        away from the origin, which is what a test needing well-conditioned
        points wants -- a Jacobian entry scaling like ``1/r`` is unusable at
        ``r = 1e-3``. The floor is ignored for coordinates that do not
        degenerate at zero. `None` removes the bound entirely.

    Returns
    -------
    dict[str, Any]
        A mapping from component names to quantity-like values.

    Examples
    --------
    >>> import coordinaxs.hypothesis.main as cxst
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc

    >>> @given(p=cxst.cdicts(cxc.cart3d))
    ... def test_cdict(p):
    ...     assert set(p.keys()) == {"x", "y", "z"}
    ...     assert all(isinstance(v, u.Q) for v in p.values())

    """
    # Draw shape if it's a strategy
    shape: Shape = draw_if_strategy(draw, shape)

    domains = component_domains(chart)
    data: CDict = {}

    for cname, cdim in zip(chart.components, chart.coord_dimensions, strict=True):
        dim = u.dimension(cdim) if isinstance(cdim, str) and cdim is not None else cdim
        interval = domains.get(cname, FREE)

        # Draw the unit first, then express the domain *in that unit*. Passing
        # the dimension straight to `ust.quantities` would let it pick a unit
        # internally, leaving no way to convert the bounds -- and a bound is
        # only meaningful alongside the unit it is measured in. Drawing the
        # unit here keeps the unit diversity (which exercises unit handling)
        # while making the physical constraint hold whatever is drawn.
        unit = draw(ust.units(dim)) if dim is not None else None

        data[cname] = draw(
            _component_quantities(
                interval=interval,
                dim=dim,
                unit=unit,
                dtype=dtype,
                shape=shape,
                elements=elements,
                magnitude=magnitude,
            )
        )

    return data
