"""Utility functions for charts."""

__all__ = ()


from jaxtyping import ArrayLike

import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.api.charts as api
from .base import AbstractChart
from .custom_types import CDict
from coordinax.internal import QuantityMatrix

# ===================================================================
# CDict


@plum.dispatch
def cdict(obj: CDict, /) -> CDict:
    """Return a dictionary as-is.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> d = {"x": 1.0, "y": 2.0}
    >>> cx.cdict(d)
    {'x': 1.0, 'y': 2.0}

    """
    return obj


@plum.dispatch
def cdict(chart: AbstractChart, obj: ArrayLike, /) -> CDict:  # type: ignore[type-arg]
    """Extract component dictionary from an array.

    Treats the array as a Cartesian vector with components in the last
    dimension. The appropriate Cartesian chart is determined from the last
    dimension of the quantity.

    Raises
    ------
    ValueError
        If the last dimension of the quantity doesn't match a known Cartesian
        chart (0D, 1D, 2D, or 3D).

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import jax.numpy as jnp
    >>> arr = jnp.array([1.0, 2.0, 3.0])
    >>> d = cx.cdict(cx.cart3d, arr)
    >>> d
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    """
    obj = jnp.asarray(obj)
    if obj.shape[-1] != len(chart.components):
        msg = (
            f"Array last dimension {obj.shape[-1]} does not match "
            f"chart with {len(chart.components)} components."
        )
        raise ValueError(msg)
    return {k: obj[..., i] for i, k in enumerate(chart.components)}


@plum.dispatch
def cdict(obj: u.AbstractQuantity, /) -> CDict:
    """Extract component dictionary from a Quantity.

    Treats the Quantity as a Cartesian vector with components in the last
    dimension. The appropriate Cartesian chart is determined from the last
    dimension of the quantity.

    Raises
    ------
    ValueError
        If the last dimension of the quantity doesn't match a known Cartesian
        chart (0D, 1D, 2D, or 3D).

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import unxt as u
    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> d = cx.cdict(q)
    >>> d
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    chart = api.guess_chart(obj)
    return api.cdict(chart, obj)


@plum.dispatch
def cdict(chart: AbstractChart, obj: u.AbstractQuantity, /) -> CDict:  # type: ignore[type-arg]
    """Extract component dictionary from a Quantity.

    Treats the Quantity as a vector with components in the last dimension,
    splitting along that axis according to the chart's component names.

    This function requires that:
    1. The last dimension of the quantity matches the number of chart components
    2. The chart has homogeneous coordinate dimensions (all components have the
       same physical dimension, like Cartesian charts)

    Raises
    ------
    ValueError
        If the last dimension of the quantity doesn't match the chart's
        component count, or if dimensions don't match.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> d = cxc.cdict(cxc.cart3d, q)
    >>> d
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    if obj.shape[-1] != len(chart.components):
        msg = (
            f"Quantity last dimension {obj.shape[-1]} does not match "
            f"chart with {len(chart.components)} components."
        )
        raise ValueError(msg)

    return {k: obj[..., i] for i, k in enumerate(chart.components)}


@plum.dispatch
def cdict(chart: AbstractChart, obj: QuantityMatrix, /) -> CDict:  # type: ignore[type-arg]
    """Extract component dictionary from a 1D ``QuantityMatrix``.

    This overload supports heterogeneous per-component units by constructing
    one quantity per chart component from the corresponding numeric slice and
    unit in the ``QuantityMatrix``.

    Raises
    ------
    ValueError
        If ``obj`` is not 1D, or if the last dimension does not match the
        chart component count.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> from coordinax.internal import QuantityMatrix

    >>> q = QuantityMatrix(
    ...     jnp.array([1.0, 2.0, 3.0]),
    ...     unit=(u.unit("m"), u.unit("km/s"), u.unit("rad")),
    ... )
    >>> d = cxc.cdict(cxc.cart3d, q)
    >>> d
    {'x': Q(1., 'm'), 'y': Q(2., 'km / s'), 'z': Q(3., 'rad')}

    """
    if obj.ndim != 1:
        msg = f"QuantityMatrix must be 1D for cdict, got ndim={obj.ndim}."
        raise ValueError(msg)

    if obj.shape[-1] != len(chart.components):
        msg = (
            f"QuantityMatrix last dimension {obj.shape[-1]} does not match "
            f"chart with {len(chart.components)} components."
        )
        raise ValueError(msg)

    return {
        k: u.Q(obj.value[..., i], obj.unit[i]) for i, k in enumerate(chart.components)
    }
