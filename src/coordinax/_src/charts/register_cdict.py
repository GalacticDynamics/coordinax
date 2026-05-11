"""Utility functions for charts."""

__all__ = ()


from jaxtyping import Array, ArrayLike
from typing import cast

import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.api.charts as cxcapi
from coordinax._src.base import AbstractChart
from coordinax._src.custom_types import CDict
from coordinax.internal import QuantityMatrix, UnitsMatrix

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
def cdict(obj: CDict, chart: AbstractChart, /) -> CDict:
    """Return a dictionary as-is.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> d = {"x": 1.0, "y": 2.0}
    >>> cxc.cdict(d, cxc.cart2d)
    {'x': 1.0, 'y': 2.0}

    """
    return chart.check_data(obj, keys=True)


# ===================================================================
# Quantity


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
    >>> cx.cdict(q)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    chart = cxcapi.guess_chart(obj)
    out = cxcapi.cdict(obj, chart)
    return cast("CDict", out)


@plum.dispatch
def cdict(obj: u.AbstractQuantity, keys: tuple[str, ...], /) -> CDict:
    """Extract component dictionary from a Quantity using specified keys.

    Treats the Quantity as a vector with components in the last dimension,
    splitting along that axis according to the provided keys.

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
    >>> cxc.cdict(q, ('x', 'y', 'z'))
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    if obj.shape[-1] != len(keys):
        raise ValueError(
            f"Quantity last dimension {obj.shape[-1]} does not match "
            f"provided keys {len(keys)}."
        )

    return {k: obj[..., i] for i, k in enumerate(keys)}


@plum.dispatch
def cdict(obj: QuantityMatrix, keys: tuple[str, ...], /) -> CDict:
    """Extract component dictionary from a 1D ``QuantityMatrix``.

    This overload supports heterogeneous per-component units by constructing
    one quantity per chart component from the corresponding numeric slice and
    unit in the ``QuantityMatrix``.

    Raises
    ------
    ValueError
        If ``obj`` is not 1D, or if the last dimension does not match the
        number of provided keys.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.internal import QuantityMatrix

    >>> q = QuantityMatrix(jnp.array([1.0, 2.0, 3.0]),
    ...                    unit=("m", "km/s", "rad"))
    >>> cxc.cdict(q, ('x', 'y', 'z'))
    {'x': Q(1., 'm'), 'y': Q(2., 'km / s'), 'z': Q(3., 'rad')}

    """
    if obj.unit.ndim != 1:
        msg = f"QuantityMatrix must be 1D for cdict, got ndim={obj.ndim}."
        raise ValueError(msg)

    if obj.shape[-1] != len(keys):
        msg = (
            f"QuantityMatrix last dimension {obj.shape[-1]} does not match "
            f"provided keys {len(keys)}."
        )
        raise ValueError(msg)

    return {k: u.Q(obj.value[..., i], obj.unit[i]) for i, k in enumerate(keys)}


@plum.dispatch
def cdict(obj: u.AbstractQuantity, chart: AbstractChart, /) -> CDict:
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
    >>> cxc.cdict(q, cxc.cart3d)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    return cxcapi.cdict(obj, chart.components)  # ty: ignore[invalid-return-type]


@plum.dispatch
def cdict(obj: QuantityMatrix, chart: AbstractChart, /) -> CDict:
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
    >>> cxc.cdict(q, cxc.cart3d)
    {'x': Q(1., 'm'), 'y': Q(2., 'km / s'), 'z': Q(3., 'rad')}

    """
    return cxcapi.cdict(obj, chart.components)  # ty: ignore[invalid-return-type]


# ===================================================================
# Array-like


@plum.dispatch
def cdict(obj: ArrayLike, keys: tuple[str, ...], /) -> CDict:
    """Extract component dictionary from an array.

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
    >>> cx.cdict(arr, ('x', 'y', 'z'))
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    """
    obj = jnp.asarray(obj)
    if obj.shape[-1] != len(keys):
        msg = (
            f"Array last dimension {obj.shape[-1]} does not match "
            f"provided keys {len(keys)}."
        )
        raise ValueError(msg)
    return {k: obj[..., i] for i, k in enumerate(keys)}


@plum.dispatch
def cdict(obj: ArrayLike, chart: AbstractChart, /) -> CDict:
    """Extract component dictionary from an array.

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
    >>> cx.cdict(arr, cx.cart3d)
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    """
    return cxcapi.cdict(obj, chart.components)  # ty: ignore[invalid-return-type]


@plum.dispatch
def cdict(
    obj: ArrayLike,
    unit: u.AbstractUnit | str | UnitsMatrix | None,
    keys: tuple[str, ...],
    /,
) -> CDict:
    """Extract component dictionary from an array.

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
    >>> cx.cdict(arr, "m", ('x', 'y', 'z'))
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    theunit = u.unit(unit) if unit is not None else None
    vals = jnp.asarray(obj) if theunit is None else u.Q(jnp.asarray(obj), theunit)
    return {k: vals[..., i] for i, k in enumerate(keys)}


@plum.dispatch
def cdict(
    obj: ArrayLike,
    unit: u.AbstractUnit | str | UnitsMatrix | None,
    chart: AbstractChart,
    /,
) -> CDict:
    """Extract component dictionary from an array.

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
    >>> cx.cdict(arr, "m", cx.cart3d)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    return cxcapi.cdict(obj, unit, chart.components)  # ty: ignore[invalid-return-type]


@plum.dispatch
def cdict(obj: ArrayLike, unit: u.AbstractUnit | str | UnitsMatrix, /) -> CDict:
    """Extract component dictionary from an array.

    Treats the array as a Cartesian vector with components in the last
    dimension. The appropriate Cartesian chart is determined from the last
    dimension of the quantity.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import jax.numpy as jnp
    >>> arr = jnp.array([1.0, 2.0, 3.0])
    >>> cxc.cdict(arr, "m")
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    q = u.Q(jnp.asarray(obj), cast("u.AbstractUnit", u.unit(unit)))
    out = cxcapi.cdict(q)
    return cast("CDict", out)


@plum.dispatch
def cdict(obj: ArrayLike, usys: u.AbstractUnitSystem, chart: AbstractChart, /) -> CDict:
    """Extract component dictionary from an array.

    Raises
    ------
    ValueError
        If the last dimension of the quantity doesn't match a known Cartesian
        chart (0D, 1D, 2D, or 3D).

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import unxt as u
    >>> import jax.numpy as jnp
    >>> arr = jnp.array([1.0, 2.0, 3.0])
    >>> cx.cdict(arr, u.unitsystems.si, cx.cart3d)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    obj: Array = jnp.asarray(obj)  # TODO: asanyarray

    if obj.shape[-1] != len(chart.components):
        msg = (
            f"Array last dimension {obj.shape[-1]} does not match "
            f"chart with {len(chart.components)} components."
        )
        raise ValueError(msg)
    return {
        k: u.Q(obj[..., i], usys[d])
        for i, (k, d) in enumerate(
            zip(chart.components, chart.coord_dimensions, strict=False)
        )
    }
