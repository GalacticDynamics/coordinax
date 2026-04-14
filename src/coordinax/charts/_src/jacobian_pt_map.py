r"""Jacobian of the point transition map between coordinate charts.

This module defines ``jacobian_pt_map``, which computes the Jacobian matrix of
the chart transition map (point-map) between two charts, evaluated at a given
base point.

Mathematical background:

Given charts $C_1$ and $C_2$ with a transition map $\tau: C_1 \to C_2$, the
Jacobian at a base point $p$ (expressed in $C_1$ coordinates) is

$$ J^j{}_i(p) = \frac{\\partial \tau^j}{\partial q^i}\bigg|_p $$

where $q^i$ are the $C_1$ coordinates and $\tau^j$ are the $C_2$ coordinates.
The result is a 2-D {class}`~coordinax.internal.QuantityMatrix` of shape
$(n_\\mathrm{out},\\, n_\\mathrm{in})$ whose $(j, i)$ element carries units

$$ \mathrm{unit}(J^j{}_i) = \frac{\mathrm{unit}(\tau^j)}{\mathrm{unit}(q^i)} $$

For example, for Cart3D $\to$ Spherical3D (units $\mathrm{m, m, m} \to
\mathrm{m, rad, rad}$):

- $J^r{}_x$: $\mathrm{m}/\mathrm{m}$ (dimensionless)
- $J^\theta{}_x$: $\mathrm{rad}/\mathrm{m}$
- $J^\phi{}_y$: $\mathrm{rad}/\mathrm{m}$

Examples
--------
>>> import coordinax.charts as cxc
>>> import unxt as u
>>> J = cxc.jacobian_pt_map(
...     {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
...     cxc.cart3d, cxc.sph3d,
... )
>>> J.value.shape
(3, 3)

"""

__all__ = ("jacobian_pt_map",)

from collections.abc import Callable
from jaxtyping import Array
from typing import Any, cast

import jax
import jax.numpy as jnp
import plum

import quaxed.numpy as qnp
import unxt as u

import coordinax.api.charts as cxcapi
from .base import AbstractChart
from .custom_types import CDict
from .d2 import Cart2D, Polar2D
from coordinax.internal import (
    QuantityMatrix,
    UnitsMatrix,
    pack_to_qmatrix,
)
from coordinax.internal.custom_types import OptUSys

DMLS: u.AbstractUnit = cast("u.AbstractUnit", u.unit(""))


# ===================================================================
# Partial function
# NOTE: jitting this makes a MUCH faster function than passing `at`, since it
# closes over the fixed args/kwargs and constructs the point-map function once.
# The returned function is fast to call since it only takes `at` as an argument,
# which is what we want to jit-compile for repeated calls at different base
# points.


@plum.dispatch
def jacobian_pt_map(at: None, /, *fixed_args: Any, **fixed_kw: Any) -> Any:
    """Higher-order function for fixed-arg Jacobian point map.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> map = cxc.jacobian_pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> map(at)
    QuantityMatrix(
        [[ 1.,  0.,  0.],
         [-0., -0., -1.],
         [ 0.,  1.,  0.]],
        '((, , ), (rad / m, rad / m, rad / m), (rad / m, rad / m, rad / m))'
    )

    >>> import jax
    >>> J = jax.vmap(map)(jax.tree.map(lambda x: x[None], at))
    >>> J.shape
    (1, 3, 3)

    """
    return lambda at, *args, **kw: cxcapi.jacobian_pt_map(
        at, *fixed_args, *args, **fixed_kw, **kw
    )


@plum.dispatch
def jacobian_pt_map(
    from_chart: AbstractChart, to_chart: AbstractChart, /, *, usys: u.AbstractUnitSystem
) -> Callable[[object], Any]:
    """Higher-order function for fixed-arg Jacobian point map.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> map = cxc.jacobian_pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> map(at)
    QuantityMatrix(
        [[ 1.,  0.,  0.],
         [-0., -0., -1.],
         [ 0.,  1.,  0.]],
        '((, , ), (rad / m, rad / m, rad / m), (rad / m, rad / m, rad / m))'
    )

    >>> import jax
    >>> J = jax.vmap(map)(jax.tree.map(lambda x: x[None], at))
    >>> J.shape
    (1, 3, 3)

    """
    return lambda at: cxcapi.jacobian_pt_map(at, from_chart, to_chart, usys=usys)


# ===================================================================
# Generic Dispatches


@plum.dispatch
def jacobian_pt_map(
    at: Array,
    from_chart: AbstractChart,
    to_chart: AbstractChart,
    /,
    *,
    usys: u.AbstractUnitSystem,
) -> Array:
    r"""Compute the Jacobian at a plain-array base point.

    Treats *at* as a flat numeric array whose elements are the ``from_chart``
    coordinates expressed. Returns the JAX array Jacobian $J^j{}_i = \partial
    \tau^j / \partial q^i$, without unit annotation.

    Parameters
    ----------
    at
        Base point in ``from_chart`` coordinates, shape ``(n_in,)``.  Each
        element should be in the ``usys`` unit for the corresponding coordinate
        dimension.
    from_chart
        Source coordinate chart.
    to_chart
        Target coordinate chart.
    usys
        Unit system used to build the point-map function via ``pt_map(None,
        from_chart, to_chart, usys=usys)``.  **Required.**

    Returns
    -------
    Array
        Jacobian array of shape ``(n_out, n_in)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> at = jnp.array([1.0, 0.0, 0.0])
    >>> jac_fn = cxc.jacobian_pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> J = jac_fn(at)
    >>> J
    Array([[ 1.,  0.,  0.],
           [-0., -0., -1.],
           [ 0.,  1.,  0.]], dtype=float64)

    >>> import jax
    >>> J = jax.vmap(jac_fn)(at[None])
    >>> J.shape
    (1, 3, 3)

    """
    # Prepare the Jacobian of the point-map function w.r.t. the base point.
    # Close over the args/kwargs to construct the point-map function once.
    pt_map_fn = cxcapi.pt_map(None, from_chart, to_chart, usys=usys)
    jac_pt_map_fn = jax.jacfwd(pt_map_fn)

    return jac_pt_map_fn(at)  # Compute Jacobian as array


def _repack_q_from_jac(jac_qq: QuantityMatrix, /) -> QuantityMatrix:
    r"""Rebuild a 2-D ``QuantityMatrix`` Jacobian from the raw ``jax.jacfwd`` output.

    When ``jax.jacfwd`` differentiates a function that maps a 1-D
    ``QuantityMatrix`` of shape ``(n_in,)`` to a 1-D ``QuantityMatrix`` of
    shape ``(n_out,)``, the result is a 2-D ``QuantityMatrix`` of shape
    ``(n_out, n_in)`` whose ``.value`` is *itself* a 1-D ``QuantityMatrix``
    carrying the input units (one per column), and whose ``.unit`` is a 1-D
    ``UnitsMatrix`` carrying the output units (one per row).

    This helper extracts both unit layers to build the correct 2-D
    ``UnitsMatrix``:  ``units[j, i] = uto_[j] / ufrom_[i]``.

    """
    ufrom_, uto_ = jac_qq.value.unit, jac_qq.unit  # ty: ignore[unresolved-attribute]
    units = UnitsMatrix(tuple(tuple(tj / fi for fi in ufrom_) for tj in uto_))
    return QuantityMatrix(jac_qq.value.value, units)  # ty: ignore[unresolved-attribute]


@plum.dispatch
def jacobian_pt_map(
    at: CDict,
    from_chart: AbstractChart,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> Array | QuantityMatrix:
    r"""Compute the Jacobian at a coordinate-dictionary base point.

    The primary dict-input dispatch.  Branches on whether the values of *at*
    carry physical units:

    **Array-valued branch** (no units in any value)
        Stacks the dict values into a plain array via ``jnp.stack``, then
        forwards to ``jacobian_pt_map(at_arr, from_chart, to_chart, usys=usys)``
        which requires *usys*.  For chart pairs without an analytical
        ``Array`` dispatch this means *usys* must be provided.

    **Quantity-valued branch** (at least one value carries a unit)
        Packs *at* into a 1-D ``QuantityMatrix`` via
        ``pack_to_qmatrix(at, keys=from_chart.components)``, casts to
        ``float``, then computes ``J_qq = jax.jacfwd(pt_map_fn)(at_in)``.
        Because ``jacfwd`` applied to a ``QuantityMatrix``-in /
        ``QuantityMatrix``-out function yields a nested ``QuantityMatrix``,
        ``_repack_q_from_jac`` is called to extract the correct 2-D unit
        structure.

    Returns
    -------
    Array | QuantityMatrix
        Plain array when *at* is array-valued; ``QuantityMatrix`` of shape
        ``(n_out, n_in)`` with per-element units otherwise.

    Raises
    ------
    ValueError
        If *at* keys do not match ``from_chart.components`` (via
        ``check_data``).
    plum.NotFoundLookupError
        If the array-valued branch cannot resolve a dispatch (e.g.
        generic chart pair with ``usys=None``).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Quantity-valued dict (no usys needed):

    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> J = cxc.jacobian_pt_map(at, cxc.cart3d, cxc.sph3d)
    >>> J.value.shape
    (3, 3)

    Plain-array dict (usys required):

    >>> import jax.numpy as jnp
    >>> at_arr = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> J2 = cxc.jacobian_pt_map(at_arr, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> J2.shape
    (3, 3)

    """
    # Determine whether the input is array-valued or quantity-valued.  If it's
    # array-valued, we can skip the packing and unit handling and directly
    # compute the Jacobian as an array.
    at = from_chart.check_data(at, keys=True)
    is_array = not any(hasattr(v, "unit") for v in at.values())
    # is_array &= not all(dim is None for dim in from_chart.coord_dimensions)
    if is_array:
        at_arr = jnp.stack([at[k] for k in from_chart.components], axis=-1)
        return cxcapi.jacobian_pt_map(at_arr, from_chart, to_chart, usys=usys)  # ty: ignore[invalid-return-type]

    # It's Quantity-valued.
    # Prepare the Jacobian of the point-map function w.r.t. the base point.
    # Close over the args/kwargs to construct the point-map function once.
    pt_map_fn = cxcapi.pt_map(None, from_chart, to_chart, usys=usys)
    jac_pt_map_fn = jax.jacfwd(pt_map_fn)

    # Pack the input CDict to a QMatrix
    at_in = pack_to_qmatrix(at, keys=from_chart.components)
    at_in = at_in.astype(float)

    # Compute Jacobian as QMatrix
    J_qq = jac_pt_map_fn(at_in)
    return _repack_q_from_jac(J_qq)


# ===================================================================
# Cart2D -> Polar2D


@plum.dispatch
def jacobian_pt_map(
    at: Array, from_chart: Cart2D, to_chart: Polar2D, /, *, usys: OptUSys = None
) -> Array:
    r"""Compute the Jacobian of the transition function between two charts.

    $$
    J = \frac{\partial(r,\theta)}{\partial(x,y)}
      = ( x/r & y/r \ -y/r^2 & x/r^2 )
      = ( \cos\theta & \sin\theta \ -\sin\theta/r & \cos\theta/r )
    $$

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> x = jnp.array([1.0, 1.0])
    >>> cxc.jacobian_pt_map(cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si)(x)
    Array([[ 0.70710678,  0.70710678],
           [-0.5       ,  0.5       ]], dtype=float64)

    """
    x, y = at[..., 0], at[..., 1]
    r2 = x**2 + y**2
    r = jnp.sqrt(r2)
    return jnp.array([[x, y], [-y, x]]) / jnp.array([[r], [r2]])


@plum.dispatch
def jacobian_pt_map(
    at: u.AbstractQuantity,
    from_chart: Cart2D,
    to_chart: Polar2D,
    /,
    *,
    usys: OptUSys = None,
) -> QuantityMatrix:
    r"""Compute the Jacobian of the transition function between two charts.

    $$
    J = \frac{\partial(r,\theta)}{\partial(x,y)}
      = ( x/r & y/r \ -y/r^2 & x/r^2 )
      = ( \cos\theta & \sin\theta \ -\sin\theta/r & \cos\theta/r )
    $$

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> x = u.Q(jnp.array([1.0, 1.0]), "m")
    >>> cxc.jacobian_pt_map(cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si)(x)
    QuantityMatrix([[ 0.70710678,  0.70710678],
                    [-0.5       ,  0.5       ]], '((, ), (rad / m, rad / m))')

    """
    x, y = at[..., 0], at[..., 1]
    r2 = x**2 + y**2
    r = qnp.sqrt(r2)
    x0, x1 = x / r, y / r
    x2, x3 = -y / r2, x / r2
    # Astropy treats rad as dimensionless, so x2.unit == 1/m rather than
    # the correct rad/m.  Force the right unit explicitly.
    rad_per_len = u.unit("rad") / x.unit
    return QuantityMatrix(
        jnp.array([[x0.value, x1.value], [x2.value, x3.value]]),
        unit=((x0.unit, x1.unit), (rad_per_len, rad_per_len)),
    )
