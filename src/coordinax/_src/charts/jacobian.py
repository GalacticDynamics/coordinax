r"""Jacobian of the point transition map between coordinate charts.

This module defines ``jac_pt_map``, which computes the Jacobian matrix of
the chart transition map (point-map) between two charts, evaluated at a given
base point.

Mathematical background:

Given charts $C_1$ and $C_2$ with a transition map $\tau: C_1 \to C_2$, the
Jacobian at a base point $p$ (expressed in $C_1$ coordinates) is

$$ J^j{}_i(p) = \frac{\partial \tau^j}{\partial q^i}\bigg|_p $$

where $q^i$ are the $C_1$ coordinates and $\tau^j$ are the $C_2$ coordinates.
The result is a 2-D {class}`~unxts.linalg.QuantityMatrix` of shape
$(n_\mathrm{out},\, n_\mathrm{in})$ whose $(j, i)$ element carries units

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
>>> J = cxc.jac_pt_map(
...     {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
...     cxc.cart3d, cxc.sph3d,
... )
>>> J.value.shape
(3, 3)

"""

__all__ = ("jac_pt_map",)

from collections.abc import Callable
from jaxtyping import Array
from typing import Any, Final, cast

import jax
import jax.numpy as jnp
import plum
from zeroth import zeroth

import unxt as u
import unxts.linalg as ul

import coordinaxs.api.charts as cxcapi
from .d2 import Cart2D, Polar2D
from coordinax._src.base import AbstractChart
from coordinax._src.custom_types import OptUSys
from coordinax.internal import tree_cast_int_bool_to_float
from coordinaxs.api.custom_types import CDict

# ===================================================================
# Partial function
# NOTE: jitting this makes a MUCH faster function than passing `at`, since it
# closes over the fixed args/kwargs and constructs the point-map function once.
# The returned function is fast to call since it only takes `at` as an argument,
# which is what we want to jit-compile for repeated calls at different base
# points.


@plum.dispatch
def jac_pt_map(at: None, /, *fixed_args: Any, **fixed_kw: Any) -> Any:
    """Higher-order function for fixed-arg Jacobian point map.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> map = cxc.jac_pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> map(at)
    QM(
        [[ 1.,  0.,  0.],
         [ 0.,  0., -1.],
         [ 0.,  1.,  0.]],
        '((, , ), (rad / m, rad / m, rad / m), (rad / m, rad / m, rad / m))'
    )

    >>> import jax
    >>> J = jax.vmap(map)(jax.tree.map(lambda x: x[None], at))
    >>> J.shape
    (1, 3, 3)

    """
    return lambda at, *args, **kw: cxcapi.jac_pt_map(
        at, *fixed_args, *args, **fixed_kw, **kw
    )


@plum.dispatch
def jac_pt_map(
    from_chart: AbstractChart, to_chart: AbstractChart, /, *, usys: OptUSys
) -> Callable[[object], Any]:
    """Higher-order function for fixed-arg Jacobian point map.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> map = cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> map(at)
    QM(
        [[ 1.,  0.,  0.],
         [ 0.,  0., -1.],
         [ 0.,  1.,  0.]],
        '((, , ), (rad / m, rad / m, rad / m), (rad / m, rad / m, rad / m))'
    )

    >>> import jax
    >>> J = jax.vmap(map)(jax.tree.map(lambda x: x[None], at))
    >>> J.shape
    (1, 3, 3)

    """
    return lambda at: cxcapi.jac_pt_map(at, from_chart, to_chart, usys=usys)


# ===================================================================
# Generic Dispatches


@plum.dispatch
def jac_pt_map(
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

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> jac_fn = cxc.jac_pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)

    >>> at = jnp.array([1, 0, 0])
    >>> jac_fn(at)
    Array([[ 1.,  0.,  0.],
           [ 0.,  0., -1.],
           [ 0.,  1.,  0.]], dtype=float64)

    >>> import jax
    >>> J = jax.vmap(jac_fn)(at[None])
    >>> J.shape
    (1, 3, 3)

    """
    # jacfwd requires real floating inputs; promote only integer/bool.
    at = tree_cast_int_bool_to_float(jnp.asarray(at))

    # Prepare the Jacobian of the point-map function w.r.t. the base point.
    # Close over the args/kwargs to construct the point-map function once.
    pt_map_fn = cxcapi.pt_map(None, from_chart, to_chart, usys=usys)
    jac_pt_map_fn = jax.jacfwd(pt_map_fn)

    return jac_pt_map_fn(at)  # Compute Jacobian as array


def _repack_q_from_jac(jac_qq: ul.QuantityMatrix, /) -> ul.QuantityMatrix:
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
    ufrom_t, uto_t = ufrom_.to_tuple(), uto_.to_tuple()
    units = ul.UnitsMatrix(tuple(tuple(uj / ui for ui in ufrom_t) for uj in uto_t))
    return ul.QuantityMatrix(jac_qq.value.value, units)  # ty: ignore[unresolved-attribute]


def _jac_via_autodiff(
    at: CDict, from_chart: AbstractChart, to_chart: AbstractChart, usys: OptUSys, /
) -> ul.QuantityMatrix:
    """Differentiate the transition map at a unitful point.

    The general route, for any chart pair with no closed form registered.
    """
    # Close over the args/kwargs to construct the point-map function once.
    pt_map_fn = cxcapi.pt_map(None, from_chart, to_chart, usys=usys)
    jac_pt_map_fn = jax.jacfwd(pt_map_fn)

    at_in = tree_cast_int_bool_to_float(cxcapi.carray(at, from_chart.components))
    return _repack_q_from_jac(jac_pt_map_fn(at_in))


#: Sentinel for "this point is not batched", so an unbatched call is not
#: confused with a Jacobian that happens to be `None`.
_UNBATCHED: Final = object()


def _jac_over_batch(
    at: CDict, from_chart: AbstractChart, to_chart: AbstractChart, usys: OptUSys, /
) -> Any:
    """Map the pointwise Jacobian over any leading batch axes.

    A chart map is pointwise, so a batch of points is a batch of independent
    Jacobians. Differentiating the batch as one function would instead give the
    ``(*batch, n_out, *batch, n_in)`` Jacobian of ``R^(N*n) -> R^(N*n)``:
    correct for *that* function, but block-diagonal with every off-diagonal
    block identically zero, and O(N^2) to hold.

    `jnp.vectorize` maps this in one call but coerces inputs with ``asarray``,
    so it cannot carry the Quantity route; timings were within noise. The shape
    is static, so the unbatched path costs nothing here.

    Returns `_UNBATCHED` when *at* is a single point.
    """
    batch = jnp.shape(zeroth(at.values()))
    if not batch:
        return _UNBATCHED

    def one(at_i: CDict) -> Any:
        return cxcapi.jac_pt_map(at_i, from_chart, to_chart, usys=usys)

    for _ in batch:
        one = jax.vmap(one)
    return one(at)


@plum.dispatch
def jac_pt_map(
    at: CDict,
    from_chart: AbstractChart,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> Array | ul.QuantityMatrix:
    r"""Compute the Jacobian at a coordinate-dictionary base point.

    The primary dict-input dispatch.  Branches on whether the values of *at*
    carry physical units:

    **Array-valued branch** (no units in any value)
        Stacks the dict values into a plain array via ``jnp.stack``, then
        forwards to ``jac_pt_map(at_arr, from_chart, to_chart, usys=usys)``
        which requires *usys*.  For chart pairs without an analytical
        ``Array`` dispatch this means *usys* must be provided.

    **Quantity-valued branch** (at least one value carries a unit)
        Packs *at* into a 1-D ``QuantityMatrix`` via
        ``carray(at, from_chart.components)``, promotes any
        integer or boolean leaves to the default floating-point dtype (other
        dtypes, including complex, are left unchanged and will raise a
        ``TypeError`` from ``jax.jacfwd`` if passed), then computes
        ``J_qq = jax.jacfwd(pt_map_fn)(at_in)``.
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
    >>> J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
    >>> J.value.shape
    (3, 3)

    Plain-array dict (usys required):

    >>> import jax.numpy as jnp
    >>> at_arr = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> J2 = cxc.jac_pt_map(at_arr, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    >>> J2.shape
    (3, 3)

    """
    at = from_chart.check_data(at, keys=True)

    batched = _jac_over_batch(at, from_chart, to_chart, usys)
    if batched is not _UNBATCHED:
        return batched

    # Determine whether the input is array-valued or quantity-valued.  If it's
    # array-valued, we can skip the packing and unit handling and directly
    # compute the Jacobian as an array.
    is_array = not any(hasattr(v, "unit") for v in at.values())
    if is_array:
        at_arr = jnp.stack([at[k] for k in from_chart.components], axis=-1)
        return cxcapi.jac_pt_map(at_arr, from_chart, to_chart, usys=usys)  # ty: ignore[invalid-return-type]

    return _jac_via_autodiff(at, from_chart, to_chart, usys)


# ===================================================================
# Cart2D -> Polar2D


@plum.dispatch
def jac_pt_map(
    at: CDict, from_chart: Cart2D, to_chart: Polar2D, /, *, usys: OptUSys = None
) -> Array | ul.QuantityMatrix:
    """Route a coordinate dict to the closed-form Jacobian below.

    The generic `CDict` dispatch sends a *unitful* point through
    `jax.jacfwd`, which costs a trace per call and ignores the closed form
    sitting next to it -- 3589us against 595us for the same point with bare
    arrays, which do route here. Dispatching on the chart pair lets `plum`
    pick the analytic method for both input kinds.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(1.0, "m")}
    >>> cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)
    QM([[ 0.70710678,  0.70710678],
        [-0.5       ,  0.5       ]], '((, ), (rad / m, rad / m))')

    """
    at = from_chart.check_data(at, keys=True)

    batched = _jac_over_batch(at, from_chart, to_chart, usys)
    if batched is not _UNBATCHED:
        return batched

    keys = from_chart.components
    units = [u.unit_of(at[k]) for k in keys]

    # Bare arrays already reach the closed form through the generic dispatch;
    # reproduce that route rather than changing it.
    if all(unit is None for unit in units):
        at_arr = jnp.stack([at[k] for k in keys], axis=-1)
        return cxcapi.jac_pt_map(at_arr, from_chart, to_chart, usys=usys)  # ty: ignore[invalid-return-type]

    # Only pack when the components already agree on a unit. Converting them
    # to a common one would be arithmetically fine but would re-label the
    # output: `jacfwd` gives a `km / m` entry where packing gives a
    # dimensionless one. Same Jacobian, different presentation, so leave a
    # mixed-unit point on the route it takes today.
    if any(unit != units[0] for unit in units):
        return _jac_via_autodiff(at, from_chart, to_chart, usys)

    packed = tree_cast_int_bool_to_float(
        u.Quantity(
            jnp.stack([u.ustrip(units[0], at[k]) for k in keys], axis=-1), units[0]
        )
    )
    return cxcapi.jac_pt_map(packed, from_chart, to_chart, usys=usys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def jac_pt_map(
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
    >>> cxc.jac_pt_map(cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si)(x)
    Array([[ 0.70710678,  0.70710678],
           [-0.5       ,  0.5       ]], dtype=float64)

    """
    x, y = at[..., 0], at[..., 1]
    r2 = x**2 + y**2
    r = jnp.sqrt(r2)
    return jnp.array([[x, y], [-y, x]]) / jnp.array([[r], [r2]])


@plum.dispatch
def jac_pt_map(
    at: u.AbstractQuantity,
    from_chart: Cart2D,
    to_chart: Polar2D,
    /,
    *,
    usys: OptUSys = None,
) -> ul.QuantityMatrix:
    r"""Compute the Jacobian of the transition function between two charts.

    $$
    J = \frac{\partial(r,\theta)}{\partial(x,y)}
      = ( x/r & y/r \ -y/r^2 & x/r^2 )
      = ( \cos\theta & \sin\theta \ -\sin\theta/r & \cos\theta/r )
    $$

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> x = u.Q(jnp.array([1.0, 1.0]), "m")
    >>> cxc.jac_pt_map(cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si)(x)
    QM([[ 0.70710678,  0.70710678],
        [-0.5       ,  0.5       ]], '((, ), (rad / m, rad / m))')

    """
    # Resolve the unit once and do the arithmetic on raw arrays: every
    # primitive on a `Quantity` operand costs a `quax` trace, and this body
    # ends up taking `.value` anyway. Measured 2344us -> 496us on a scalar.
    unit: Any = u.unit_of(at)
    v = cast("Array", u.ustrip(unit, at))
    x, y = v[..., 0], v[..., 1]
    r2 = x**2 + y**2
    r = jnp.sqrt(r2)
    # The rows carry different units: the radial row is a ratio of lengths,
    # the angular row an angle per length. Astropy treats rad as
    # dimensionless, so `rad / m` has to be spelled out rather than derived.
    dimensionless = unit / unit
    rad_per_len = u.unit("rad") / unit
    return ul.QuantityMatrix(
        jnp.array([[x / r, y / r], [-y / r2, x / r2]]),
        unit=((dimensionless, dimensionless), (rad_per_len, rad_per_len)),
    )
