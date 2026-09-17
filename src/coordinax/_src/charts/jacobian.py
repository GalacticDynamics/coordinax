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

import functools as ft
import operator

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
from .d3 import (
    Cart3D,
    Cylindrical3D,
    LonCosLatSpherical3D,
    LonLatSpherical3D,
    Spherical3D,
)
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
         [-0.,  1.,  0.]],
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
         [-0.,  1.,  0.]],
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

    This is the fallback: a chart pair with a closed-form Jacobian registered
    is served by that instead. `Cylindrical3D -> Spherical3D` has none, so it
    comes here.

    >>> jac_fn = cxc.jac_pt_map(None, cxc.cyl3d, cxc.sph3d, usys=u.unitsystems.si)

    >>> at = jnp.array([1.0, 0.0, 0.0])
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
    if (type(from_chart), type(to_chart)) in _CLOSED_FORM_PAIRS:
        return _jac_from_dict_via_closed_form(at, from_chart, to_chart, usys)  # ty: ignore[invalid-return-type]

    is_array = not any(hasattr(v, "unit") for v in at.values())
    if is_array:
        at_arr = jnp.stack([at[k] for k in from_chart.components], axis=-1)
        return cxcapi.jac_pt_map(at_arr, from_chart, to_chart, usys=usys)  # ty: ignore[invalid-return-type]

    return _jac_via_autodiff(at, from_chart, to_chart, usys)


def _real_float_point(at: Array, /) -> Array:
    """Normalise a bare-array point the way the autodiff route does.

    A closed form will happily evaluate on integers, booleans or complex
    numbers, where `jax.jacfwd` promotes the first two and rejects the last.
    The analytic dispatches are meant to be indistinguishable from
    differentiating the map, so they have to agree about their inputs too.
    """
    at = tree_cast_int_bool_to_float(jnp.asarray(at))
    if jnp.issubdtype(at.dtype, jnp.complexfloating):
        msg = (
            "jacfwd requires real-valued inputs (input dtype that is a "
            f"sub-dtype of np.floating), but got {at.dtype}."
        )
        raise TypeError(msg)
    return at


#: The angle unit a bare value carries when no unit system says otherwise.
RAD: Final = u.unit("rad")
ANGLE: Final = u.dimension("angle")


def _usys_angle_per_rad(usys: OptUSys, /) -> Any:
    """How many of *usys*' angle units make one radian: 1 for rad, 180/pi for deg.

    A bare-array Jacobian carries no units, so an angular row has to be
    expressed in whatever angle unit the caller's values are in. The transition
    map writes its angles through ``usys["angle"]``, so the derivative of one
    is scaled the same way. Getting this wrong is silent: the numbers stay
    plausible and are only off by a constant.

    An angular *output* row is multiplied by this; an angular *input* column is
    divided by it. The name says which way round it goes, because the two are
    indistinguishable at a glance and reciprocal.
    """
    return 1.0 if usys is None else u.uconvert_value(usys["angle"], RAD, 1.0)


# ===================================================================
# Cart2D -> Polar2D


#: Dividing two units costs ~20x a dict lookup and a chart has only a handful,
#: so the quotients are cached across calls. Cached *here* rather than per
#: call: a fresh `lru_cache` each time caches nothing and costs more than the
#: divisions it replaces (14us against 2.7us for nine entries).
_unit_quotient = ft.lru_cache(maxsize=None)(operator.truediv)


#: Chart pairs with a hand-written Jacobian. Populated by `_closed_form` at
#: each definition, so a closed form added without that line is simply not
#: routed to -- rather than routed to and failing.
_CLOSED_FORM_PAIRS: set[tuple[type, type]] = set()


def _closed_form(from_cls: type, to_cls: type, /) -> Callable[[Any], Any]:
    """Mark the decorated `Array` Jacobian as the closed form for a chart pair."""

    def register(fn: Any) -> Any:
        _CLOSED_FORM_PAIRS.add((from_cls, to_cls))
        return fn

    return register


def _jac_from_dict_via_closed_form(
    at: CDict, from_chart: AbstractChart, to_chart: AbstractChart, usys: OptUSys, /
) -> Any:
    """Send a coordinate dict to the closed form registered for its chart pair.

    Called with a checked, unbatched point: the dispatch above has already
    validated the keys and mapped any leading axes.

    The closed forms take bare values, so strip every component to a canonical
    unit -- angles to radians, everything else to the first unit of its kind --
    differentiate there, and put the units back afterwards. Convertible units
    are equivalent, so canonicalising costs a label and nothing else.
    """
    keys = from_chart.components
    units: list[Any] = [u.unit_of(at[k]) for k in keys]

    # No units to strip or restore; *usys* says what the numbers mean.
    if all(unit is None for unit in units):
        at_arr = jnp.stack([at[k] for k in keys], axis=-1)
        return cxcapi.jac_pt_map(at_arr, from_chart, to_chart, usys=usys)
    # A bare component beside a unitful one has no unit to canonicalise to.
    if any(unit is None for unit in units):
        return _jac_via_autodiff(at, from_chart, to_chart, usys)

    # `dimension_of` is a `plum` dispatch, so resolve each component's once.
    dims = [u.dimension_of(unit) for unit in units]

    # One unit per dimension: radians for angles, the first one seen otherwise.
    # Angles are seeded because a closed form emits them whether or not the
    # input had any -- `Cart3D -> Spherical3D` takes three lengths.
    canonical: dict[Any, Any] = {ANGLE: RAD}
    for unit, dim in zip(units, dims, strict=True):
        canonical.setdefault(dim, RAD if dim == ANGLE else unit)

    out_dims = [u.dimension(dim) for dim in to_chart.coord_dimensions]
    targets = [canonical[dim] for dim in dims]
    values = [
        u.ustrip(unit, at[k])
        if target == unit
        else u.uconvert_value(target, unit, u.ustrip(unit, at[k]))
        for k, unit, target in zip(keys, units, targets, strict=True)
    ]
    raw = tree_cast_int_bool_to_float(jnp.stack(values, axis=-1))

    jac = jnp.asarray(cxcapi.jac_pt_map(raw, from_chart, to_chart, usys=None))

    # Column *i* was differentiated with respect to the canonical unit, so
    # rescale it to the caller's and label it to match. A chart has a handful
    # of distinct units, and dividing two costs far more than a dict lookup.
    scale = [
        1.0 if target == unit else u.uconvert_value(target, unit, 1.0)
        for unit, target in zip(units, targets, strict=True)
    ]
    unit_rows = tuple(
        tuple(_unit_quotient(canonical[dim], col) for col in units) for dim in out_dims
    )
    return ul.QuantityMatrix(jac * jnp.asarray(scale, dtype=jac.dtype), unit=unit_rows)


@_closed_form(Cart2D, Polar2D)
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

    as written, in radians per unit length. The result carries no units, so it
    has to mean the same thing `pt_map(..., usys=usys)` does: the angular row
    is scaled by $d\theta_{usys}/d\theta_{rad}$, which is 1 for radians and
    $180/\pi$ for degrees. A `Quantity` point goes to the overload below
    instead, which labels the row `rad / length` and needs no such scaling.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> x = jnp.array([1.0, 1.0])
    >>> cxc.jac_pt_map(cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si)(x)
    Array([[ 0.70710678,  0.70710678],
           [-0.5       ,  0.5       ]], dtype=float64)

    The same point under a degree system: the radial row is unchanged, the
    angular row is the same derivative expressed per degree.

    >>> degrees = u.unitsystem("m", "deg", "kg", "s")
    >>> cxc.jac_pt_map(cxc.cart2d, cxc.polar2d, usys=degrees)(x)
    Array([[  0.70710678,   0.70710678],
           [-28.64788976,  28.64788976]], dtype=float64)

    """
    at = _real_float_point(at)
    x, y = at[..., 0], at[..., 1]
    # `hypot`, and two divisions by `r` rather than one by `r**2`: both squares
    # overflow float32 well inside the coordinate range the charts handle.
    r = jnp.hypot(x, y)
    xr, yr = x / r, y / r
    # The angular row is an angle per length, so it is expressed in the unit
    # system's angle unit -- the same one `pt_map` writes ``theta`` in.
    ang = _usys_angle_per_rad(usys)
    return jnp.array([[xr, yr], [-yr * ang / r, xr * ang / r]])


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
    r = jnp.hypot(x, y)
    xr, yr = x / r, y / r
    # The rows carry different units: the radial row is a ratio of lengths,
    # the angular row an angle per length. Astropy treats rad as
    # dimensionless, so `rad / m` has to be spelled out rather than derived.
    dimensionless = unit / unit
    rad_per_len = u.unit("rad") / unit
    return ul.QuantityMatrix(
        jnp.array([[xr, yr], [-yr / r, xr / r]]),
        unit=((dimensionless, dimensionless), (rad_per_len, rad_per_len)),
    )


# ===================================================================
# Cart3D <-> Cylindrical3D
#
# `jax.jacfwd` builds and evaluates a jaxpr on every eager call; the closed
# forms below do not, and run in roughly a quarter of the time -- measured
# 2.5x to 3.6x through the public dict API on a scalar point. Under `jit`
# the tracing happens once and the gap narrows to a few percent (5.3us
# against 6.0us), so this is an eager-path win.


@_closed_form(Cart3D, Cylindrical3D)
@plum.dispatch
def jac_pt_map(
    at: Array, from_chart: Cart3D, to_chart: Cylindrical3D, /, *, usys: OptUSys = None
) -> Array:
    r"""Compute the Jacobian of ``Cart3D -> Cylindrical3D``.

    $$
    J = \frac{\partial(\rho,\phi,z)}{\partial(x,y,z)}
      = \begin{pmatrix}
          x/\rho & y/\rho & 0 \\
          -y/\rho^2 & x/\rho^2 & 0 \\
          0 & 0 & 1
        \end{pmatrix}
    $$

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> at = jnp.array([1.0, 0.0, 3.0])
    >>> cxc.jac_pt_map(at, cxc.cart3d, cxc.cyl3d, usys=u.unitsystems.si)
    Array([[ 1.,  0.,  0.],
           [-0.,  1.,  0.],
           [ 0.,  0.,  1.]], dtype=float64)

    """
    at = _real_float_point(at)
    x, y = at[..., 0], at[..., 1]
    rho = jnp.hypot(x, y)
    xr, yr = x / rho, y / rho
    ang = _usys_angle_per_rad(usys)
    zero, one = jnp.zeros_like(x), jnp.ones_like(x)
    return jnp.array(
        [[xr, yr, zero], [-yr * ang / rho, xr * ang / rho, zero], [zero, zero, one]]
    )


@plum.dispatch
def jac_pt_map(
    at: u.AbstractQuantity,
    from_chart: Cart3D,
    to_chart: Cylindrical3D,
    /,
    *,
    usys: OptUSys = None,
) -> ul.QuantityMatrix:
    r"""Compute the Jacobian of ``Cart3D -> Cylindrical3D`` at a unitful point.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> at = u.Q(jnp.array([1.0, 0.0, 3.0]), "m")
    >>> cxc.jac_pt_map(at, cxc.cart3d, cxc.cyl3d).unit
    UnitsMatrix("((, , ), (rad / m, rad / m, rad / m), (, , ))")

    """
    unit: Any = u.unit_of(at)
    v = cast("Array", u.ustrip(unit, at))
    x, y = v[..., 0], v[..., 1]
    rho = jnp.hypot(x, y)
    xr, yr = x / rho, y / rho
    zero, one = jnp.zeros_like(x), jnp.ones_like(x)
    dmls, rad_per_len = unit / unit, RAD / unit
    return ul.QuantityMatrix(
        jnp.array([[xr, yr, zero], [-yr / rho, xr / rho, zero], [zero, zero, one]]),
        unit=(
            (dmls, dmls, dmls),
            (rad_per_len, rad_per_len, rad_per_len),
            (dmls, dmls, dmls),
        ),
    )


@_closed_form(Cylindrical3D, Cart3D)
@plum.dispatch
def jac_pt_map(
    at: Array, from_chart: Cylindrical3D, to_chart: Cart3D, /, *, usys: OptUSys = None
) -> Array:
    r"""Compute the Jacobian of ``Cylindrical3D -> Cart3D``.

    $$
    J = \frac{\partial(x,y,z)}{\partial(\rho,\phi,z)}
      = \begin{pmatrix}
          \cos\phi & -\rho\sin\phi & 0 \\
          \sin\phi & \rho\cos\phi & 0 \\
          0 & 0 & 1
        \end{pmatrix}
    $$

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> at = jnp.array([2.0, 0.0, 3.0])
    >>> cxc.jac_pt_map(at, cxc.cyl3d, cxc.cart3d, usys=u.unitsystems.si)
    Array([[ 1., -0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  1.]], dtype=float64)

    """
    at = _real_float_point(at)
    ang = _usys_angle_per_rad(usys)
    rho, phi = at[..., 0], at[..., 1] / ang
    cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
    zero, one = jnp.zeros_like(rho), jnp.ones_like(rho)
    return jnp.array(
        [
            [cos_phi, -rho * sin_phi / ang, zero],
            [sin_phi, rho * cos_phi / ang, zero],
            [zero, zero, one],
        ]
    )


# ===================================================================
# Cart3D <-> Spherical3D
#
# `theta` is the colatitude, measured from +z.


@_closed_form(Cart3D, Spherical3D)
@plum.dispatch
def jac_pt_map(
    at: Array, from_chart: Cart3D, to_chart: Spherical3D, /, *, usys: OptUSys = None
) -> Array:
    r"""Compute the Jacobian of ``Cart3D -> Spherical3D``.

    $$
    J = \frac{\partial(r,\theta,\phi)}{\partial(x,y,z)}
      = \begin{pmatrix}
          x/r & y/r & z/r \\
          xz/(r^2\rho) & yz/(r^2\rho) & -\rho/r^2 \\
          -y/\rho^2 & x/\rho^2 & 0
        \end{pmatrix}
    $$

    with $\rho=\sqrt{x^2+y^2}$ the cylindrical radius.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> at = jnp.array([1.0, 0.0, 0.0])
    >>> cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    Array([[ 1.,  0.,  0.],
           [ 0.,  0., -1.],
           [-0.,  1.,  0.]], dtype=float64)

    """
    at = _real_float_point(at)
    x, y, z = at[..., 0], at[..., 1], at[..., 2]
    # Every entry is an O(1) direction cosine over a length, so `r**2` is never
    # formed: `r2 * rho` overflows float32 at coordinates the charts otherwise
    # handle, and the affected entry silently becomes zero.
    rho = jnp.hypot(x, y)
    r = jnp.hypot(rho, z)
    xr, yr, zr = x / r, y / r, z / r
    xrho, yrho = x / rho, y / rho
    ang = _usys_angle_per_rad(usys)
    zero = jnp.zeros_like(x)
    return jnp.array(
        [
            [xr, yr, zr],
            [xrho * zr * ang / r, yrho * zr * ang / r, -(rho / r) * ang / r],
            [-yrho * ang / rho, xrho * ang / rho, zero],
        ]
    )


@plum.dispatch
def jac_pt_map(
    at: u.AbstractQuantity,
    from_chart: Cart3D,
    to_chart: Spherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> ul.QuantityMatrix:
    r"""Compute the Jacobian of ``Cart3D -> Spherical3D`` at a unitful point.

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> at = u.Q(jnp.array([1.0, 0.0, 0.0]), "m")
    >>> cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d).value
    Array([[ 1.,  0.,  0.],
           [ 0.,  0., -1.],
           [-0.,  1.,  0.]], dtype=float64)

    """
    unit: Any = u.unit_of(at)
    v = cast("Array", u.ustrip(unit, at))
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    rho = jnp.hypot(x, y)
    r = jnp.hypot(rho, z)
    xr, yr, zr = x / r, y / r, z / r
    xrho, yrho = x / rho, y / rho
    zero = jnp.zeros_like(x)
    dmls, rad_per_len = unit / unit, RAD / unit
    return ul.QuantityMatrix(
        jnp.array(
            [
                [xr, yr, zr],
                [xrho * zr / r, yrho * zr / r, -(rho / r) / r],
                [-yrho / rho, xrho / rho, zero],
            ]
        ),
        unit=(
            (dmls, dmls, dmls),
            (rad_per_len, rad_per_len, rad_per_len),
            (rad_per_len, rad_per_len, rad_per_len),
        ),
    )


@_closed_form(Spherical3D, Cart3D)
@plum.dispatch
def jac_pt_map(
    at: Array, from_chart: Spherical3D, to_chart: Cart3D, /, *, usys: OptUSys = None
) -> Array:
    r"""Compute the Jacobian of ``Spherical3D -> Cart3D``.

    $$
    J = \frac{\partial(x,y,z)}{\partial(r,\theta,\phi)}
      = \begin{pmatrix}
          s_\theta c_\phi & r c_\theta c_\phi & -r s_\theta s_\phi \\
          s_\theta s_\phi & r c_\theta s_\phi & r s_\theta c_\phi \\
          c_\theta & -r s_\theta & 0
        \end{pmatrix}
    $$

    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> at = jnp.array([1.0, 0.0, 0.0])
    >>> cxc.jac_pt_map(at, cxc.sph3d, cxc.cart3d, usys=u.unitsystems.si)
    Array([[ 0.,  1., -0.],
           [ 0.,  0.,  0.],
           [ 1., -0.,  0.]], dtype=float64)

    """
    at = _real_float_point(at)
    ang = _usys_angle_per_rad(usys)
    r = at[..., 0]
    theta, phi = at[..., 1] / ang, at[..., 2] / ang
    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    sin_p, cos_p = jnp.sin(phi), jnp.cos(phi)
    zero = jnp.zeros_like(r)
    return jnp.array(
        [
            [sin_t * cos_p, r * cos_t * cos_p / ang, -r * sin_t * sin_p / ang],
            [sin_t * sin_p, r * cos_t * sin_p / ang, r * sin_t * cos_p / ang],
            [cos_t, -r * sin_t / ang, zero],
        ]
    )


# ===================================================================
# LonLatSpherical3D <-> LonCosLatSpherical3D
#
# `lon_coslat = lon * cos(lat)`, with `lat` and `distance` untouched. The
# generic route reaches these through colatitude and back, which is both the
# slowest transition in the suite and more arithmetic than the map needs.


@_closed_form(LonLatSpherical3D, LonCosLatSpherical3D)
@plum.dispatch
def jac_pt_map(
    at: Array,
    from_chart: LonLatSpherical3D,
    to_chart: LonCosLatSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> Array:
    r"""Compute the Jacobian of ``LonLatSpherical3D -> LonCosLatSpherical3D``.

    $$
    J = \frac{\partial(\lambda\cos\phi, \phi, d)}{\partial(\lambda, \phi, d)}
      = \begin{pmatrix}
          \cos\phi & -\lambda\sin\phi & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1
        \end{pmatrix}
    $$

    Both angles are converted to radians for the trigonometry. The *entries*
    then need no further scaling: each is an angle per angle, so the caller's
    unit cancels between numerator and denominator.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    >>> at = jnp.array([0.0, 0.0, 2.0])
    >>> cxc.jac_pt_map(at, cxc.lonlat_sph3d, cxc.loncoslat_sph3d,
    ...                usys=u.unitsystems.si)
    Array([[ 1., -0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]], dtype=float64)

    """
    at = _real_float_point(at)
    ang = _usys_angle_per_rad(usys)
    lon, lat = at[..., 0] / ang, at[..., 1] / ang
    zero, one = jnp.zeros_like(lon), jnp.ones_like(lon)
    return jnp.array(
        [
            [jnp.cos(lat), -lon * jnp.sin(lat), zero],
            [zero, one, zero],
            [zero, zero, one],
        ]
    )


@_closed_form(LonCosLatSpherical3D, LonLatSpherical3D)
@plum.dispatch
def jac_pt_map(
    at: Array,
    from_chart: LonCosLatSpherical3D,
    to_chart: LonLatSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> Array:
    r"""Compute the Jacobian of ``LonCosLatSpherical3D -> LonLatSpherical3D``.

    $$
    J = \frac{\partial(\lambda, \phi, d)}{\partial(\lambda\cos\phi, \phi, d)}
      = \begin{pmatrix}
          1/\cos\phi & \lambda\cos\phi\,\sin\phi/\cos^2\phi & 0 \\
          0 & 1 & 0 \\ 0 & 0 & 1
        \end{pmatrix}
    $$

    Singular at the poles, where `cos(lat)` vanishes and longitude is not
    recoverable from `lon * cos(lat)` -- the same place the map itself is.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    >>> at = jnp.array([0.0, 0.0, 2.0])
    >>> cxc.jac_pt_map(at, cxc.loncoslat_sph3d, cxc.lonlat_sph3d,
    ...                usys=u.unitsystems.si)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float64)

    """
    at = _real_float_point(at)
    ang = _usys_angle_per_rad(usys)
    lon_coslat, lat = at[..., 0] / ang, at[..., 1] / ang
    cos_lat = jnp.cos(lat)
    zero, one = jnp.zeros_like(lat), jnp.ones_like(lat)
    return jnp.array(
        [
            [one / cos_lat, lon_coslat * jnp.sin(lat) / cos_lat**2, zero],
            [zero, one, zero],
            [zero, zero, one],
        ]
    )
