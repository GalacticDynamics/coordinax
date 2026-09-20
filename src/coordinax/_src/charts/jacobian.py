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
from coordinax._src.exceptions import NoGlobalCartesianChartError
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

    This is the last fallback: a pair with a closed form of its own is served
    by that, and a pair whose two legs through the Cartesian chart have one
    is chained instead. Only a pair reaching neither is differentiated here.

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

    # Two chained closed forms beat differentiating the composite, so the
    # array route takes the same chain the dict route does.
    pivot = _fast_route(from_chart, to_chart)
    if pivot is not None and pivot is not _NO_FAST_ROUTE:
        return _jac_chained(at, from_chart, pivot, to_chart, usys)

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

    The primary dict-input dispatch.  Takes the first route that applies:

    **Closed form registered for the chart pair**
        `_jac_from_dict_via_closed_form` canonicalises the components,
        evaluates the closed form, and restores the units -- for unitful and
        plain-array dicts alike.

    **Array-valued branch** (no units in any value)
        Stacks the dict values into a plain array via ``jnp.stack``, then
        forwards to ``jac_pt_map(at_arr, from_chart, to_chart, usys=usys)``.
        A pair with a closed form reads bare angles as radians; any other
        needs *usys* to say what the numbers mean, and raises `ValueError`
        without one.

    **Quantity-valued branch** (a unitful value, no closed form for the pair)
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
        ``check_data``), or -- from ``pt_map`` -- "usys must be provided for
        array input" when bare values reach a pair with no closed form.

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

    # A hand-written Jacobian beats differentiating the map, whatever the
    # values carry, so this is asked first -- and a pair with none of its own
    # still avoids `jacfwd` if two chained ones reach it.
    pivot = _fast_route(from_chart, to_chart)
    if pivot is not _NO_FAST_ROUTE:
        return _jac_from_dict_via_closed_form(at, from_chart, to_chart, usys, pivot)

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


@ft.lru_cache(maxsize=128)
def _conversion_plan(
    units: tuple[Any, ...], out_dims: tuple[str, ...], /
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
    """Plan the trip to canonical units and back, for one set of input units.

    Returns the unit to differentiate each component in, the factor its column
    is rescaled by afterwards, and the unit of every entry of the result.

    Pure in the units, so the whole plan is cached: a chart is called far more
    often than it is called with a set of units it has not seen. Bounded, as
    the repo's other per-unit caches are -- a caller building units
    programmatically should not grow one without limit.

    One unit per dimension: the first one seen, except angles, which are
    seeded as radians and so never take the input's. Seeded rather than
    special-cased in the loop because a closed form emits an angle whether or
    not the input had one -- `Cart3D -> Spherical3D` takes three lengths.
    """
    canonical: dict[Any, Any] = {ANGLE: RAD}
    dims = [u.dimension_of(unit) for unit in units]
    for unit, dim in zip(units, dims, strict=True):
        canonical.setdefault(dim, unit)

    targets = tuple(canonical[dim] for dim in dims)
    # Column *i* was differentiated with respect to the canonical unit, so it
    # is rescaled to the caller's afterwards and labelled to match.
    scale = tuple(
        1.0 if target == unit else u.uconvert_value(target, unit, 1.0)
        for unit, target in zip(units, targets, strict=True)
    )
    rows = tuple(
        tuple(canonical[u.dimension(dim)] / col for col in units) for dim in out_dims
    )
    return targets, scale, rows


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


#: Returned by `_fast_route` when neither a closed form nor a chained pair of
#: them applies, and the map has to be differentiated after all.
_NO_FAST_ROUTE: Final = object()


def _fast_route(from_chart: AbstractChart, to_chart: AbstractChart, /) -> Any:
    """How this pair avoids `jacfwd`, if it can.

    `None` when the pair has a closed form of its own, the chart to pivot
    through when two chained ones cover it, and `_NO_FAST_ROUTE` when neither
    does.
    """
    if (type(from_chart), type(to_chart)) in _CLOSED_FORM_PAIRS:
        return None
    try:
        pivot = from_chart.cartesian
        # Both charts must pivot through the *same* Cartesian chart. The
        # registry keys on chart types, which say nothing about the manifold,
        # so two charts of chainable types can still be on different ones --
        # a transition `pt_map` refuses. Comparing the charts, not their
        # types, is what keeps that refusal reachable.
        if to_chart.cartesian != pivot:
            return _NO_FAST_ROUTE
    except NoGlobalCartesianChartError:  # an intrinsic chart has no Cartesian
        return _NO_FAST_ROUTE
    legs = ((type(from_chart), type(pivot)), (type(pivot), type(to_chart)))
    if all(leg in _CLOSED_FORM_PAIRS for leg in legs):
        return pivot
    return _NO_FAST_ROUTE


def _jac_chained(
    at: Array,
    from_chart: AbstractChart,
    pivot: AbstractChart,
    to_chart: AbstractChart,
    usys: OptUSys,
    /,
) -> Array:
    """Chain two closed forms rather than differentiate the composite map.

    `J(A -> C) = J(B -> C) @ J(A -> B)`, each leg evaluated where it starts,
    so the pivot point is computed alongside the two matrices. `pt_map`
    already falls back through the Cartesian chart for a pair with no direct
    transition; this follows the same pivot.
    """
    keys = from_chart.components
    point = {k: at[..., i] for i, k in enumerate(keys)}
    mid = cast("CDict", cxcapi.pt_map(point, from_chart, pivot, usys=usys))
    mid_at = jnp.stack([mid[k] for k in pivot.components], axis=-1)
    first = jnp.asarray(cxcapi.jac_pt_map(at, from_chart, pivot, usys=usys))
    second = jnp.asarray(cxcapi.jac_pt_map(mid_at, pivot, to_chart, usys=usys))
    return second @ first


def _jac_from_dict_via_closed_form(
    at: CDict,
    from_chart: AbstractChart,
    to_chart: AbstractChart,
    usys: OptUSys,
    pivot: Any,
    /,
) -> Any:
    """Send a coordinate dict to the closed form that serves its chart pair.

    Called with a checked, unbatched point and the route `_fast_route` chose:
    *pivot* is `None` when the pair has a closed form of its own, and the
    chart to chain through when two of them cover it instead. Either way the
    Jacobian arrives on bare values and the unit handling here is the same.

    The closed forms take bare values, so strip every component to a canonical
    unit -- angles to radians, everything else to the first unit of its kind --
    differentiate there, and put the units back afterwards. Convertible units
    are equivalent, so canonicalising costs a label and nothing else.

    Bare on purpose, not for want of a container: a `QuantityMatrix` carries
    per-column units and would pack this point exactly. Feeding one to a closed
    form would put the arithmetic back on `Quantity` operands, and `quax` costs
    a trace per primitive there -- the whole reason these bodies compute on raw
    arrays.
    """
    keys = from_chart.components
    units = tuple(u.unit_of(at[k]) for k in keys)

    # No units to strip or restore; *usys* says what the numbers mean.
    if all(unit is None for unit in units):
        # Straight to the `Array` route, which takes the same three routes
        # this one does -- including the chain, so no pivot branch here.
        at_arr = jnp.stack([at[k] for k in keys], axis=-1)
        return cxcapi.jac_pt_map(at_arr, from_chart, to_chart, usys=usys)
    # A bare component beside a unitful one has no unit to canonicalise to.
    if any(unit is None for unit in units):
        return _jac_via_autodiff(at, from_chart, to_chart, usys)

    targets, scale, unit_rows = _conversion_plan(units, to_chart.coord_dimensions)

    values = [
        u.ustrip(unit, at[k])
        if target == unit
        else u.uconvert_value(target, unit, u.ustrip(unit, at[k]))
        for k, unit, target in zip(keys, units, targets, strict=True)
    ]
    raw = tree_cast_int_bool_to_float(jnp.stack(values, axis=-1))

    jac = (
        _jac_chained(raw, from_chart, pivot, to_chart, None)
        if pivot is not None
        else jnp.asarray(cxcapi.jac_pt_map(raw, from_chart, to_chart, usys=None))
    )

    if any(f != 1.0 for f in scale):
        jac = jac * jnp.asarray(scale, dtype=jac.dtype)
    return ul.QuantityMatrix(jac, unit=unit_rows)


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


# ===================================================================
# Cart3D <-> LonLatSpherical3D
#
# `lat` is the latitude, measured from the equator, where `Spherical3D` takes
# a colatitude from +z.


@_closed_form(Cart3D, LonLatSpherical3D)
@plum.dispatch
def jac_pt_map(
    at: Array,
    from_chart: Cart3D,
    to_chart: LonLatSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> Array:
    r"""Compute the Jacobian of ``Cart3D -> LonLatSpherical3D``.

    $$
    J = \frac{\partial(\lambda, \phi, d)}{\partial(x,y,z)}
      = \begin{pmatrix}
          -y/\rho^2 & x/\rho^2 & 0 \\
          -xz/(r^2\rho) & -yz/(r^2\rho) & \rho/r^2 \\
          x/r & y/r & z/r
        \end{pmatrix}
    $$

    The latitude row is the colatitude's negated, `lat` increasing towards +z
    where `theta` decreases.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    >>> at = jnp.array([1.0, 0.0, 0.0])
    >>> cxc.jac_pt_map(at, cxc.cart3d, cxc.lonlat_sph3d, usys=u.unitsystems.si)
    Array([[-0.,  1.,  0.],
           [-0., -0.,  1.],
           [ 1.,  0.,  0.]], dtype=float64)

    """
    at = _real_float_point(at)
    x, y, z = at[..., 0], at[..., 1], at[..., 2]
    rho = jnp.hypot(x, y)
    r = jnp.hypot(rho, z)
    xr, yr, zr = x / r, y / r, z / r
    xrho, yrho = x / rho, y / rho
    ang = _usys_angle_per_rad(usys)
    zero = jnp.zeros_like(x)
    return jnp.array(
        [
            [-yrho * ang / rho, xrho * ang / rho, zero],
            [-xrho * zr * ang / r, -yrho * zr * ang / r, (rho / r) * ang / r],
            [xr, yr, zr],
        ]
    )


@_closed_form(LonLatSpherical3D, Cart3D)
@plum.dispatch
def jac_pt_map(
    at: Array,
    from_chart: LonLatSpherical3D,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> Array:
    r"""Compute the Jacobian of ``LonLatSpherical3D -> Cart3D``.

    $$
    J = \frac{\partial(x,y,z)}{\partial(\lambda, \phi, d)}
      = \begin{pmatrix}
          -d\cos\phi\sin\lambda & -d\sin\phi\cos\lambda & \cos\phi\cos\lambda \\
          d\cos\phi\cos\lambda & -d\sin\phi\sin\lambda & \cos\phi\sin\lambda \\
          0 & d\cos\phi & \sin\phi
        \end{pmatrix}
    $$

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    >>> at = jnp.array([0.0, 0.0, 2.0])
    >>> cxc.jac_pt_map(at, cxc.lonlat_sph3d, cxc.cart3d, usys=u.unitsystems.si)
    Array([[-0., -0.,  1.],
           [ 2., -0.,  0.],
           [ 0.,  2.,  0.]], dtype=float64)

    """
    at = _real_float_point(at)
    ang = _usys_angle_per_rad(usys)
    lon, lat, dist = at[..., 0] / ang, at[..., 1] / ang, at[..., 2]
    cos_lat, sin_lat = jnp.cos(lat), jnp.sin(lat)
    cos_lon, sin_lon = jnp.cos(lon), jnp.sin(lon)
    zero = jnp.zeros_like(lon)
    return jnp.array(
        [
            [
                -dist * cos_lat * sin_lon / ang,
                -dist * sin_lat * cos_lon / ang,
                cos_lat * cos_lon,
            ],
            [
                dist * cos_lat * cos_lon / ang,
                -dist * sin_lat * sin_lon / ang,
                cos_lat * sin_lon,
            ],
            [zero, dist * cos_lat / ang, sin_lat],
        ]
    )


# ===================================================================
# Cylindrical3D <-> Spherical3D


@_closed_form(Cylindrical3D, Spherical3D)
@plum.dispatch
def jac_pt_map(
    at: Array,
    from_chart: Cylindrical3D,
    to_chart: Spherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> Array:
    r"""Compute the Jacobian of ``Cylindrical3D -> Spherical3D``.

    $$
    J = \frac{\partial(r,\theta,\phi)}{\partial(\rho,\phi,z)}
      = \begin{pmatrix}
          \rho/r & 0 & z/r \\ z/r^2 & 0 & -\rho/r^2 \\ 0 & 1 & 0
        \end{pmatrix}
    $$

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    >>> at = jnp.array([1.0, 0.0, 0.0])
    >>> cxc.jac_pt_map(at, cxc.cyl3d, cxc.sph3d, usys=u.unitsystems.si)
    Array([[ 1.,  0.,  0.],
           [ 0.,  0., -1.],
           [ 0.,  1.,  0.]], dtype=float64)

    """
    at = _real_float_point(at)
    rho, z = at[..., 0], at[..., 2]
    r = jnp.hypot(rho, z)
    ang = _usys_angle_per_rad(usys)
    zero, one = jnp.zeros_like(rho), jnp.ones_like(rho)
    return jnp.array(
        [
            [rho / r, zero, z / r],
            [(z / r) * ang / r, zero, -(rho / r) * ang / r],
            [zero, one, zero],
        ]
    )


@_closed_form(Spherical3D, Cylindrical3D)
@plum.dispatch
def jac_pt_map(
    at: Array,
    from_chart: Spherical3D,
    to_chart: Cylindrical3D,
    /,
    *,
    usys: OptUSys = None,
) -> Array:
    r"""Compute the Jacobian of ``Spherical3D -> Cylindrical3D``.

    $$
    J = \frac{\partial(\rho,\phi,z)}{\partial(r,\theta,\phi)}
      = \begin{pmatrix}
          \sin\theta & r\cos\theta & 0 \\ 0 & 0 & 1 \\
          \cos\theta & -r\sin\theta & 0
        \end{pmatrix}
    $$

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    >>> at = jnp.array([2.0, 0.0, 0.0])
    >>> cxc.jac_pt_map(at, cxc.sph3d, cxc.cyl3d, usys=u.unitsystems.si)
    Array([[ 0.,  2.,  0.],
           [ 0.,  0.,  1.],
           [ 1., -0.,  0.]], dtype=float64)

    """
    at = _real_float_point(at)
    ang = _usys_angle_per_rad(usys)
    r, theta = at[..., 0], at[..., 1] / ang
    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    zero, one = jnp.zeros_like(r), jnp.ones_like(r)
    return jnp.array(
        [
            [sin_t, r * cos_t / ang, zero],
            [zero, zero, one],
            [cos_t, -r * sin_t / ang, zero],
        ]
    )


# ===================================================================
# Cart3D <-> LonCosLatSpherical3D
#
# `lon_coslat = lon * cos(lat)`. Both directions are a `LonLat` Jacobian and
# that chart's own closed form, chained -- not a third derivation of the same
# trigonometry. They are registered so that pairs pivoting through `Cart3D`
# reach `LonCosLat` too.

#: The pivot both directions below chain through.
_LONLAT3D: Final = LonLatSpherical3D()


@_closed_form(Cart3D, LonCosLatSpherical3D)
@plum.dispatch
def jac_pt_map(
    at: Array,
    from_chart: Cart3D,
    to_chart: LonCosLatSpherical3D,
    /,
    *,
    usys: OptUSys = None,
) -> Array:
    r"""Compute the Jacobian of ``Cart3D -> LonCosLatSpherical3D``.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    >>> at = jnp.array([1.0, 0.0, 0.0])
    >>> cxc.jac_pt_map(at, cxc.cart3d, cxc.loncoslat_sph3d, usys=u.unitsystems.si)
    Array([[0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.]], dtype=float64)

    """
    return _jac_chained(_real_float_point(at), from_chart, _LONLAT3D, to_chart, usys)


@_closed_form(LonCosLatSpherical3D, Cart3D)
@plum.dispatch
def jac_pt_map(
    at: Array,
    from_chart: LonCosLatSpherical3D,
    to_chart: Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> Array:
    r"""Compute the Jacobian of ``LonCosLatSpherical3D -> Cart3D``.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import jax.numpy as jnp

    >>> at = jnp.array([0.0, 0.0, 2.0])
    >>> cxc.jac_pt_map(at, cxc.loncoslat_sph3d, cxc.cart3d, usys=u.unitsystems.si)
    Array([[0., 0., 1.],
           [2., 0., 0.],
           [0., 2., 0.]], dtype=float64)

    """
    return _jac_chained(_real_float_point(at), from_chart, _LONLAT3D, to_chart, usys)
