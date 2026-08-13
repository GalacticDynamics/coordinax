r"""The metric quadratic form, shared by every magnitude-like manifold verb.

Every function that asks "how big is this, according to the metric" evaluates the
same contraction:

$$ Q_p(v) = v^\top G(p)\, v. $$

`~coordinax.manifolds.norm` is its square root, and
`~coordinax.manifolds.separation` is that applied to a coordinate difference.
Anything defined on an *indefinite* metric — where the square root has no real
value — needs the contraction **unrooted** instead.

Keeping one implementation matters for more than line count: the unit handling
here is fiddly (mixed-unit components, bare arrays that carry no units at all,
per-component units that must survive the contraction), and a second copy
reliably ends up with a subset of the checks.  It is a private helper rather than
public API because the two public spellings, `norm` and ``interval``, already
cover the ways a caller wants to ask.
"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array
from typing import Any

import plum

import quaxed.numpy as jnp
import unxts.linalg as ul
from unxt.quantity import is_any_quantity

import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from coordinax._src.base import AbstractChart
from coordinax._src.custom_types import OptUSys
from coordinax._src.metric.matrix import AbstractMetricMatrix, DiagonalMetric
from coordinaxs.api.custom_types import CDict

_MSG_PACKED_SHAPE = (
    "{fname}(): packed Quantity has trailing dimension {got}, but chart "
    "component{plural} {comps} expect{verb} {want}."
)

_MSG_PACKED_NDIM = (
    "{fname}(): packed Quantity is a scalar, but chart components {comps} "
    "expect a trailing axis of length {want}."
)

_MSG_MIXED = (
    "{fname}(): mixed CDict with both Quantity and bare Array values is not "
    "supported. All components must be either all Quantity or all bare Array."
)

_MSG_USYS = (
    "{fname}(): `usys` is required when `v` is a CDict of bare arrays "
    "(no unit information). "
    "Example: pass `usys=unxt.unitsystems.si`."
)

_MSG_CROSS_VECTOR_MIXED = (
    "{fname}(): all vectors must be consistently either Quantity-valued or "
    "bare-array-valued. Mixing a Quantity CDict with a bare-array CDict "
    "across arguments is not supported."
)


def bilinear_form(
    uvec: CDict,
    vvec: CDict,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
    fname: str = "bilinear_form",
    require_usys: bool = True,
) -> Any:
    r"""Return $u^\top G(\text{at})\, v$ for the chart's metric.

    The general contraction: `quadratic_form` is the ``u is v`` case, and
    `~coordinax.manifolds.angle_between` needs all three of $g(u,v)$, $g(u,u)$
    and $g(v,v)$. No square root is taken, so this is defined for every metric,
    including indefinite ones.

    Parameters
    ----------
    uvec, vvec
        Component dictionaries of the two vectors. Within each, values must be
        *all* `unxt.Quantity` or *all* bare arrays; a mixture raises `TypeError`.
    chart
        The chart whose manifold supplies the metric, evaluated at ``at``.
        Components are read in ``chart.components`` order.
    at
        Base point at which to evaluate the metric.
    usys
        Unit system. Required when the vectors hold bare arrays, since there is
        then no unit information to work from.
    fname
        Name of the calling function, used in error messages so a caller of
        `norm` is not told about `bilinear_form`. Mirrors the ``fname``
        argument of `require_positive_definite`.
    require_usys
        Whether bare-array components must be accompanied by ``usys``. True for
        verbs whose *result* carries units derived from the inputs (`norm`,
        ``interval``), where declaring a unit system is a meaningful contract.
        False for verbs returning a dimensionless ratio
        (`~coordinax.manifolds.angle_between`), where every unit cancels and the
        demand would be ceremony.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> from coordinax._src.manifolds.quadratic_form import bilinear_form

    Orthogonal directions contract to zero; a vector with itself gives its
    squared length:

    >>> at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
    >>> xhat = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> yhat = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")}
    >>> bilinear_form(xhat, yhat, cxc.cart3d, at=at).round(2)
    Q(0., 'm2')
    >>> bilinear_form(xhat, xhat, cxc.cart3d, at=at).round(2)
    Q(1., 'm2')

    It is symmetric, as the metric is:

    >>> v = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> a = bilinear_form(v, xhat, cxc.cart3d, at=at)
    >>> b = bilinear_form(xhat, v, cxc.cart3d, at=at)
    >>> bool(a == b)
    True

    """
    keys = chart.components
    mm, packed = _prepare(
        chart,
        (uvec, vvec),
        at=at,
        usys=usys,
        fname=fname,
        keys=keys,
        require_usys=require_usys,
    )
    return _contract(mm, *packed)


def quadratic_form(
    v: CDict,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
    fname: str = "quadratic_form",
    require_usys: bool = True,
) -> Any:
    r"""Return $v^\top G(\text{at})\, v$ for the chart's metric.

    The ``u is v`` case of `bilinear_form`, which is what every
    magnitude-like verb needs: `~coordinax.manifolds.norm` is its square root,
    and ``interval`` is it applied to a coordinate difference. No square root is
    taken here, so it is defined for every metric -- including indefinite ones,
    where it is a signed interval rather than a magnitude.

    Parameters
    ----------
    v
        Component dictionary of the vector to contract. Values must be *all*
        `unxt.Quantity` or *all* bare arrays; a mixture raises `TypeError`.
    chart
        The chart whose manifold supplies the metric, evaluated at ``at``.
    at
        Base point at which to evaluate the metric.
    usys
        Unit system. Required when ``v`` holds bare arrays.
    fname
        Name of the calling function, used in error messages.
    require_usys
        Whether bare-array components must be accompanied by ``usys``; see
        `bilinear_form`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> from coordinax._src.manifolds.quadratic_form import quadratic_form

    A 3-4-0 vector in flat space contracts to ``25 m2`` -- the *square* of the
    norm ``5 m``:

    >>> at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
    >>> v = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> quadratic_form(v, cxc.cart3d, at=at).round(2)
    Q(25., 'm2')

    Unlike a norm it stays defined when the metric is indefinite, going negative
    for a timelike vector:

    >>> at4 = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> v4 = {"ct": u.Q(5.0, "m"), "x": u.Q(1.0, "m"),
    ...       "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> quadratic_form(v4, cxc.minkowskict, at=at4).round(2)
    Q(-24., 'm2')

    """
    keys = chart.components
    mm, packed = _prepare(
        chart, (v,), at=at, usys=usys, fname=fname, keys=keys, require_usys=require_usys
    )
    return _contract(mm, packed[0], packed[0])


def gram(
    uvec: CDict,
    vvec: CDict,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
    fname: str = "gram",
    require_usys: bool = True,
) -> tuple[Any, Any, Any]:
    r"""Return $(g(u,v),\; g(u,u),\; g(v,v))$ from a *single* metric evaluation.

    Three `bilinear_form` calls would rebuild the metric matrix three times.
    Under `jax.jit` that costs nothing -- XLA common-subexpression-eliminates
    the identical builds, which is why the compiled op count is unchanged -- but
    eagerly it is a measured ~1.6x on `~coordinax.manifolds.angle_between`, the
    one caller that needs all three at the same base point.

    So the metric is evaluated once and each vector packed once, then contracted
    three times. Same unit handling as `bilinear_form`, since it is the same
    `_prepare`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> from coordinax._src.manifolds.quadratic_form import gram

    >>> at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
    >>> xhat = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> v = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> uv, uu, vv = gram(xhat, v, cxc.cart3d, at=at)
    >>> uv.round(2), uu.round(2), vv.round(2)
    (Q(3., 'm2'), Q(1., 'm2'), Q(25., 'm2'))

    """
    mm, (u_, v_) = _prepare(
        chart,
        (uvec, vvec),
        at=at,
        usys=usys,
        fname=fname,
        keys=chart.components,
        require_usys=require_usys,
    )
    return _contract(mm, u_, v_), _contract(mm, u_, u_), _contract(mm, v_, v_)


def _check_packed_shape(v: Any, keys: tuple[str, ...], fname: str, /) -> None:
    """Validate a packed Quantity's trailing axis against the chart.

    Splitting the quantity into a CDict used to perform this check on the way
    past. Skipping the round-trip skipped the check with it, so a mis-sized
    quantity reached `QuantityMatrix` and failed there -- still an error, never a
    silent broadcast, but phrased in terms of matrix internals rather than the
    chart contract the caller actually broke.
    """
    want = len(keys)
    shape = jnp.shape(v.value)
    if not shape:
        raise ValueError(_MSG_PACKED_NDIM.format(fname=fname, comps=keys, want=want))
    if shape[-1] != want:
        raise ValueError(
            _MSG_PACKED_SHAPE.format(
                fname=fname,
                got=shape[-1],
                plural="" if want == 1 else "s",
                comps=keys,
                verb="s" if want == 1 else "",
                want=want,
            )
        )


def _prepare(
    chart: AbstractChart,
    vecs: tuple[Any, ...],
    /,
    *,
    at: CDict,
    usys: OptUSys,
    fname: str,
    keys: tuple[str, ...],
    require_usys: bool,
) -> tuple[Any, tuple[Any, ...]]:
    """Validate the vectors, build the metric matrix, and pack for contraction.

    All the unit handling lives here, once: it is the fiddly part, and a second
    copy reliably ends up with a subset of the checks.
    """
    packed: list[Any] = []
    for vec in vecs:
        if is_any_quantity(vec):
            # Already packed: nothing to validate component-wise, and the dict
            # round-trip below would only take it apart to put it back together.
            # The shape check that round-trip used to perform is kept, though --
            # see `_check_packed_shape`.
            _check_packed_shape(vec, keys, fname)
            packed.append(None)
            continue
        qty_flags = [is_any_quantity(val) for val in vec.values()]
        if any(qty_flags) and not all(qty_flags):
            raise TypeError(_MSG_MIXED.format(fname=fname))
        if require_usys and not all(qty_flags) and usys is None:
            raise TypeError(_MSG_USYS.format(fname=fname))
        packed.append(qty_flags)

    # Cross-argument check: all vectors must be consistently Quantity or bare-array.
    # Each element in `packed` is a list of bool flags; `all(flags)` means
    # all-Quantity, `not all(flags)` means all-bare-array (the within-vector
    # check above ensures no mixing).
    if len(packed) > 1:
        is_qty_per_vec = [all(qty_flags) for qty_flags in packed]
        if any(is_qty_per_vec) and not all(is_qty_per_vec):
            raise TypeError(_MSG_CROSS_VECTOR_MIXED.format(fname=fname))

    # ``metric_matrix`` returns a typed ``AbstractMetricMatrix`` (Diagonal/Dense)
    # and handles both bare-array and Quantity ``at`` values; it needs no unit
    # system of its own.
    mm = cxmapi.metric_matrix(chart.M, at, chart)

    out: list[Any] = []
    for vec, qty_flags in zip(vecs, packed, strict=True):
        if qty_flags is None:
            # Packed Quantity -> QuantityMatrix directly, one uniform unit.
            out.append(
                ul.QuantityMatrix(
                    vec.value, unit=ul.UnitsMatrix.full((len(keys),), vec.unit)
                )
            )
        elif not all(qty_flags):
            # Bare arrays — stack on the last axis for correct batch broadcasting.
            out.append(jnp.stack([jnp.asarray(vec[k]) for k in keys], axis=-1))
        else:
            # Pack into a QuantityMatrix, preserving per-component units, so the
            # contraction handles mixed-unit components (m/s alongside rad/s).
            out.append(cxcapi.carray(vec, keys))
    return mm, tuple(out)


@plum.dispatch
def _contract(mm: AbstractMetricMatrix, uvec: ul.QM, vvec: ul.QM, /) -> Any:
    r"""Fallback: densify, then contract $u^\top G v$ -- $O(n^2)$."""
    return uvec @ (mm.to_dense() @ vvec)


@plum.dispatch
def _contract(mm: AbstractMetricMatrix, uvec: Array, vvec: Array, /) -> Any:
    r"""Fallback for stacked bare arrays -- $O(n^2)$.

    ``einsum`` rather than ``@``, so a leading batch axis stays distinct from
    the component axis.
    """
    return jnp.einsum("...i,...ij,...j->...", uvec, mm.to_dense().matrix, vvec)


@plum.dispatch
def _contract(mm: DiagonalMetric, uvec: ul.QM, vvec: ul.QM, /) -> Any:
    r"""Diagonal fast path -- $\sum_i g_{ii} u_i v_i$, $O(n)$ instead of $O(n^2)$.

    `DiagonalMetric` stores only the diagonal precisely so the full matrix need
    never be materialised -- its own docstring says so -- but `to_dense` throws
    that structure away. Every metric the library ships is diagonal in its
    canonical chart (`FlatMetric`, `RoundMetric`, `MinkowskiMetric`), so this is
    the common path, not a corner.

    Measured on 100k points vmapped inside one `jit`: **2.2x** at n=2 (`sph2`),
    **8.3x** at n=3 (`cart3d`), **3.9x** at n=4 (`minkowskict`).

    Written ``(d * u) @ v`` rather than with a `sum`: a 1-D `QuantityMatrix`
    cannot be reduced over its logical axis, and this spelling keeps
    per-component units riding along -- the diagonal is itself a
    `QuantityMatrix` on a mixed-unit chart such as ``sph3d``.
    """
    return (mm.diagonal * uvec) @ vvec


@plum.dispatch
def _contract(mm: DiagonalMetric, uvec: Array, vvec: Array, /) -> Any:
    r"""Diagonal fast path for stacked bare arrays -- $O(n)$.

    Summed explicitly rather than via ``@``, which would contract a leading
    batch axis instead of the component axis.

    A *unitful* diagonal is sent back to the dense path: on a mixed-unit chart
    the per-component units cannot be factored out of a sum against unitless
    vectors. That combination does not work on the dense path either, so this
    keeps the pre-existing failure rather than substituting a different one.
    """
    d = mm.diagonal
    if isinstance(d, ul.QM):
        return jnp.einsum("...i,...ij,...j->...", uvec, mm.to_dense().matrix, vvec)
    return jnp.sum(d * uvec * vvec, axis=-1)
