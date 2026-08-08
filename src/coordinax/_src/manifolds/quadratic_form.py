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
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.metric.matrix import AbstractMetricMatrix, DiagonalMetric

_MSG_MIXED = (
    "{fname}(): mixed CDict with both Quantity and bare Array values is not "
    "supported. All components must be either all Quantity or all bare Array."
)

_MSG_USYS = (
    "{fname}(): `usys` is required when `v` is a CDict of bare arrays "
    "(no unit information). "
    "Example: pass `usys=unxt.unitsystems.si`."
)


def quadratic_form(
    v: CDict,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
    fname: str = "quadratic_form",
) -> Any:
    r"""Return $v^\top G(\text{at})\, v$ for the chart's metric.

    No square root is taken, so this is defined for every metric — including
    indefinite ones, where it is the signed interval rather than a magnitude.

    Parameters
    ----------
    v
        Component dictionary of the vector to contract. Values must be *all*
        `unxt.Quantity` or *all* bare arrays; a mixture raises `TypeError`.
    chart
        The chart whose manifold supplies the metric. The metric is evaluated at
        ``at``, and ``v``'s components are read in ``chart.components`` order.
    at
        Base point at which to evaluate the metric.
    usys
        Unit system. Required when ``v`` holds bare arrays, since there is then
        no unit information to work from; optional otherwise.
    fname
        Name of the calling function, used in error messages so a caller of
        `norm` is not told about ``quadratic_form``. Mirrors the ``fname``
        argument of `require_positive_definite`.

    Examples
    --------
    >>> import jax.numpy as jnp
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

    qty_flags = [is_any_quantity(val) for val in v.values()]
    if any(qty_flags) and not all(qty_flags):
        raise TypeError(_MSG_MIXED.format(fname=fname))
    is_qty = all(qty_flags)

    if not is_qty and usys is None:
        raise TypeError(_MSG_USYS.format(fname=fname))

    # ``metric_matrix`` returns a typed ``AbstractMetricMatrix`` (Diagonal/Dense)
    # and handles both bare-array and Quantity ``at`` values; it needs no unit
    # system of its own.
    mm = cxmapi.metric_matrix(chart.M, at, chart)

    if not is_qty:  # Bare arrays — stack on last axis for correct batch broadcasting
        v_vec = jnp.stack([jnp.asarray(v[k]) for k in keys], axis=-1)
        return _contract(mm, v_vec)

    # Pack into a QuantityMatrix, preserving per-component units, then contract
    # via QuantityMatrix/AbstractMetricMatrix arithmetic, which handles all unit
    # conversions correctly (including mixed-unit components like m/s and rad/s).
    v_qm: ul.QM = cxcapi.carray(v, keys)  # ty: ignore[invalid-assignment]
    return _contract(mm, v_qm)


@plum.dispatch
def _contract(mm: AbstractMetricMatrix, v: ul.QM, /) -> Any:
    r"""Fallback: densify, then contract $v^\top G v$ -- $O(n^2)$."""
    return v @ (mm.to_dense() @ v)


@plum.dispatch
def _contract(mm: AbstractMetricMatrix, v: Array, /) -> Any:
    r"""Fallback for a stacked bare array -- $O(n^2)$.

    ``einsum`` rather than ``@``, so a leading batch axis stays distinct from
    the component axis.
    """
    return jnp.einsum("...i,...ij,...j->...", v, mm.to_dense().matrix, v)


@plum.dispatch
def _contract(mm: DiagonalMetric, v: ul.QM, /) -> Any:
    r"""Diagonal fast path -- $\sum_i g_{ii} v_i^2$, $O(n)$ instead of $O(n^2)$.

    `DiagonalMetric` stores only the diagonal precisely so the full matrix need
    never be materialised -- its own docstring says so -- but `to_dense` throws
    that structure away. Every metric the library ships is diagonal in its
    canonical chart (`FlatMetric`, `RoundMetric`, `MinkowskiMetric`), so this is
    the common path, not a corner.

    Measured on 100k points vmapped inside one `jit`: **2.2x** at n=2 (`sph2`),
    **8.3x** at n=3 (`cart3d`), **3.9x** at n=4 (`minkowskict`).

    Written ``(d * v) @ v`` rather than with a `sum`: a 1-D `QuantityMatrix`
    cannot be reduced over its logical axis, and this spelling keeps
    per-component units riding along -- the diagonal is itself a
    `QuantityMatrix` on a mixed-unit chart such as ``sph3d``.
    """
    return (mm.diagonal * v) @ v


@plum.dispatch
def _contract(mm: DiagonalMetric, v: Array, /) -> Any:
    r"""Diagonal fast path for a stacked bare array -- $O(n)$.

    Summed explicitly rather than via ``@``, which would contract a leading
    batch axis instead of the component axis.

    A *unitful* diagonal is sent back to the dense path: on a mixed-unit chart
    the per-component units cannot be factored out of a sum against a unitless
    vector. That combination does not work on the dense path either (it raises
    from the unit machinery), so this keeps the pre-existing failure rather than
    substituting a different one.
    """
    d = mm.diagonal
    if isinstance(d, ul.QM):
        return jnp.einsum("...i,...ij,...j->...", v, mm.to_dense().matrix, v)
    return jnp.sum(d * v * v, axis=-1)
