"""Dispatch implementations for :func:`coordinaxs.api.manifolds.angle_between`."""

__all__: tuple[str, ...] = ()


import equinox as eqx
import jax
import plum

import quaxed.numpy as qnp
import unxt as u

import coordinax.angles as cxa
import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from ._utils import as_quantity_matrix, require_positive_definite
from coordinax._src.base import AbstractChart, AbstractMetricField
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.metric.matrix import DenseMetric


@plum.dispatch
def angle_between(
    chart: AbstractChart,
    uvec: CDict,
    vvec: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> cxa.AbstractAngle:
    """Manifold-level dispatch: delegate to the attached metric.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
    >>> uvec = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}
    >>> vvec = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m")}
    >>> cxm.angle_between(cxc.cart2d, uvec, vvec, at=at)
    Angle(1.57079633, 'rad')

    """
    return cxmapi.angle_between(chart.M.metric, chart, uvec, vvec, at=at, usys=usys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def angle_between(
    metric: AbstractMetricField,
    chart: AbstractChart,
    uvec: CDict,
    vvec: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> cxa.AbstractAngle:
    """Return the metric angle between two tangent vectors.

    The input component dictionaries are interpreted as tangent-vector
    components in the coordinate basis of ``chart``. The metric is evaluated at
    the base point ``at``.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.FlatMetric(3)
    >>> at = {
    ...     "r": u.Q(2.0, "m"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(0.0, "rad"),
    ... }
    >>> uvec = {"r": u.Q(0.0, "m"), "theta": u.Angle(1.0, "rad"),
    ...         "phi": u.Angle(0.0, "rad")}
    >>> vvec = {"r": u.Q(0.0, "m"), "theta": u.Angle(0.0, "rad"),
    ...         "phi": u.Angle(1.0, "rad")}
    >>> cxm.angle_between(metric, cxc.sph3d, uvec, vvec, at=at)
    Angle(1.57079633, 'rad')

    """
    # Guards `chart.M.metric` rather than the `metric` argument: the matrix below
    # comes from `chart.M`, so checking the argument would validate one metric
    # while computing with another.
    require_positive_definite(chart.M.metric, "angle_between")

    chart.check_data(at, keys=True, values=False)
    chart.check_data(uvec, keys=True, values=False)
    chart.check_data(vvec, keys=True, values=False)

    mm = cxmapi.metric_matrix(chart.M, at, chart)
    g = as_quantity_matrix(
        mm.matrix if isinstance(mm, DenseMetric) else mm.to_dense().matrix  # ty: ignore[unresolved-attribute]
    )
    u_qm = cxcapi.carray(uvec, chart.components)
    v_qm = cxcapi.carray(vvec, chart.components)

    inner = u_qm @ (g @ v_qm)
    uu = u_qm @ (g @ u_qm)
    vv = v_qm @ (g @ v_qm)

    uu, vv = _check_nonzero_norm(uu, vv)

    cosine = inner / qnp.sqrt(uu * vv)
    cosine_value = qnp.clip(u.ustrip("", cosine), -1.0, 1.0)
    return cxa.Angle(qnp.arccos(cosine_value), "rad")


def _check_nonzero_norm(*norms: u.AbstractQuantity) -> tuple[u.AbstractQuantity, ...]:
    """Raise when a norm-squared is zero or negative, traced or not.

    Mirrors `coordinax._src.charts.checks.strictly_positive`: under tracing
    (jit/vmap/grad) the check is deferred to `eqx.error_if`, so its output
    must be threaded back into the computation or the check gets DCE'd.
    """
    msg = "angle_between is undefined for zero-norm tangent vectors."
    checked = []
    for norm in norms:
        pred = u.ustrip("", qnp.any(norm <= 0))
        if isinstance(pred, jax.core.Tracer):  # ty: ignore[possibly-missing-submodule]
            norm = eqx.error_if(norm, pred, msg)  # noqa: PLW2901
        elif bool(pred):
            raise ValueError(msg)
        checked.append(norm)
    return tuple(checked)
