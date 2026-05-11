"""Dispatch implementations for :func:`coordinax.api.manifolds.angle_between`."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array

import jax
import numpy as np
import plum

import quaxed.numpy as qnp
import unxt as u

import coordinax.angles as cxa
import coordinax.api.manifolds as cxmapi
from coordinax._src.base import AbstractChart, AbstractMetric
from coordinax._src.custom_types import CDict, OptUSys
from coordinax.internal import QuantityMatrix, UnitsMatrix, pack_to_qmatrix


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

    Examples
    --------
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
    metric: AbstractMetric,
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
    components in the coordinate basis of ``chart``. The metric is evaluated
    at the base point ``at``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.EuclideanMetric(3)
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
    if not all(s > 0 for s in metric.signature):
        msg = (
            "angle_between currently supports only positive-definite metrics; "
            "pseudo-Riemannian or indefinite metrics are unsupported."
        )
        raise NotImplementedError(msg)

    chart.check_data(at, keys=True, values=False)
    chart.check_data(uvec, keys=True, values=False)
    chart.check_data(vvec, keys=True, values=False)

    g = _as_quantity_matrix(metric.metric_matrix(chart, at=at, usys=usys))
    u_qm = pack_to_qmatrix(uvec, keys=chart.components)
    v_qm = pack_to_qmatrix(vvec, keys=chart.components)

    inner = u_qm @ (g @ v_qm)
    uu = u_qm @ (g @ u_qm)
    vv = v_qm @ (g @ v_qm)

    _check_nonzero_norm(uu, vv)

    cosine = inner / qnp.sqrt(uu * vv)
    cosine_value = qnp.clip(u.ustrip("", cosine), -1.0, 1.0)
    return cxa.Angle(qnp.arccos(cosine_value), "rad")


def _as_quantity_matrix(x: QuantityMatrix | Array) -> QuantityMatrix:
    """Convert a numeric matrix into a dimensionless QuantityMatrix."""
    if isinstance(x, QuantityMatrix):
        return x

    n_rows, n_cols = x.shape[-2:]
    units = UnitsMatrix(np.full((n_rows, n_cols), u.unit("")))
    return QuantityMatrix(value=x, unit=units)


def _check_nonzero_norm(*norms: u.AbstractQuantity) -> None:
    """Raise when a norm-squared is zero or negative outside JAX tracing."""
    for norm in norms:
        value = norm.value
        if isinstance(value, jax.core.Tracer):  # ty: ignore[possibly-missing-submodule]
            continue
        if bool(qnp.any(value <= 0)):
            msg = "angle_between is undefined for zero-norm tangent vectors."
            raise ValueError(msg)
