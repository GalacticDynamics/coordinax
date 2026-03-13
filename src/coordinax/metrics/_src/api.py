__all__ = ("norm",)

from typing import TYPE_CHECKING

import plum

import unxt as u

import coordinax.charts as cxc
from coordinax.internal.custom_types import CDict

if TYPE_CHECKING:
    import coordinax.metrics  # noqa: ICN001


@plum.dispatch.abstract
def norm(
    metric: "coordinax.metrics.AbstractMetric",
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    v: CDict,
    /,
    *,
    at: CDict | None = None,
) -> u.AbstractQuantity:
    r"""Compute the norm of a vector using the given metric.

    The norm (or magnitude) of a vector is computed using the metric tensor,
    which defines the inner product on tangent spaces. This function supports
    both Euclidean and non-Euclidean metrics.

    Mathematical Definition:

    The norm of a vector $v$ is defined as:

    $$ \|v\| = \sqrt{g_{ij} v^i v^j} $$

    where $g_{ij}$ is the metric tensor evaluated at the base point.

    For Euclidean spaces, $g_{ij} = \delta_{ij}$ (the identity matrix), so:

    $$ \|v\| = \sqrt{\sum_i (v^i)^2} $$

    For curved spaces (e.g., the sphere), the metric depends on position and
    must be evaluated at a specific point.

    Parameters
    ----------
    metric : AbstractMetric
        The metric tensor to use for computing the norm.
    chart : AbstractChart
        The chart (coordinate system) in which the vector components are
        expressed. Must match the chart used to obtain the metric.
    v : CDict
        The vector as a component dictionary mapping component names to
        Quantities. Keys must match ``chart.components``.
    at : CDict | None, optional
        The position at which to evaluate the metric tensor. Required for
        non-Euclidean metrics where the metric varies with position.
        For Euclidean metrics, this parameter is ignored.

    Returns
    -------
    Quantity
        The norm (magnitude) of the vector, with units matching the input
        vector components.

    See Also
    --------
    AbstractMetric.metric_matrix : Get the metric tensor at a point.

    Notes
    -----
    - For Euclidean metrics (Cartesian, cylindrical, spherical in flat space),
      the norm is position-independent and ``at`` can be omitted.
    - For intrinsic metrics on curved manifolds (e.g., ``SphericalTwoSphere``), the
      ``at`` parameter is required since the metric varies with position.
    - The function dispatches on the metric type to use optimized
      implementations where possible.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.metrics as cxm
    >>> import unxt as u

    **Euclidean norm in Cartesian coordinates:**

    The classic 3-4-5 right triangle:

    >>> metric = cxm.EuclideanMetric(3)
    >>> v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
    >>> cxm.norm(metric, cxc.cart3d, v)
    Quantity(Array(5., dtype=float...), unit='m')

    **Position-independent for Euclidean metrics:**

    The ``at`` parameter is optional and ignored for flat spaces:

    >>> p = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
    >>> cxm.norm(metric, cxc.cart3d, v, at=p)
    Quantity(Array(5., dtype=float...), unit='m')

    **Works in any dimension:**

    >>> metric_2d = cxm.EuclideanMetric(2)
    >>> v2 = {"x": u.Q(3, "km"), "y": u.Q(4, "km")}
    >>> cxm.norm(metric_2d, cxc.cart2d, v2)
    Quantity(Array(5., dtype=float...), unit='km')

    **Velocity vectors:**

    >>> v_vel = {"x": u.Q(3, "m/s"), "y": u.Q(4, "m/s"), "z": u.Q(0, "m/s")}
    >>> cxm.norm(metric, cxc.cart3d, v_vel)
    Quantity(Array(5., dtype=float...), unit='m / s')

    """
    raise NotImplementedError  # pragma: no cover
