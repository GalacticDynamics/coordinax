"""Norm function for computing vector magnitudes using metrics."""

__all__ = ("norm",)

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from .metrics import AbstractMetric, EuclideanMetric
from coordinax._src import charts as cxc
from coordinax._src.custom_types import CsDict


@plum.dispatch.abstract
def norm(
    metric: AbstractMetric,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    v: CsDict,
    /,
    *,
    at: CsDict | None = None,
) -> u.AbstractQuantity:
    r"""Compute the norm of a vector using the given metric.

    The norm (or magnitude) of a vector is computed using the metric tensor,
    which defines the inner product on tangent spaces. This function supports
    both Euclidean and non-Euclidean metrics.

    Mathematical Definition
    -----------------------
    The norm of a vector $v$ is defined as:

    $$
       \|v\| = \sqrt{g_{ij} v^i v^j}
    $$

    where $g_{ij}$ is the metric tensor evaluated at the base point.

    For Euclidean spaces, $g_{ij} = \delta_{ij}$ (the identity matrix), so:

    $$
       \|v\| = \sqrt{\sum_i (v^i)^2}
    $$

    For curved spaces (e.g., the sphere), the metric depends on position and
    must be evaluated at a specific point.

    Parameters
    ----------
    metric : AbstractMetric
        The metric tensor to use for computing the norm. Typically obtained
        via ``metric_of(chart)``.
    chart : AbstractChart
        The chart (coordinate system) in which the vector components are
        expressed. Must match the chart used to obtain the metric.
    v : CsDict
        The vector as a component dictionary mapping component names to
        Quantities. Keys must match ``chart.components``.
    at : CsDict | None, optional
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
    metric_of : Get the default metric for a chart.
    AbstractMetric.metric_matrix : Get the metric tensor at a point.

    Notes
    -----
    - For Euclidean metrics (Cartesian, cylindrical, spherical in flat space),
      the norm is position-independent and ``at`` can be omitted.
    - For intrinsic metrics on curved manifolds (e.g., ``TwoSphere``), the
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

    >>> metric = cxm.metric_of(cxc.cart3d)
    >>> v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
    >>> cxm.norm(metric, cxc.cart3d, v)
    Quantity(Array(5., dtype=float...), unit='m')

    **Position-independent for Euclidean metrics:**

    The ``at`` parameter is optional and ignored for flat spaces:

    >>> p = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
    >>> cxm.norm(metric, cxc.cart3d, v, at=p)
    Quantity(Array(5., dtype=float...), unit='m')

    **Works in any dimension:**

    >>> metric_2d = cxm.metric_of(cxc.cart2d)
    >>> v2 = {"x": u.Q(3, "km"), "y": u.Q(4, "km")}
    >>> cxm.norm(metric_2d, cxc.cart2d, v2)
    Quantity(Array(5., dtype=float...), unit='km')

    **Velocity vectors:**

    >>> v_vel = {"x": u.Q(3, "m/s"), "y": u.Q(4, "m/s"), "z": u.Q(0, "m/s")}
    >>> cxm.norm(metric, cxc.cart3d, v_vel)
    Quantity(Array(5., dtype=float...), unit='m / s')

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
def norm(
    metric: EuclideanMetric,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    v: CsDict,
    /,
    *,
    at: CsDict | None = None,
) -> u.AbstractQuantity:
    r"""Compute the Euclidean norm of a vector.

    This is an optimized implementation for Euclidean metrics where the metric
    tensor is the identity matrix. The norm simplifies to:

    $$
       \|v\| = \sqrt{\sum_i (v^i)^2}
    $$

    The ``at`` parameter is accepted for API consistency but is ignored since
    the Euclidean metric is constant everywhere.

    Parameters
    ----------
    metric : EuclideanMetric
        The Euclidean metric (identity matrix).
    chart : AbstractChart
        The chart in which vector components are expressed.
    v : CsDict
        The vector as a component dictionary.
    at : CsDict | None, optional
        Ignored for Euclidean metrics. Accepted for API consistency.

    Returns
    -------
    Quantity
        The Euclidean norm of the vector.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.metrics as cxm
    >>> import unxt as u

    >>> metric = cxm.metric_of(cxc.cart3d)
    >>> v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
    >>> cxm.norm(metric, cxc.cart3d, v)
    Quantity(Array(5., dtype=float...), unit='m')

    Works in any dimension:

    >>> metric_2d = cxm.metric_of(cxc.cart2d)
    >>> v2 = {"x": u.Q(3, "km"), "y": u.Q(4, "km")}
    >>> cxm.norm(metric_2d, cxc.cart2d, v2)
    Quantity(Array(5., dtype=float...), unit='km')

    """
    del metric, at  # unused for Euclidean metric

    # Get unit from first component
    first_key = chart.components[0]
    unit = u.unit_of(v[first_key])

    # Stack components and compute norm
    components = jnp.stack([u.ustrip(AllowValue, unit, v[k]) for k in chart.components])
    norm_val = jnp.sqrt(jnp.sum(components**2, axis=0))

    return u.Q(norm_val, unit)


@plum.dispatch
def norm(
    metric: AbstractMetric,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    v: CsDict,
    /,
    *,
    at: CsDict,
) -> u.AbstractQuantity:
    r"""Compute the norm of a vector using a general (possibly curved) metric.

    For non-Euclidean metrics, the norm depends on the position through the
    metric tensor. This dispatch handles the general case where the metric
    matrix must be evaluated at a specific point.

    Mathematical Definition
    -----------------------
    $$
       \|v\|_p = \sqrt{g_{ij}(p) \, v^i v^j}
    $$

    where $g_{ij}(p)$ is the metric tensor evaluated at position $p$.

    Parameters
    ----------
    metric : AbstractMetric
        The metric tensor (may be position-dependent).
    chart : AbstractChart
        The chart in which vector components are expressed.
    v : CsDict
        The vector as a component dictionary.
    at : CsDict
        **Required.** The position at which to evaluate the metric tensor.

    Returns
    -------
    Quantity
        The norm of the vector computed using the metric at the given point.

    Notes
    -----
    On curved manifolds like the sphere, the metric varies with position.
    For example, on a 2-sphere with coordinates $(\theta, \phi)$:

    $$
       g = \begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}
    $$

    At the equator ($\theta = \pi/2$), both components contribute equally,
    while at the poles the $\phi$ component vanishes.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.metrics as cxm
    >>> import unxt as u
    >>> import jax.numpy as jnp

    On the two-sphere, the metric depends on position:

    >>> metric = cxm.metric_of(cxc.twosphere)
    >>> p = {"theta": u.Angle(jnp.pi/2, "rad"), "phi": u.Angle(0, "rad")}
    >>> v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(1, "rad/s")}
    >>> result = cxm.norm(metric, cxc.twosphere, v, at=p)
    >>> float(u.ustrip("rad/s", result))  # doctest: +ELLIPSIS
    1.41421...

    At the pole, the phi component doesn't contribute:

    >>> p_pole = {"theta": u.Angle(0.01, "rad"), "phi": u.Angle(0, "rad")}
    >>> v_phi = {"theta": u.Q(0, "rad/s"), "phi": u.Q(1, "rad/s")}
    >>> result_pole = cxm.norm(metric, cxc.twosphere, v_phi, at=p_pole)
    >>> float(u.ustrip("rad/s", result_pole)) < 0.02  # Nearly zero
    True

    """
    # Get the metric matrix at the position
    g = metric.metric_matrix(chart, at)

    # Get unit from first component
    first_key = chart.components[0]
    unit = u.unit_of(v[first_key])

    # Stack components into a vector
    v_arr = jnp.stack([u.ustrip(AllowValue, unit, v[k]) for k in chart.components])

    # Compute g_ij v^i v^j
    # For a vector v, the squared norm is v^T @ g @ v
    squared_norm = jnp.einsum("i,ij,j->", v_arr, g, v_arr)

    return u.Q(jnp.sqrt(squared_norm), unit)
