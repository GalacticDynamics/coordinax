"""Metric definitions for coordinate representations."""

__all__ = (
    "AbstractMetric",
    "EuclideanMetric",
    "MinkowskiMetric",
    "SphereMetric",
)

import abc
from dataclasses import dataclass

from jaxtyping import Array
from typing import Any, final

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import coordinax.charts as cxc
from coordinax.api import CsDict


class AbstractMetric(metaclass=abc.ABCMeta):
    r"""Abstract base class for metrics on representations.

    Mathematical definition:
    $$
       g_p: T_p M \times T_p M \to \mathbb{R}
       \\
       (g_p)_{ij} = g(\partial_i, \partial_j)
    $$

    Parameters
    ----------
    rep
        Representation whose components index the chart basis.
    p_pos
        Coordinate values at which the metric is evaluated.

    Returns
    -------
    Array
        Metric matrix ``g_{ij}`` in the chart basis for ``rep``.

    Notes
    -----
    - The metric matrix is defined on the tangent space of the chart.
    - This class describes the bilinear form; it does not store coordinates.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.metrics as cxm
    >>> import unxt as u
    >>> metric = cxm.metric_of(cxc.cart3d)
    >>> p = {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"),
    ...      "z": u.Q(0.0, "km")}
    >>> metric.metric_matrix(cxc.cart3d, p).shape
    (3, 3)

    """

    @property
    @abc.abstractmethod
    def signature(self) -> tuple[int, ...]:
        """Return the signature of the metric as a tuple of integers."""
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def metric_matrix(
        self, chart: cxc.AbstractChart[Any, Any], p_pos: CsDict, /
    ) -> Array:
        """Return the metric matrix evaluated at position ``p_pos``."""
        raise NotImplementedError


@final
@dataclass(frozen=True, slots=True)
class EuclideanMetric(AbstractMetric):
    r"""Euclidean metric on an ``n``-dimensional Euclidean space.

    Mathematical definition:
    $$
       g_{ij} = \delta_{ij}
       \\
       \text{signature} = (1,\ldots,1)
    $$

    Parameters
    ----------
    n
        Dimension of the Euclidean space.

    Returns
    -------
    Array
        Identity matrix of shape ``(n, n)``.

    Notes
    -----
    - This metric is used for Euclidean reps exposed in ``cxc.cart3d``.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> metric = cxm.metric_of(cxc.cart3d)
    >>> p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"),
    ...      "z": u.Q(3.0, "km")}
    >>> metric.metric_matrix(cxc.cart3d, p).shape
    (3, 3)

    """

    n: int

    @property
    def signature(self) -> tuple[int, ...]:
        return (1,) * self.n

    def metric_matrix(
        self, chart: cxc.AbstractChart[Any, Any], p_pos: CsDict, /
    ) -> Array:
        del chart, p_pos
        return jnp.eye(self.n)


@final
@dataclass(frozen=True, slots=True)
class MinkowskiMetric(AbstractMetric):
    r"""Minkowski metric with signature ``(-,+,+,+)`` by default.

    Mathematical definition:
    $$
       \eta = \mathrm{diag}(\sigma_0,\sigma_1,\sigma_2,\sigma_3)
       \\
       \sigma = (-1, 1, 1, 1) \quad \text{(default)}
    $$

    Parameters
    ----------
    signature
        Tuple of signs defining the diagonal of the metric.

    Returns
    -------
    Array
        Constant diagonal matrix ``\eta``.

    Notes
    -----
    - The default convention is ``(-,+,+,+)`` (time-like negative).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> from unxt.quantity import AllowValue
    >>> rep = cxc.SpaceTimeCT(cxc.cart3d)
    >>> eta = cxm.metric_of(rep).metric_matrix(rep, {})
    >>> v = u.Q([2.0, 0.0, 0.0, 0.0], "km")
    >>> v_val = u.ustrip(AllowValue, "km", v)
    >>> bool(jnp.allclose(v_val @ eta @ v_val, -4.0))
    True

    """

    @property
    def signature(self) -> tuple[int, ...]:
        return (-1, 1, 1, 1)

    def metric_matrix(
        self, chart: cxc.AbstractChart[Any, Any], p_pos: CsDict, /
    ) -> Array:
        del chart, p_pos
        return jnp.diag(jnp.array(self.signature))


@final
@dataclass(frozen=True, slots=True)
class SphereMetric(AbstractMetric):
    r"""Intrinsic metric on the unit two-sphere in ``(theta, phi)`` coordinates.

    Mathematical definition:
    $$
       g_{\theta\theta} = 1
       \\
       g_{\phi\phi} = \sin^2\theta
    $$

    Parameters
    ----------
    p_pos
        Intrinsic coordinates with ``theta`` as an angle.

    Returns
    -------
    Array
        Diagonal matrix with entries ``(1, sin(theta)^2)``.

    Notes
    -----
    - This is the intrinsic metric of the unit sphere; radius scaling is handled
      by the embedding parameters, not by ``TwoSphere`` itself.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx
    >>> import unxt as u
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> metric = cxm.metric_of(cxc.twosphere)
    >>> g = metric.metric_matrix(cxc.twosphere, p)
    >>> bool(jnp.allclose(g[1, 1], 1.0))
    True

    """

    @property
    def signature(self) -> tuple[int, ...]:
        return (1, 1)

    def metric_matrix(
        self, chart: cxc.AbstractChart[Any, Any], p_pos: CsDict, /
    ) -> Array:
        del chart
        theta = u.ustrip(AllowValue, "rad", p_pos["theta"])
        sin_theta = jnp.sin(theta)
        zero = jnp.zeros_like(sin_theta)
        one = jnp.ones_like(sin_theta)
        return jnp.array([[one, zero], [zero, sin_theta**2]])
