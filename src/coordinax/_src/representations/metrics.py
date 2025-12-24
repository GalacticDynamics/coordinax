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
from typing import Final

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from .custom_types import PDict
from .euclidean import AbstractRep


class AbstractMetric(metaclass=abc.ABCMeta):
    r"""Abstract base class for metrics on representations.

    Mathematical definition
    -----------------------
    .. math::
       g_p: T_p M \times T_p M \to \mathbb{R}
       \\
       (g_p)_{ij} = g(\partial_i, \partial_j)

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
    >>> import coordinax as cx
    >>> import unxt as u
    >>> metric = cx.r.metric_of(cx.r.cart3d)
    >>> p = {"x": u.Quantity(1.0, "km"), "y": u.Quantity(0.0, "km"),
    ...      "z": u.Quantity(0.0, "km")}
    >>> metric.metric_matrix(cx.r.cart3d, p).shape
    (3, 3)

    """

    signature: tuple[int, ...]

    @abc.abstractmethod
    def metric_matrix(self, rep: AbstractRep, p_pos: PDict, /) -> Array:
        """Return the metric matrix evaluated at position ``p_pos``."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class EuclideanMetric(AbstractMetric):
    r"""Euclidean metric on an ``n``-dimensional Euclidean space.

    Mathematical definition
    -----------------------
    .. math::
       g_{ij} = \delta_{ij}
       \\
       \text{signature} = (1,\ldots,1)

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
    - This metric is used for Euclidean reps exposed in ``cx.r``.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> metric = cx.r.metric_of(cx.r.cart3d)
    >>> p = {"x": u.Quantity(1.0, "km"), "y": u.Quantity(2.0, "km"),
    ...      "z": u.Quantity(3.0, "km")}
    >>> metric.metric_matrix(cx.r.cart3d, p).shape
    (3, 3)

    """

    n: int

    @property
    def signature(self) -> tuple[int, ...]:
        return (1,) * self.n

    def metric_matrix(self, rep: AbstractRep, p_pos: PDict, /) -> Array:
        del rep, p_pos
        return jnp.eye(self.n)


@dataclass(frozen=True, slots=True)
class MinkowskiMetric(AbstractMetric):
    r"""Minkowski metric with signature ``(-,+,+,+)`` by default.

    Mathematical definition
    -----------------------
    .. math::
       \eta = \mathrm{diag}(\sigma_0,\sigma_1,\sigma_2,\sigma_3)
       \\
       \sigma = (-1, 1, 1, 1) \quad \text{(default)}

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
    >>> rep = cx.r.SpaceTimeCT(cx.r.cart3d)
    >>> eta = cx.r.metric_of(rep).metric_matrix(rep, {})
    >>> v = u.Quantity([2.0, 0.0, 0.0, 0.0], "km")
    >>> v_val = u.ustrip(AllowValue, "km", v)
    >>> bool(jnp.allclose(v_val @ eta @ v_val, -4.0))
    True

    """

    signature: tuple[int, ...] = (-1, 1, 1, 1)

    def metric_matrix(self, rep: AbstractRep, p_pos: PDict, /) -> Array:
        del rep, p_pos
        return jnp.diag(jnp.array(self.signature))


@dataclass(frozen=True, slots=True)
class SphereMetric(AbstractMetric):
    r"""Intrinsic metric on the unit two-sphere in ``(theta, phi)`` coordinates.

    Mathematical definition
    -----------------------
    .. math::
       g_{\theta\theta} = 1
       \\
       g_{\phi\phi} = \sin^2\theta

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
    >>> g = cx.r.metric_of(cx.r.twosphere).metric_matrix(cx.r.twosphere, p)
    >>> bool(jnp.allclose(g[1, 1], 1.0))
    True

    """

    signature: Final[tuple[int, ...]] = (1, 1)

    def metric_matrix(self, rep: AbstractRep, p_pos: PDict, /) -> Array:
        del rep
        theta = u.ustrip(AllowValue, "rad", p_pos["theta"])
        sin_theta = jnp.sin(theta)
        zero = jnp.zeros_like(sin_theta)
        one = jnp.ones_like(sin_theta)
        return jnp.array([[one, zero], [zero, sin_theta**2]])
