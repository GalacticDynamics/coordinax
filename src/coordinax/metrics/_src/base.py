"""Metric definitions for coordinate representations."""

__all__ = ("AbstractMetric",)

import abc

from jaxtyping import Array
from typing import Any

import unxt as u

import coordinax.charts as cxc
from . import api
from coordinax.internal.custom_types import CDict


class AbstractMetric(metaclass=abc.ABCMeta):
    r"""Abstract base class for metrics on representations.

    The metric defines a bilinear form on the tangent space of a chart.

    $$ g_p: T_p M \times T_p M \to \mathbb{R} $$

    The metric can be represented as a matrix in the coordinate basis of a
    chart. Let $(U,\varphi)$ be a chart with coordinates $q = (q^1, \dots,
    q^n)$.

    The coordinate basis of the tangent space is

    $\left\{ \frac{\partial}{\partial q^i} \right\}_{i=1}^n$.

    The metric matrix is defined by evaluating the metric on these basis
    vectors:

    $$ g_{ij}(q) = g_p \left( \frac{\partial}{\partial q^i},
    \frac{\partial}{\partial q^j} \right). $$

    This gives an n \times n matrix

    $$
    g(q) = \begin{pmatrix}
        g_{11}(q) & \cdots & g_{1n}(q) \\
        \vdots & \ddots & \vdots \\
        g_{n1}(q) & \cdots & g_{nn}(q)
        \end{pmatrix}.
    $$

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.metrics as cxm
    >>> import unxt as u

    >>> metric = cxm.EuclideanMetric(3)
    >>> q = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
    >>> metric.metric_matrix(cxc.cart2d, q)

    """

    @property
    def ndim(self) -> int:
        """Return the dimension of the metric (inferred from the chart)."""
        return len(self.signature)

    @property
    @abc.abstractmethod
    def signature(self) -> tuple[int, ...]:
        """Return the signature of the metric as a tuple of integers."""
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def metric_matrix(self, chart: cxc.AbstractChart[Any, Any], q: CDict, /) -> Array:
        """Return the metric matrix evaluated at position ``q``.

        Parameters
        ----------
        chart
            Chart whose components index the metric basis.
        q
            Coordinate values at which the metric is evaluated.

        Returns
        -------
        Array
            Metric matrix ``g_{ij}`` in the chart basis for ``chart``.

        """
        raise NotImplementedError

    def norm(
        self,
        chart: cxc.AbstractChart[Any, Any],
        v: CDict,
        /,
        *,
        at: CDict | None = None,
    ) -> u.AbstractQuantity:
        """Compute the norm of a vector with respect to this metric.

        Parameters
        ----------
        chart
            Chart in which the vector components are expressed.
        v
            Vector components as a coordinate dictionary.
        at
            Optional position at which to evaluate the metric (required for
            non-Euclidean metrics).

        Returns
        -------
        AbstractQuantity
            Norm of the vector with respect to this metric.

        Notes
        -----
        - For Euclidean metrics, the norm is independent of position and ``at``
          is ignored.
        - For non-Euclidean metrics, the norm depends on the position through
          the metric tensor evaluated at ``at``.

        """
        return api.norm(self, chart, v, at=at)
