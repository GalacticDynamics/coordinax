"""Manifold definitions and manifold inference helpers."""

__all__ = ("AbstractMetricField", "AbstractDiagonalMetricField")

import abc

from typing import Any

import jax

import coordinax.api.manifolds as cxmapi
from coordinax._src.custom_types import OptUSys


@jax.tree_util.register_static
class AbstractMetricField(metaclass=abc.ABCMeta):
    r"""Abstract base class for metrics of manifolds.

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

    The metric matrix is computed via the standalone dispatch function
    :func:`coordinax.manifolds.metric_matrix`.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> cxm.FlatMetric(3).ndim
    3

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

    def norm(
        self, v: Any, *args: Any, at: Any, usys: OptUSys = None, **kwargs: Any
    ) -> Any:
        r"""Compute the norm $\|v\|_g = \sqrt{g(v, v)}$.

        Convenience wrapper that calls
        ``cxmapi.norm(v, self, chart, at=at, usys=usys)`` directly.  The
        ``chart`` must be passed as the second positional argument (after
        ``v``).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        >>> metric = cxm.FlatMetric(3)
        >>> at = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}

        With a CDict of quantities (usys optional):

        >>> v = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s"), "z": u.Q(0.0, "m/s")}
        >>> metric.norm(v, cxc.cart3d, at=at)
        Q(5., 'm / s')

        With a stacked ``jax.Array`` (usys required):

        >>> v = jnp.array([3.0, 4.0, 0.0])
        >>> metric.norm(v, cxc.cart3d, at=at, usys=u.unitsystems.si)
        Array(5., dtype=float64)

        """
        return cxmapi.norm(v, self, *args, at=at, usys=usys, **kwargs)


class AbstractDiagonalMetricField(AbstractMetricField):
    r"""Abstract base class for metrics whose matrix is diagonal.

    A metric is **diagonal** (equivalently, the coordinate chart is an
    **orthogonal coordinate system**) when all off-diagonal entries of the
    metric matrix vanish at every base point in every chart in the metric's
    diagonal domain:

    $$g_{ij}(p) = 0 \quad \text{for } i \neq j.$$

    The coordinate basis vectors $\partial/\partial q^i$ are then mutually
    orthogonal. The diagonal entries $g_{ii}(p)$ give the squared scale factors

    $$h_i(p)^2 = g_{ii}(p),$$

    and the infinitesimal line element simplifies to

    $$ds^2 = \sum_i g_{ii}(q)\,(dq^i)^2.$$

    **Role: structural marker, not behavioral interface.**

    This class adds no new abstract methods beyond those of `AbstractMetricField`.
    Its purpose is to declare that the ``metric_matrix`` dispatch
    **must** return a :class:`~coordinax._src.metric.matrix.DiagonalMetric`
    at every valid base point for charts where this metric is used as diagonal
    (typically orthogonal charts).

    In particular, manifold/atlas chart membership (for example, ``has_chart``)
    is a broader structural notion and does not, by itself, imply orthogonality
    or diagonality.

    See Also
    --------
    FlatMetric : flat Riemannian metric on $\mathbb{R}^n$; in Cartesian
        charts $g = I_n$; in orthogonal curvilinear charts computed by Jacobian
        pullback $g = J^\top J$.
    RoundMetric : round metric on $S^{n-1}$ in the intrinsic
        hyperspherical chart; diagonal entries follow the cumulative-sine rule
        $g_{kk} = \prod_{j < k} \sin^2\!\theta_j$.
    MinkowskiMetric : Lorentzian pseudo-Riemannian metric
        $\eta = \operatorname{diag}(-1, 1, 1, 1)$ on Minkowski spacetime;
        diagonal in the canonical Cartesian spacetime chart.

    Examples
    --------
    >>> import coordinax.manifolds as cxm

    ``FlatMetric`` is an ``AbstractDiagonalMetricField``:

    >>> isinstance(cxm.FlatMetric(3), AbstractDiagonalMetricField)
    True

    ``MinkowskiMetric`` is also an ``AbstractDiagonalMetricField``:

    >>> isinstance(cxm.MinkowskiMetric(), AbstractDiagonalMetricField)
    True

    General (non-diagonal) metrics such as ``PullbackMetric`` are not:

    >>> import unxt as u
    >>> isinstance(
    ...     cxm.PullbackMetric(
    ...         cxm.TwoSphereIn3D(radius=u.Q(1.0, "m")),
    ...         cxm.FlatMetric(3),
    ...     ),
    ...     AbstractDiagonalMetricField,
    ... )
    False

    """
