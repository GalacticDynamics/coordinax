"""Manifold definitions and manifold inference helpers."""

__all__ = ("AbstractMetric",)

import abc

from jaxtyping import Array, Bool
from typing import Any

import jax
import jax.numpy as jnp

import coordinax.angles as cxa
import coordinax.api.manifolds as cxmapi
import coordinax.charts as cxc
from .custom_types import CDict, OptUSys
from coordinax.internal import QuantityMatrix, UnitsMatrix


@jax.tree_util.register_static
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
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> metric = cxm.EuclideanMetric(3)
    >>> at = {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"),
    ...       "z": u.Q(0.0, "km")}
    >>> metric.metric_matrix(cxc.cart3d, at=at)
    QuantityMatrix([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]], '((, , ), (, , ), (, , ))')

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
    def metric_matrix(
        self,
        chart: cxc.AbstractChart,
        /,
        *,
        at: CDict,
        usys: OptUSys = None,
    ) -> QuantityMatrix | Array:
        r"""Compute the metric tensor $g_{ij}$ at base point ``at``.

        Parameters
        ----------
        chart : AbstractChart
            The coordinate chart in which to express the metric.
        at : CDict
            Base point (component dict in ``chart``) at which to evaluate.
        usys : OptUSys, optional
            Unit system to use for the metric evaluation.

        Returns
        -------
        QuantityMatrix, shape (n, n)
            Symmetric positive-definite metric matrix ``g_{ij}`` in the chart
            basis for ``chart``.

        """
        raise NotImplementedError  # pragma: no cover

    def scale_factors(
        self,
        chart: cxc.AbstractChart[Any, Any],
        /,
        *,
        at: CDict,
        usys: OptUSys = None,
    ) -> QuantityMatrix:
        r"""Return the diagonal entries of the metric matrix in ``chart`` at ``at``.

        This is a thin convenience wrapper over
        ``cxmapi.scale_factors(self, chart, at=at, usys=usys)``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        In Cartesian coordinates for Euclidean space, the diagonal entries are all 1:

        >>> metric = cxm.EuclideanMetric(3)
        >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        >>> gdiag = metric.scale_factors(cxc.cart3d, at=at)
        >>> gdiag
        QuantityMatrix([1., 1., 1.], '(, , )')

        In spherical coordinates on Euclidean space, the entries depend on the
        base point:

        >>> at_sph = {
        ...     "r": u.Q(2.0, "km"),
        ...     "theta": u.Angle(jnp.pi / 2, "rad"),
        ...     "phi": u.Angle(0.0, "rad"),
        ... }
        >>> metric.scale_factors(cxc.sph3d, at=at_sph)
        QuantityMatrix([1., 4., 4.], '(, km2 / rad2, km2 / rad2)')

        """
        return cxmapi.scale_factors(self, chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]

    def cholesky(
        self,
        chart: cxc.AbstractChart[Any, Any],
        /,
        *,
        at: CDict,
        usys: OptUSys = None,
    ) -> QuantityMatrix | Array:
        r"""Return the lower-triangular Cholesky factor $L$ of the metric matrix.

        Computes the factorization $g = L\,L^\top$ where $L$ is the unique
        lower-triangular matrix with strictly positive diagonal entries.
        With the convention used here, the vielbein is $E = L^\top$; use
        ``L.value.T`` (or plain ``.T`` for a bare array) to obtain it.

        Parameters
        ----------
        chart : AbstractChart
            The coordinate chart in which to express the metric.
        at : CDict
            Base point (component dict in ``chart``) at which to evaluate.
        usys : OptUSys, optional
            Unit system to use for the metric evaluation.

        Returns
        -------
        QuantityMatrix or Array, shape (n, n)
            Lower-triangular Cholesky factor $L$ satisfying $g = L\,L^\top$.
            Returns a ``QuantityMatrix`` when the metric matrix carries units;
            returns a plain ``Array`` otherwise.  The unit of element
            $L_{ij}$ is $\sqrt{u_{ij}}$ where $u_{ij}$ is the unit of $g_{ij}$.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        Euclidean metric in Cartesian coordinates — Cholesky of the identity:

        >>> metric = cxm.EuclideanMetric(3)
        >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        >>> metric.cholesky(cxc.cart3d, at=at)
        QuantityMatrix([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]], '((, , ), (, , ), (, , ))')

        Euclidean metric in spherical coordinates — Cholesky is diag(1, r, r sinθ):

        >>> at = {
        ...     "r": u.Q(2.0, "m"),
        ...     "theta": u.Angle(jnp.pi / 2, "rad"),
        ...     "phi": u.Angle(0.0, "rad"),
        ... }
        >>> metric.cholesky(cxc.sph3d, at=at)
        QuantityMatrix(
            [[1., 0., 0.],
             [0., 2., 0.],
             [0., 0., 2.]],
            '((, m(1/2) / rad(1/2), m(1/2) / rad(1/2)), ...)'
        )

        """
        G = self.metric_matrix(chart, at=at, usys=usys)
        if isinstance(G, QuantityMatrix):
            l_units = UnitsMatrix(G.unit._units**0.5)
            return QuantityMatrix(jnp.linalg.cholesky(G.value), unit=l_units)
        return jnp.linalg.cholesky(G)

    def angle_between(
        self,
        chart: cxc.AbstractChart[Any, Any],
        uvec: CDict,
        vvec: CDict,
        /,
        *,
        at: CDict,
        usys: OptUSys = None,
    ) -> cxa.AbstractAngle:
        r"""Return the metric angle between two tangent vectors.

        This is a thin convenience wrapper over
        ``cxmapi.angle_between(self, chart, uvec, vvec, at=at, usys=usys)``.
        """
        return cxmapi.angle_between(self, chart, uvec, vvec, at=at, usys=usys)  # ty: ignore[invalid-return-type]

    def is_diagonal(
        self, chart: cxc.AbstractChart[Any, Any], /, *, at: CDict, usys: OptUSys = None
    ) -> Bool[Array, ""]:
        r"""Return ``True`` if the metric matrix is diagonal at base point ``at``.

        A metric is diagonal when all off-diagonal entries $g_{ij}$ with
        $i \neq j$ are numerically zero (within floating-point tolerance,
        checked via ``jnp.allclose``), i.e. the coordinate basis is
        orthogonal at ``at``.

        Parameters
        ----------
        chart : AbstractChart
            The coordinate chart in which to evaluate the metric.
        at : CDict
            Base point at which to check the metric matrix.
        usys : OptUSys, optional
            Unit system for the evaluation.

        Returns
        -------
        bool
            ``True`` if all off-diagonal metric components vanish.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        Euclidean metric in spherical coordinates is diagonal (orthogonal chart):

        >>> metric = cxm.EuclideanMetric(3)
        >>> at = {"r": u.Q(3.0, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.0, "rad")}
        >>> metric.is_diagonal(cxc.sph3d, at=at)
        Array(True, dtype=bool)

        """
        G = self.metric_matrix(chart, at=at, usys=usys)
        val = G.value if hasattr(G, "value") else G
        off_diagonal = jnp.subtract(val, jnp.diag(jnp.diag(val)))
        return jnp.allclose(off_diagonal, 0)
