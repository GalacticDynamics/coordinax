"""Typed metric matrix representations.

These types encapsulate the result of evaluating a metric field at a specific
``(manifold, point, chart)`` triple via the ``metric_matrix`` dispatch API.
They encode the sparsity structure (diagonal vs. dense) and provide operations
consistent with that structure.
"""

__all__ = ("AbstractMetricMatrix", "DiagonalMetric", "DenseMetric")

import abc
import functools as ft
import operator

from jaxtyping import Array
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import quax

import quaxed.numpy as qnp
import unxt as u

from coordinax.internal import (
    QMatrix,
    UnitsMatrix,
    det as _det_primitive,
    inv as _inv_primitive,
)

_det = quax.quaxify(_det_primitive)
_inv = quax.quaxify(_inv_primitive)
_matmul = quax.quaxify(jnp.matmul)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sine_product_diagonal(thetas: Array, scale: Any, /) -> Array:
    r"""Cumulative-product diagonal for the round-sphere metric.

    Given polar angles $\theta_1, \dots, \theta_k$ and a scale factor
    $R$ (e.g. the sphere radius or the radial coordinate $r$), returns the
    length-$(k+1)$ diagonal

    .. math::

        \bigl[R^2,\;
              R^2\sin^2\theta_1,\;
              R^2\sin^2\theta_1\sin^2\theta_2,\;
              \ldots\bigr]

    This is shared by the :class:`~coordinax._src.spherical.manifold.HyperSphereSn`
    and :class:`~coordinax._src.euclidean.manifold.EuclideanManifold` +
    ``HyperSphericalChart`` dispatch rules.

    Parameters
    ----------
    thetas : Array, shape ``(k,)``
        Polar angles $\theta_1, \dots, \theta_k$ in radians (the *last*
        azimuthal angle $\phi$ is excluded by the caller).
    scale : scalar
        Scale factor $R$ (sphere radius) or $r$ (radial coordinate).

    Returns
    -------
    Array, shape ``(k+1,)``
        The metric diagonal.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinax._src.metric.matrix import _sine_product_diagonal

    Unit sphere, S² (one polar angle θ = π/2):

    >>> _sine_product_diagonal(jnp.array([jnp.pi / 2]), 1.0)
    Array([1., 1.], dtype=float64)

    Radius-2 sphere, S² (θ = π/6):

    >>> import jax
    >>> _sine_product_diagonal(jnp.array([jnp.pi / 6]), 2.0)
    Array([4., 1.], dtype=float64)

    """
    sin2 = qnp.sin(thetas) ** 2
    cumprod = jnp.concat([jnp.ones(1, dtype=sin2.dtype), jnp.cumprod(sin2)])
    return scale**2 * cumprod


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AbstractMetricMatrix(eqx.Module):
    """Abstract base class for typed metric matrix representations.

    Concrete subclasses encode the sparsity structure of a metric matrix
    (diagonal vs. dense) and provide matrix-level operations consistent with
    that structure.
    """

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """Dimension of the metric."""
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def to_dense(self) -> "DenseMetric":
        """Return an equivalent :class:`DenseMetric`.

        For diagonal metrics, off-diagonal entries are zero.
        For dense metrics, returns ``self``.
        """
        raise NotImplementedError  # pragma: no cover


# ---------------------------------------------------------------------------
# Diagonal metric
# ---------------------------------------------------------------------------


@final
class DiagonalMetric(AbstractMetricMatrix):
    r"""Diagonal metric matrix stored as a 1-D array or QMatrix.

    Encodes a metric whose coordinate matrix is diagonal — i.e. orthogonal
    coordinate charts.  Storing only the diagonal avoids materialising the full
    $n \times n$ matrix and makes operations like matrix-vector products and
    inversion run in $O(n)$.

    Parameters
    ----------
    diagonal : QMatrix or Array, shape ``(n,)``
        The diagonal entries $g_{11}, g_{22}, \\ldots, g_{nn}$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> d = DiagonalMetric(jnp.array([1.0, 4.0, 9.0]))
    >>> d.ndim
    3
    >>> d.determinant
    Array(36., dtype=float64)
    >>> d.inverse.diagonal
    Array([1.        , 0.25      , 0.11111111], dtype=float64)

    """

    diagonal: QMatrix | Array

    @property
    def ndim(self) -> int:
        """Dimension of the metric."""
        return int(self.diagonal.shape[-1])

    def to_dense(self) -> "DenseMetric":
        r"""Convert to a full $n \times n$ matrix with zeros off the diagonal.

        When the diagonal is a :class:`~coordinax.internal.QMatrix`,
        the off-diagonal entry ``(i, j)`` is assigned the geometric-mean unit
        ``sqrt(diag_unit[i] * diag_unit[j])``.  This choice ensures that
        ``g[i, j] * v[j]`` is unit-compatible with ``g[i, i] * v[i]`` during
        matrix-vector contraction, which is required for the
        :func:`~coordinax.internal.QMatrix` dot-product to succeed even
        when the coordinate components have different physical dimensions (e.g.
        metres and radians in spherical coordinates).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from coordinax._src.metric.matrix import DiagonalMetric

        Plain array — off-diagonal entries are zero:

        >>> d = DiagonalMetric(jnp.array([1.0, 4.0]))
        >>> d.to_dense().matrix
        Array([[1., 0.],
               [0., 4.]], dtype=float64)

        QMatrix diagonal — diagonal units are preserved and off-diagonal
        entries get the geometric-mean unit:

        >>> from coordinax.internal import QMatrix
        >>> d = DiagonalMetric(QMatrix(jnp.array([1.0, 4.0]), unit=("m2", "s2")))
        >>> d.to_dense().matrix.unit[0, 0]
        Unit("m2")
        >>> d.to_dense().matrix.unit[1, 1]
        Unit("s2")
        >>> d.to_dense().matrix.unit[0, 1]  # geometric mean: sqrt(m2 * s2)
        Unit("m s")

        """
        if isinstance(self.diagonal, QMatrix):
            # Off-diagonal entries are numerically zero, but their units must
            # be chosen so that  g[i,j] * v[j]  is unit-compatible with
            # g[i,i] * v[i]  for any tangent vector v.  The physically correct
            # unit for entry (i, j) of a metric tensor g is
            #
            #   [g_{ij}] = [ds²] / ([coord_i] · [coord_j])
            #             = sqrt([g_{ii}] · [g_{jj}])
            #
            # Using the geometric mean for off-diagonal entries ensures that
            # the scale-factor computation in _dot_general_2d_1d can always
            # convert every term to the reference unit ref[i] = g[i,0]*v[0].
            n = self.ndim
            dense_val = jnp.diag(self.diagonal.value)
            du = self.diagonal.unit._units  # shape (n,)
            row_units = tuple(
                tuple(du[i] if i == j else (du[i] * du[j]) ** 0.5 for j in range(n))
                for i in range(n)
            )
            return DenseMetric(QMatrix(dense_val, unit=UnitsMatrix(row_units)))
        return DenseMetric(jnp.diag(self.diagonal))

    def __matmul__(
        self, other: "Array | QMatrix | u.AbstractQuantity", /
    ) -> "Array | QMatrix | u.AbstractQuantity":
        """Apply this diagonal metric to a vector — element-wise product.

        When either the diagonal or ``other`` carries units, the operation is
        routed through :meth:`to_dense` so that unit propagation is handled
        correctly by the :class:`~coordinax.internal.QMatrix` Quax
        dispatches.  Plain-array inputs use a fast O(n) element-wise multiply.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from coordinax._src.metric.matrix import DiagonalMetric

        Plain array diagonal, plain array vector:

        >>> d = DiagonalMetric(jnp.array([1.0, 4.0, 9.0]))
        >>> d @ jnp.array([1.0, 2.0, 3.0])
        Array([ 1.,  8., 27.], dtype=float64)

        QMatrix diagonal, plain array vector — result carries diagonal units:

        >>> from coordinax.internal import QMatrix
        >>> d = DiagonalMetric(
        ...     QMatrix(jnp.array([2.0, 3.0]), unit=("m2 / rad2", "m2 / rad2"))
        ... )
        >>> w = d @ jnp.array([1.0, 1.0])
        >>> w.unit.to_string()
        '(m2 / rad2, m2 / rad2)'
        >>> w.value
        Array([2., 3.], dtype=float64)

        QMatrix diagonal, Quantity vector — full unit tracking:

        >>> import unxt as u
        >>> w2 = d @ u.Q(jnp.array([1.0, 1.0]), "rad")
        >>> w2.unit.to_string()
        '(m2 / rad, m2 / rad)'
        >>> w2.value
        Array([2., 3.], dtype=float64)

        QMatrix diagonal, QMatrix vector — full unit tracking:

        >>> v = QMatrix(jnp.array([1.0, 1.0]), unit=("rad", "rad"))
        >>> w3 = d @ v
        >>> w3.unit.to_string()
        '(m2 / rad, m2 / rad)'
        >>> w3.value
        Array([2., 3.], dtype=float64)

        """
        if isinstance(self.diagonal, QMatrix) or isinstance(
            other, (QMatrix, u.AbstractQuantity)
        ):
            # Route through the dense path for correct unit propagation.
            return self.to_dense().__matmul__(other)
        # Fast O(n) path: plain-array element-wise multiply.
        return self.diagonal * other

    @property
    def inverse(self) -> "DiagonalMetric":
        """Inverse diagonal metric — reciprocal of each diagonal entry.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from coordinax._src.metric.matrix import DiagonalMetric

        >>> d = DiagonalMetric(jnp.array([2.0, 4.0]))
        >>> d.inverse.diagonal
        Array([0.5 , 0.25], dtype=float64)

        """
        if isinstance(self.diagonal, QMatrix):
            inv_vals = 1.0 / self.diagonal.value
            return DiagonalMetric(QMatrix(inv_vals, unit=self.diagonal.unit.inverse()))
        return DiagonalMetric(1.0 / self.diagonal)

    @property
    def determinant(self) -> "Array | u.AbstractQuantity":
        """Product of the diagonal entries.

        Returns a :class:`~unxt.AbstractQuantity` when the diagonal is a
        :class:`~coordinax.internal.QMatrix`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from coordinax._src.metric.matrix import DiagonalMetric

        Bare array — returns a plain :class:`~jaxtyping.Array`:

        >>> DiagonalMetric(jnp.array([2.0, 3.0])).determinant
        Array(6., dtype=float64)

        QMatrix diagonal — returns a :class:`~unxt.Quantity`:

        >>> import unxt as u
        >>> from coordinax.internal import QMatrix
        >>> d = DiagonalMetric(QMatrix(jnp.array([2.0, 3.0]), unit=("m2", "s2")))
        >>> d.determinant
        Q(6., 'm2 s2')

        """
        if isinstance(self.diagonal, QMatrix):
            det_val = qnp.prod(self.diagonal.value)
            det_unit = ft.reduce(operator.mul, self.diagonal.unit)
            return u.Q(det_val, det_unit)
        return qnp.prod(self.diagonal)


# ---------------------------------------------------------------------------
# Dense metric
# ---------------------------------------------------------------------------


@final
class DenseMetric(AbstractMetricMatrix):
    r"""Dense symmetric metric matrix.

    Stores the full $n \\times n$ metric matrix.  Used for non-orthogonal
    charts or metrics that cannot be expressed diagonally.

    Parameters
    ----------
    matrix : QMatrix or Array, shape ``(n, n)``
        The full metric matrix $g_{ij}$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinax._src.metric.matrix import DenseMetric

    >>> g = DenseMetric(jnp.eye(3))
    >>> g.ndim
    3
    >>> g.determinant
    Array(1., dtype=float64)

    """

    matrix: QMatrix | Array

    @property
    def ndim(self) -> int:
        """Dimension of the metric."""
        return int(self.matrix.shape[-1])

    def to_dense(self) -> "DenseMetric":
        """Return ``self`` — already in dense form.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from coordinax._src.metric.matrix import DenseMetric

        >>> g = DenseMetric(jnp.eye(2))
        >>> g.to_dense() is g
        True

        """
        return self

    def __matmul__(self, other: "Array | QMatrix", /) -> "Array | QMatrix":
        """Apply this metric matrix to a vector via matrix-vector product.

        When the metric matrix is a :class:`~coordinax.internal.QMatrix`,
        a plain-array ``other`` is treated as dimensionless so that units flow
        through the contraction correctly.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from coordinax._src.metric.matrix import DenseMetric

        Plain array metric, plain array vector:

        >>> g = DenseMetric(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
        >>> g @ jnp.array([1.0, 1.0])
        Array([2., 3.], dtype=float64)

        QMatrix metric, plain array vector — result carries metric units:

        >>> from coordinax.internal import QMatrix, UnitsMatrix
        >>> g = DenseMetric(
        ...     QMatrix(
        ...         jnp.array([[2.0, 0.0], [0.0, 3.0]]),
        ...         unit=UnitsMatrix((
        ...             ("m2 / rad2", "m2 / rad2"),
        ...             ("m2 / rad2", "m2 / rad2"),
        ...         )),
        ...     )
        ... )
        >>> w = g @ jnp.array([1.0, 1.0])
        >>> w.unit.to_string()
        '(m2 / rad2, m2 / rad2)'
        >>> w.value
        Array([2., 3.], dtype=float64)

        QMatrix metric, QMatrix vector — full unit tracking:

        >>> v = QMatrix(jnp.array([1.0, 1.0]), unit=("rad / s", "rad / s"))
        >>> w2 = g @ v
        >>> w2.unit.to_string()
        '(m2 / (rad s), m2 / (rad s))'

        """
        return _matmul(self.matrix, other)

    @property
    def inverse(self) -> "DenseMetric":
        """Inverse via :func:`jax.numpy.linalg.inv` (positive-definite assumption).

        Returns a :class:`~coordinax.internal.QMatrix`-backed
        :class:`DenseMetric` with units ``1 / ref_unit`` when the matrix
        carries units.  Assumes all entries share the same unit (physically
        well-formed metrics from the Cartesian-Jacobian pullback always satisfy
        this).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from coordinax._src.metric.matrix import DenseMetric

        Bare array:

        >>> g = DenseMetric(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
        >>> g.inverse.matrix
        Array([[0.5 , 0.  ],
               [0.  , 0.25]], dtype=float64)

        QMatrix — inverse carries reciprocal units:

        >>> import unxt as u
        >>> from coordinax.internal import QMatrix, UnitsMatrix
        >>> g = DenseMetric(
        ...     QMatrix(
        ...         jnp.array([[4.0, 0.0], [0.0, 1.0]]),
        ...         unit=UnitsMatrix((
        ...             ("m2 / rad2", "m2 / rad2"),
        ...             ("m2 / rad2", "m2 / rad2"),
        ...         )),
        ...     )
        ... )
        >>> g.inverse.matrix.unit[0, 0]
        Unit("rad2 / m2")
        >>> g.inverse.matrix.value
        Array([[0.25, 0.  ],
               [0.  , 1.  ]], dtype=float64)

        """
        return DenseMetric(_inv(self.matrix))

    @property
    def determinant(self) -> "Array | u.AbstractQuantity":
        """Determinant via the custom ``det_p`` JAX primitive.

        Routes through Quax, so a :class:`~coordinax.internal.QMatrix`
        matrix returns a :class:`~unxt.AbstractQuantity` while a plain array
        returns a bare :class:`~jaxtyping.Array`.  The unit is the product of
        the main-diagonal units — valid for diagonal and uniform-unit matrices.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from coordinax._src.metric.matrix import DenseMetric

        Bare array — returns a plain :class:`~jaxtyping.Array`:

        >>> DenseMetric(jnp.eye(3)).determinant
        Array(1., dtype=float64)

        QMatrix — returns a :class:`~unxt.Quantity`:

        >>> import unxt as u
        >>> from coordinax.internal import QMatrix
        >>> g = DenseMetric(QMatrix(jnp.eye(2), unit=(("m2", ""), ("", "s2"))))
        >>> g.determinant
        Q(1., 'm2 s2')

        """
        return _det(self.matrix)
