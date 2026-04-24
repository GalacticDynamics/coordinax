"""Manifold definitions and manifold inference helpers."""

__all__ = ("AbstractDiagonalMetric",)

from jaxtyping import Array, Bool
from typing import Any

import jax.numpy as jnp

import coordinax.charts as cxc
from .base import AbstractMetric
from coordinax.internal.custom_types import CDict, OptUSys


class AbstractDiagonalMetric(AbstractMetric):
    r"""Abstract base class for metrics whose matrix is diagonal.

    A metric is **diagonal** (equivalently, the coordinate chart is an
    **orthogonal coordinate system**) when all off-diagonal entries of the
    metric matrix vanish at every base point:

    $$g_{ij}(p) = 0 \quad \text{for } i \neq j.$$

    The coordinate basis vectors $\partial/\partial q^i$ are then mutually
    orthogonal. The diagonal entries $g_{ii}(p)$ give the squared scale
    factors

    $$h_i(p)^2 = g_{ii}(p),$$

    and the infinitesimal line element simplifies to

    $$ds^2 = \sum_i g_{ii}(q)\,(dq^i)^2.$$

    This class is a **structural marker**: it adds no new abstract methods
    beyond those of `AbstractMetric`, but declares that ``metric_matrix``
    **must** return a diagonal matrix (all off-diagonal entries zero) at every
    valid base point. This structural guarantee allows dispatch-level
    specialisations — for example computing ``scale_factors`` directly from
    the diagonal without constructing the full $n \times n$ matrix.

    Notes
    -----
    Subclasses must implement the two abstract members inherited from
    `AbstractMetric`:

    - ``signature`` (property): a tuple of $\pm 1$ of length ``ndim``
      encoding the metric signature. Positive entries denote Riemannian
      (space-like) directions; a single ``-1`` entry denotes a time-like
      direction (Lorentzian signature).
    - ``metric_matrix(chart, /, *, at, usys=None)`` (method): must return a
      diagonal ``QuantityMatrix`` (or plain ``Array``) of shape
      ``(ndim, ndim)`` with all off-diagonal entries exactly zero.

    Concrete subclasses must be immutable frozen dataclasses and registered as
    static JAX PyTree nodes via ``@jax.tree_util.register_static``.

    `AbstractMetric.is_diagonal` inspects the matrix at a **specific base
    point** and returns a ``bool``. ``AbstractDiagonalMetric`` makes this an
    unconditional **structural promise** across all base points: instances are
    always diagonal regardless of the chart or base point.

    See Also
    --------
    EuclideanMetric : flat Riemannian metric on $\mathbb{R}^n$; in Cartesian
        charts $g = I_n$; in orthogonal curvilinear charts computed by
        Jacobian pullback $g = J^\top J$.
    HyperSphericalMetric : round metric on $S^{n-1}$ in the intrinsic
        hyperspherical chart; diagonal entries follow the cumulative-sine
        rule $g_{kk} = \prod_{j < k} \sin^2\!\theta_j$.
    MinkowskiMetric : Lorentzian pseudo-Riemannian metric
        $\eta = \operatorname{diag}(-1, 1, 1, 1)$ on Minkowski spacetime;
        diagonal in the canonical Cartesian spacetime chart.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> from coordinax.manifolds._src.diagonal import AbstractDiagonalMetric

    ``EuclideanMetric`` is an ``AbstractDiagonalMetric``:

    >>> isinstance(cxm.EuclideanMetric(3), AbstractDiagonalMetric)
    True

    ``MinkowskiMetric`` is also an ``AbstractDiagonalMetric``:

    >>> isinstance(cxm.MinkowskiMetric(), AbstractDiagonalMetric)
    True

    General (non-diagonal) metrics such as ``InducedMetric`` are not:

    >>> import unxt as u
    >>> isinstance(
    ...     cxm.InducedMetric(
    ...         cxm.TwoSphereIn3D(radius=u.Q(1.0, "m")),
    ...         cxm.EuclideanMetric(3),
    ...     ),
    ...     AbstractDiagonalMetric,
    ... )
    False

    """

    def is_diagonal(
        self, chart: cxc.AbstractChart[Any, Any], /, *, at: CDict, usys: OptUSys = None
    ) -> Bool[Array, ""]:
        r"""Return ``True`` if the metric matrix is diagonal at base point ``at``.

        A metric is diagonal when all off-diagonal entries $g_{ij}$ with
        $i \neq j$ are exactly zero, i.e. the coordinate basis is
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
        del chart, at, usys
        return jnp.ones((), dtype=bool)
