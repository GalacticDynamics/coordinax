"""Intrinsic metric field types.

A *metric field* is a smooth family of inner-product structures on the tangent
spaces of a manifold.  It describes the *kind* of geometry — flat, round,
Lorentzian, etc. — without referencing a particular coordinate chart or
producing a matrix of components.  The latter is the responsibility of
:class:`~coordinax._src.metric.matrix.AbstractMetricMatrix` together with the
``metric_matrix`` dispatch API.
"""

__all__ = ("AbstractMetricField", "AbstractDiagonalMetricField", "RoundMetric")

import abc

from typing import final

import equinox as eqx
import jax.tree_util as jtu

import unxt as u


@jtu.register_static
class AbstractMetricField(metaclass=abc.ABCMeta):
    r"""Abstract base class for intrinsic metric fields.

    A metric field associates each point $p$ of a manifold $M$ with an
    inner-product $g_p$ on the tangent space $T_p M$.  Subclasses encode the
    *kind* of geometry (flat, round, Lorentzian, …) without specifying a
    coordinate representation.

    Subclasses must implement the :attr:`signature` property.

    Notes
    -----
    Concrete parameter-free subclasses are registered as static JAX pytree
    leaves via ``@jax.tree_util.register_static``.  Subclasses that carry
    JAX-traced parameters (e.g. :class:`RoundMetric` with its ``radius``) are
    :class:`equinox.Module` pytrees instead.

    """

    @property
    def ndim(self) -> int:
        """Intrinsic dimension — length of :attr:`signature`."""
        return len(self.signature)

    @property
    @abc.abstractmethod
    def signature(self) -> tuple[int, ...]:
        """Metric signature as a tuple of ±1 values.

        Returns ``(1, 1, ..., 1)`` for Riemannian metrics and includes ``-1``
        entries for pseudo-Riemannian (Lorentzian) metrics.
        """
        raise NotImplementedError  # pragma: no cover


@jtu.register_static
class AbstractDiagonalMetricField(AbstractMetricField, metaclass=abc.ABCMeta):
    """Structural marker for metric fields that are diagonal in their natural chart.

    Subclassing :class:`AbstractDiagonalMetricField` signals that there exists
    at least one coordinate chart in which the metric matrix is diagonal.  The
    ``metric_matrix`` dispatch rules for such charts can therefore return a
    :class:`~coordinax._src.metric.matrix.DiagonalMetric` instead of a dense
    matrix.
    """


@final
class RoundMetric(AbstractDiagonalMetricField, eqx.Module):
    r"""Constant positive-curvature (round) metric on the *n*-sphere $S^n$.

    The geometric radius ``radius`` sets the overall scale: the metric is
    $g = R^2 \, \hat{g}$ where $\hat{g}$ is the round metric on the unit sphere.

    Unlike the other :class:`AbstractMetricField` subtypes, :class:`RoundMetric`
    is an :class:`equinox.Module` rather than a ``register_static`` dataclass.
    This means ``radius`` is a *dynamic* JAX leaf — it can be JIT-compiled,
    differentiated with :func:`jax.grad`, and batched with :func:`jax.vmap`.

    Parameters
    ----------
    ndim : int
        Intrinsic dimension of the sphere (e.g. 2 for $S^2$).  Stored as a
        static equinox field (part of the treedef, not a JAX array).
    radius : unxt.AbstractQuantity
        Geometric radius.  Stored as a dynamic equinox field (JAX leaf).

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax._src.metric.field import RoundMetric

    >>> m = RoundMetric(ndim=2, radius=u.Q(1.0, "m"))
    >>> m.ndim
    2
    >>> m.signature
    (1, 1)
    >>> m.radius
    Q(1., 'm')

    The radius is a JAX leaf — the pytree contains it as a dynamic leaf:

    >>> import jax
    >>> leaves, treedef = jax.tree_util.tree_flatten(m)
    >>> len(leaves)  # only radius is dynamic
    1

    """

    # NOTE: the field is named _ndim (not ndim) to avoid conflicting with the
    # `ndim` property inherited from AbstractMetricField.  The public interface
    # still accepts `ndim` via the custom __init__.
    _ndim: int = eqx.field(static=True)
    radius: u.AbstractQuantity
    """Geometric radius (dynamic JAX leaf — JIT/grad/vmap-friendly)."""

    def __init__(self, *, ndim: int, radius: u.AbstractQuantity) -> None:
        object.__setattr__(self, "_ndim", ndim)
        object.__setattr__(self, "radius", radius)

    @property
    def ndim(self) -> int:
        """Intrinsic dimension of the sphere (static — part of the treedef)."""
        return self._ndim

    @property
    def signature(self) -> tuple[int, ...]:
        """All-positive signature ``(1, 1, ..., 1)`` for a round sphere."""
        return (1,) * self._ndim
