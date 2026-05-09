"""Manifold definitions and manifold inference helpers."""

__all__ = ("AbstractTopologicalManifold", "NoManifold", "no_manifold")

import abc

from typing import Any

import jax.tree_util as jtu
import wadler_lindig as wl

import dataclassish


@jtu.register_static
class AbstractTopologicalManifold(metaclass=abc.ABCMeta):
    r"""Abstract base for topological manifold objects in *coordinax*.

    A **topological manifold** of dimension $n$ is a topological space $M$
    satisfying three axioms:

    - **Hausdorff**: distinct points have disjoint open neighbourhoods.
    - **Second-countable**: the topology admits a countable basis.
    - **Locally Euclidean of dimension $n$**: every point has a neighbourhood
      homeomorphic to $\mathbb{R}^n$ via a chart map
      $\varphi : U \subset M \to \mathbb{R}^n$.

    At the topological level one can ask whether maps are continuous, but not
    whether they are differentiable. Differentiability — and therefore calculus,
    tangent vectors, and metrics — requires an additional smooth atlas, which is
    provided by concrete subclasses (e.g.
    {class}`~coordinax.manifolds.EuclideanManifold`).

    ``AbstractTopologicalManifold`` is the root base class for all manifold
    objects in `coordinax`. It exposes only the topological layer:

    - **dimension introspection** — the intrinsic dimension ``ndim``,
    - **chart membership** — ``has_chart`` and ``check_chart``.

    All geometric and numerical work is delegated to the concrete subclass and
    its associated atlas; this abstract class is a pure structural descriptor.

    Attributes
    ----------
    ndim : int
        Intrinsic dimension $n$ of the manifold.

    Notes
    -----
    - Manifold objects are **structural descriptors**: they carry no numerical
      point data.
    - Instances are registered with JAX as static pytree nodes via
      ``jax.tree_util.register_static``, so they can appear as compile-time
      metadata inside JIT-compiled functions.
    - Subclasses follow the abstract-final pattern: one abstract base, one
      ``@final`` concrete class — no intermediate inheritance layers.

    Examples
    --------
    ``AbstractTopologicalManifold`` cannot be instantiated directly; use a
    concrete subclass such as {class}`~coordinax.manifolds.EuclideanManifold`
    or {class}`~coordinax.manifolds.HyperSphericalManifold`.

    **Basic construction and introspection**

    The Euclidean 3-manifold $\mathbb{R}^3$:

    >>> import coordinax.manifolds as cxm
    >>> M = cxm.EuclideanManifold(3)
    >>> M
    EuclideanManifold(ndim=3)

    The intrinsic dimension:

    >>> M.ndim
    3

    **Chart membership**

    {meth}`has_chart` returns ``True`` when a chart belongs to the manifold's
    atlas:

    >>> import coordinax.charts as cxc
    >>> M.has_chart(cxc.cart3d)
    True

    >>> M.has_chart(cxc.sph3d)
    True

    Charts with the wrong dimensionality are rejected:

    >>> M.has_chart(cxc.cart2d)
    False

    {meth}`check_chart` raises {exc}`ValueError` for unsupported charts:

    >>> try:
    ...     M.check_chart(cxc.cart2d)
    ... except ValueError as e:
    ...     print(e)
    Chart Cart2D() is not supported by this manifold atlas.

    """

    ndim: int
    """Intrinsic dimension of the manifold."""

    @abc.abstractmethod
    def has_chart(self, chart: Any, /) -> bool:
        """Return whether ``chart`` belongs to this manifold atlas.

        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(2)
        >>> M.has_chart(cxc.cart2d)
        True
        >>> M.has_chart(cxc.cart3d)
        False

        """
        raise NotImplementedError  # pragma: no cover

    # =====================================================

    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        """Return the string representation.

        >>> import wadler_lindig as wl
        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(3)
        >>> wl.pprint(M)
        EuclideanManifold(ndim=3)

        """
        return wl.bracketed(
            begin=wl.TextDoc(f"{type(self).__name__}("),
            docs=wl.named_objs(list(dataclassish.field_items(self)), **kw),
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=4,
        )

    def __repr__(self) -> str:
        """Return the string representation."""
        return wl.pformat(self, width=88)

    def __str__(self) -> str:
        """Return the string representation."""
        return wl.pformat(self, width=88)


@jtu.register_static
class NoManifold(AbstractTopologicalManifold):
    """A degenerate placeholder manifold with no charts and no geometry.

    ``NoManifold`` is a sentinel value used when a manifold object is required
    by the API but none has been specified by the user.

    - ``ndim == -1`` signals "no manifold specified".
    - ``has_chart(chart)`` always returns ``False``.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> M = cxm.NoManifold()
    >>> M.ndim
    False
    >>> M.has_chart(cxc.cart2d)
    False

    """

    ndim: int = False
    """Stand-in dimension of the degenerate manifold."""

    def has_chart(self, chart: Any, /) -> bool:
        """Return whether ``chart`` belongs to this manifold atlas."""
        return hasattr(chart, "M") and isinstance(chart.M, NoManifold)


no_manifold = NoManifold()
"""Canonical instance of `coordinax.manifolds.NoManifold`."""
