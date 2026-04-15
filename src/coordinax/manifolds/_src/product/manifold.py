"""Product Manifolds."""

__all__ = ("CartesianProductManifold",)


import dataclasses

from typing import final
from typing_extensions import override

import jax

from .atlas import CartesianProductAtlas
from .metric import CartesianProductMetric
from coordinax.manifolds._src.base import AbstractManifold


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class CartesianProductManifold(AbstractManifold):
    r"""Manifold defined as a Cartesian product of other manifolds.

    Given smooth manifolds $M_1, M_2, \ldots, M_k$ of intrinsic dimensions $n_1,
    n_2, \ldots, n_k$, the **Cartesian product manifold** is

    $$M = M_1 \times M_2 \times \cdots \times M_k,$$

    whose points are $k$-tuples $(p_1, p_2, \ldots, p_k)$ with $p_i \in M_i$.
    The product is itself a smooth manifold of dimension

    $$\dim(M) = n_1 + n_2 + \cdots + n_k.$$

    **Smooth structure.** The atlas $\mathcal{A}_M$ of the product manifold
    consists precisely of Cartesian product charts

    $$C_1 \times C_2 \times \cdots \times C_k, \quad C_i \in
    \mathcal{A}_{M_i},$$

    encoded as {class}`~coordinax.charts.CartesianProductChart` instances.
    Transition maps on $M$ factor component-wise: if $\tau_i : C_i^\alpha \to
    C_i^\beta$ is the transition map on $M_i$, then the product transition map
    is

    $$\tau : (p_1, \ldots, p_k) \mapsto
        \bigl(\tau_1(p_1), \ldots, \tau_k(p_k)\bigr).$$

    **Naming and indexing.** Each factor is assigned a string name via
    ``factor_names``.  Names must be unique and are used to retrieve individual
    factor manifolds or factor atlases from the product atlas via string
    indexing (e.g. ``manifold.atlas["S2"]``).

    Parameters
    ----------
    factors : tuple[AbstractManifold, ...]
        The constituent manifolds $M_1, \ldots, M_k$ that form the product.
    factor_names : tuple[str, ...]
        Unique string names for each factor, in the same order as ``factors``.
        Used as keys when indexing into the product atlas.

    Attributes
    ----------
    atlas : CartesianProductAtlas
        The product atlas formed from the factor atlases.
    metric : CartesianProductMetric
        The canonical product metric formed from the factor metrics.
    ndim : int
        Total intrinsic dimension $\sum_i n_i$.
    default_chart : CartesianProductChart
        Product of the default charts from each factor atlas.

    Examples
    --------
    **Basic construction**

    The most common example is the phase-space-like manifold $S^2 \times
    \mathbb{R}$, which pairs the 2-sphere (2 angular degrees of freedom) with
    the real line (1 radial degree of freedom):

    >>> import coordinax.manifolds as cxm
    >>> import wadler_lindig as wl

    >>> M = cxm.CartesianProductManifold(
    ...     factors=(cxm.HyperSphericalManifold(), cxm.EuclideanManifold(1)),
    ...     factor_names=("S2", "R1"),
    ... )
    >>> wl.pprint(M, width=60)
    CartesianProductManifold(
        factors=(
          HyperSphericalManifold(ndim=2),
          EuclideanManifold(ndim=1)
        ),
        factor_names=('S2', 'R1')
    )

    **Dimension**

    The dimension equals the sum of the factor dimensions, $\dim(S^2) +
    \dim(\mathbb{R}) = 2 + 1 = 3$:

    >>> M.ndim
    3

    **Atlas and default chart**

    The atlas is a {class}`CartesianProductAtlas` whose default chart is the
    Cartesian product of the default charts of each factor:

    >>> M.atlas
    CartesianProductAtlas(factors=(HyperSphericalAtlas(ndim=2), EuclideanAtlas(ndim=1)),
        factor_names=('S2', 'R1'))

    >>> M.default_chart
    CartesianProductChart(
        factors=(SphericalTwoSphere(), Cart1D()), factor_names=('S2', 'R1')
    )

    Factor atlases can be retrieved by name:

    >>> M.atlas["S2"]
    HyperSphericalAtlas(ndim=2)

    >>> M.atlas["R1"]
    EuclideanAtlas(ndim=1)

    **Chart membership**

    A {class}`~coordinax.charts.CartesianProductChart` belongs to the product
    atlas when its factor charts belong to the corresponding factor atlases:

    >>> import coordinax.charts as cxc

    >>> product_chart = cxc.CartesianProductChart(
    ...     factors=(cxc.sph2, cxc.cart1d), factor_names=("S2", "R1")
    ... )
    >>> M.has_chart(product_chart)
    True

    Non-product charts and wrong-factor charts are rejected:

    >>> M.has_chart(cxc.sph2)
    False

    **Higher-dimensional products**

    Any number of factors may be combined. The 4-dimensional manifold $S^2
    \times \mathbb{R}^2$ has $\dim = 2 + 2 = 4$:

    >>> M4 = cxm.CartesianProductManifold(
    ...     factors=(cxm.HyperSphericalManifold(), cxm.EuclideanManifold(2)),
    ...     factor_names=("S2", "R2"),
    ... )
    >>> M4.ndim
    4

    **Euclidean-Euclidean products**

    Factor manifolds need not be non-Euclidean. The product $\mathbb{R}^2 \times
    \mathbb{R}$ reproduces 3-dimensional Euclidean space (though as a product
    structure rather than a single `EuclideanManifold`):

    >>> Mprod = cxm.CartesianProductManifold(
    ...     factors=(cxm.EuclideanManifold(2), cxm.EuclideanManifold(1)),
    ...     factor_names=("xy", "z"),
    ... )
    >>> Mprod.ndim
    3

    >>> Mprod.default_chart
    CartesianProductChart(factors=(Cart2D(), Cart1D()), factor_names=('xy', 'z'))

    """

    factors: tuple[AbstractManifold, ...]
    factor_names: tuple[str, ...]

    @override
    @property
    def atlas(self) -> CartesianProductAtlas:
        """Return the product atlas for the manifold.

        >>> import coordinax.manifolds as cxm
        >>> import wadler_lindig as wl

        >>> M = cxm.CartesianProductManifold(
        ...     factors=(cxm.HyperSphericalManifold(), cxm.EuclideanManifold(1)),
        ...     factor_names=("S2", "R1"))
        >>> wl.pprint(M.atlas, width=60)
        CartesianProductAtlas(
            factors=(HyperSphericalAtlas(), EuclideanAtlas(ndim=1)),
            factor_names=('S2', 'R1')
        )

        """
        factor_atlases = tuple(factor.atlas for factor in self.factors)
        return CartesianProductAtlas(
            factors=factor_atlases, factor_names=self.factor_names
        )

    @property
    def metric(self) -> CartesianProductMetric:
        """Return the canonical product metric from the factor metrics."""
        factor_metrics = tuple(factor.metric for factor in self.factors)
        return CartesianProductMetric(factors=factor_metrics)
