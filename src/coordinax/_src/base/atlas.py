"""Manifold definitions and manifold inference helpers."""

__all__ = ("AbstractAtlas",)

import abc

from typing import TYPE_CHECKING, Any

import jax.tree_util as jtu

if TYPE_CHECKING:
    import coordinax.charts  # noqa: ICN001
    import coordinax.manifolds  # noqa: ICN001


@jtu.register_static
class AbstractAtlas(metaclass=abc.ABCMeta):
    r"""Atlas protocol for manifolds.

    An atlas defines the **set of charts** that may be used to represent
    coordinates on a manifold. In differential geometry, a smooth manifold is
    defined by a pair $(M, \\mathcal{A})$ where $M$ is a topological space and
    $\\mathcal{A}$ is a maximal smooth atlas — a collection of compatible charts
    whose domains cover the $M$.

    Responsibilities of an atlas include:

    - declaring the **dimension** of the manifold it covers,
    - determining whether a chart is **compatible** with the manifold,
    - providing a **default chart** used when one is not explicitly specified.

    The atlas does **not** perform coordinate transformations itself. Those are
    implemented by chart-level transition maps and higher-level transformation
    machinery (e.g. :func:`coordinax.charts.pt_map`).

    Notes
    -----
    - Atlas objects are **structural descriptors**, not numerical objects.
    - Multiple manifolds may share the same atlas type if their smooth
      structures coincide.
    - Charts belonging to the same atlas are assumed to have compatible
      transition maps.

    Some atlas implementations allow charts to register themselves as compatible
    coordinate systems. For example, Euclidean charts register with
    `coordinax.manifolds.EuclideanAtlas` so they can be recognized
    automatically.

    Examples
    --------
    **Constructing a Euclidean atlas**

    In the Euclidean case the atlas consists of common coordinate systems on
    $\mathbb{R}^n$.

    >>> import coordinax.manifolds as cxm
    >>> atlas = cxm.EuclideanAtlas(3)

    The atlas records the dimension of the manifold:

    >>> atlas.ndim
    3

    It can provide a canonical chart:

    >>> atlas.default_chart()
    Cart3D(M=Rn(3))

    The atlas determines whether a chart belongs to the manifold.

    >>> import coordinax.charts as cxc
    >>> cxc.cart3d in atlas
    True

    >>> cxc.cyl3d in atlas
    True

    Charts with the wrong dimensionality are rejected:

    >>> cxc.cart2d in atlas
    False

    **Atlas-manifold interaction**

    A manifold object typically owns an atlas describing its smooth structure.

    >>> from coordinax.manifolds import EuclideanManifold
    >>> M = EuclideanManifold(3)

    >>> M.atlas.ndim
    3

    The manifold uses the atlas to verify chart compatibility:

    >>> M.has_chart(cxc.cart3d)
    True

    >>> M.has_chart(cxc.cart2d)
    False

    """

    ndim: int
    """Dimension of the manifold that this atlas covers."""

    @abc.abstractmethod
    def default_chart(self) -> "coordinax.charts.AbstractChart[Any, Any, Any]":
        """Return a default chart from the atlas.

        >>> import coordinax.manifolds as cxm
        >>> atlas = cxm.EuclideanAtlas(2)
        >>> atlas.default_chart()
        Cart2D(M=Rn(2))

        """
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def has_chart(
        self, chart: "coordinax.charts.AbstractChart[Any, Any, Any]", /
    ) -> bool:
        """Return whether the atlas supports the given chart.

        >>> import coordinax.manifolds as cxm
        >>> atlas = cxm.EuclideanAtlas(2)
        >>> atlas.has_chart(cxc.cart2d)
        True

        >>> atlas.has_chart(cxc.cart3d)
        False

        """
        raise NotImplementedError  # pragma: no cover

    def __contains__(
        self, chart: "coordinax.charts.AbstractChart[Any, Any, Any]"
    ) -> bool:
        """Return whether the atlas supports the given chart.

        >>> import coordinax.manifolds as cxm
        >>> atlas = cxm.EuclideanAtlas(2)
        >>> cxc.cart2d in atlas
        True

        >>> cxc.cart3d in atlas
        False

        """
        return self.has_chart(chart)
