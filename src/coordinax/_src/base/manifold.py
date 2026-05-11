"""Manifold definitions and manifold inference helpers."""

__all__ = ("AbstractManifold",)

import abc

from typing import TYPE_CHECKING, Any

import jax.tree_util as jtu
import wadler_lindig as wl

import dataclassish

import coordinax.angles as cxa
import coordinax.api.charts as cxcapi
import coordinax.api.manifolds as cxmapi
from .atlas import AbstractAtlas
from .metric import AbstractMetric
from coordinax._src.custom_types import CDict, OptUSys
from coordinax.internal import QuantityMatrix

if TYPE_CHECKING:
    import coordinax.charts  # noqa: ICN001


@jtu.register_static
class AbstractManifold(metaclass=abc.ABCMeta):
    r"""Abstract interface for smooth manifolds.

    A **smooth manifold** of dimension $n$ is a topological space $M$ equipped
    with a maximal smooth atlas $\mathcal{A}$ --- a collection of compatible
    **charts** $(U_\alpha, \varphi_\alpha)$ whose domains cover $M$. Each chart
    $\varphi_\alpha : U_\alpha \subset M \to \mathbb{R}^n$ assigns local
    coordinates to points in an open neighbourhood $U_\alpha$.

    ``AbstractManifold`` is the base class for all manifold objects in
    `coordinax`. It couples a manifold to its atlas and exposes a small set of
    coordinate-level operations:

    - **chart introspection** --- querying which charts belong to the manifold,
    - **point transition maps** --- converting point coordinates between two
      charts in the same atlas, and
    - **Cartesian realization** --- converting point coordinates into (or out
      of) a canonical ambient Cartesian chart when the manifold admits one.

    All geometric and numerical work is delegated to the charts and the atlas;
    the manifold itself is a lightweight descriptor.

    Attributes
    ----------
    atlas : AbstractAtlas
        The atlas that defines the smooth structure of this manifold. The atlas
        records the intrinsic dimension and determines which charts are
        compatible.
    ndim : int
        Intrinsic dimension $n$ of the manifold, forwarded from
        {attr}`atlas.ndim <AbstractAtlas.ndim>`.
    default_chart : AbstractChart
        A canonical chart chosen by the atlas, forwarded from
        {meth}`atlas.default_chart() <AbstractAtlas.default_chart>`.

    Notes
    -----
    - Manifold objects are **structural descriptors**, not numerical arrays.
      They carry no point data.
    - Subclasses are typically frozen dataclasses registered with JAX as static
      pytree nodes so that they can appear as static metadata inside JIT-compiled
      functions.
    - The two-sphere $S^2$ is *not* a product manifold: its atlas requires at
      least two charts with non-trivial overlaps, and its transition maps do not
      factor. Contrast with the Euclidean case where a single global chart
      covers all of $\mathbb{R}^n$.

    Examples
    --------
    ``AbstractManifold`` cannot be instantiated directly; use a concrete
    subclass such as {class}`~coordinax.manifolds.EuclideanManifold` or
    {class}`~coordinax.manifolds.HyperSphericalManifold`.

    **Basic construction and introspection**

    The Euclidean 3-manifold $\mathbb{R}^3$ with its standard atlas:

    >>> import coordinax.manifolds as cxm
    >>> M = cxm.EuclideanManifold(3)
    >>> M
    Rn(3)

    The intrinsic dimension is read from the atlas:

    >>> M.ndim
    3

    The atlas itself is accessible and carries the same dimension:

    >>> M.atlas
    EuclideanAtlas(ndim=3)

    The default chart is the canonical Cartesian chart for that dimension:

    >>> M.default_chart()
    Cart3D(M=Rn(3))

    **Chart membership**

    {meth}`has_chart` returns ``True`` when a chart instance belongs to the
    manifold's atlas:

    >>> import coordinax.charts as cxc
    >>> M.has_chart(cxc.cart3d)
    True

    >>> M.has_chart(cxc.sph3d)
    True

    Charts with the wrong dimensionality are rejected:

    >>> M.has_chart(cxc.cart2d)
    False

    {meth}`check_chart` raises {exc}`ValueError` for unsupported charts,
    which makes it convenient as an assertion inside other methods:

    >>> try:
    ...     M.check_chart(cxc.cart2d)
    ... except ValueError as e:
    ...     print(e)
    Chart Cart2D(M=Rn(2)) is not supported by this manifold atlas.

    **Point transition maps**

    {meth}`pt_map` converts point-role coordinates between two
    charts that both belong to the manifold atlas. The transition map is the
    composition $\varphi_\beta \circ \varphi_\alpha^{-1}$; no embedding is
    involved.

    Converting the point $(0, 0, 1)$ from Cartesian to spherical on
    $\mathbb{R}^3$ --- the north pole maps to $(r, \theta, \phi) = (1, 0, 0)$:

    >>> x = {"x": 0.0, "y": 0.0, "z": 1.0}
    >>> M.pt_map(x, cxc.cart3d, cxc.sph3d)
    {'r': Array(1., ...), 'theta': Array(0., ...), 'phi': Array(0., ...)}

    Converting $(1, 1)$ from Cartesian to polar on $\mathbb{R}^2$ --- the
    point lies at distance $\sqrt{2}$ from the origin at angle $\pi/4$:

    >>> M2 = cxm.EuclideanManifold(2)
    >>> M2.pt_map({"x": 1.0, "y": 1.0}, cxc.cart2d, cxc.polar2d)
    {'r': Array(1.41421356, ...), 'theta': Array(0.78539816, ...)}

    Passing a chart not in the atlas raises {exc}`ValueError`:

    >>> try:
    ...     M.pt_map(x, cxc.cart3d, cxc.sph2)
    ... except ValueError as e:
    ...     print(e)
    Atlas EuclideanAtlas(ndim=3) does not support chart SphericalTwoSphere(M=Sn(2))

    **Non-Euclidean manifolds**

    The two-sphere $S^2$ is a 2-dimensional manifold that is *not* a subspace
    of any Euclidean atlas. Its atlas admits only angular charts:

    >>> S2 = cxm.HyperSphericalManifold()
    >>> S2.ndim
    2

    >>> S2.has_chart(cxc.sph2)
    True

    >>> S2.has_chart(cxc.cart2d)
    False

    >>> S2.default_chart()
    SphericalTwoSphere(M=Sn(2))

    """

    atlas: AbstractAtlas
    """Charts compatible with this manifold. This defines the smooth structure."""

    metric: AbstractMetric
    """The manifold's metric. This defines the geometric structure."""

    def __post_init__(self) -> None:
        self._check_ndim()

    def _check_ndim(self) -> None:
        """Check that the chart and metric dimensions match the manifold dimension."""
        if self.metric.ndim != self.atlas.ndim:
            msg = (
                f"Atlas dimension {self.atlas.ndim} "
                f"does not match metric dimension {self.metric.ndim}."
            )
            raise ValueError(msg)

    @property
    def ndim(self) -> int:
        """Return the dimension of the manifold.

        This is a convenience property that proxies to the atlas dimension,
        since the atlas defines the smooth structure of the manifold and
        therefore determines its dimension.

        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(3)
        >>> M.ndim
        3

        """
        return self.atlas.ndim

    def default_chart(self) -> "coordinax.charts.AbstractChart[Any, Any]":
        """Return a default chart from the atlas.

        This is a convenience property that proxies to the atlas default chart.

        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(2)
        >>> M.default_chart()
        Cart2D(M=Rn(2))

        """
        return self.atlas.default_chart()

    def has_chart(self, chart: "coordinax.charts.AbstractChart[Any, Any]", /) -> bool:
        """Return whether ``chart`` belongs to this manifold atlas.

        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(2)
        >>> M.has_chart(cxc.cart2d)
        True
        >>> M.has_chart(cxc.cart3d)
        False

        """
        return self.atlas.has_chart(chart)

    def check_chart(self, chart: "coordinax.charts.AbstractChart[Any, Any]", /) -> None:
        """Check that ``chart`` belongs to this manifold atlas.

        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(2)
        >>> M.check_chart(cxc.cart2d)  # does not raise

        """
        if not self.has_chart(chart):
            msg = f"Chart {chart!r} is not supported by this manifold atlas."
            raise ValueError(msg)

    # =====================================================

    def pt_map(self, x: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Transition map for points, checking the manifold.

        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        >>> M = cxm.EuclideanManifold(2)

        >>> x = {"x": 1.0, "y": 1.0}
        >>> M.pt_map(x, cxc.cart2d, cxc.polar2d)
        {'r': Array(1.41421356, dtype=float64, ...),
         'theta': Array(0.78539816, dtype=float64, ...)}

        >>> try: M.pt_map(x, cxc.cart2d, cxc.sph2)
        ... except ValueError as e: print(e)
        Atlas EuclideanAtlas(ndim=2) does not support chart SphericalTwoSphere(M=Sn(2))

        """
        return cxcapi.pt_map(x, self, *args, **kwargs)

    # =====================================================

    def scale_factors(
        self,
        chart: "coordinax.charts.AbstractChart[Any, Any]",
        /,
        *,
        at: CDict,
        usys: OptUSys = None,
    ) -> QuantityMatrix:
        r"""Return the diagonal entries of the manifold metric in ``chart`` at ``at``.

        This is a thin convenience wrapper over
        ``cxmapi.scale_factors(self.metric, chart, at=at, usys=usys)``.
        """
        return cxmapi.scale_factors(self.metric, chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]

    def angle_between(
        self,
        chart: "coordinax.charts.AbstractChart[Any, Any]",
        uvec: CDict,
        vvec: CDict,
        /,
        *,
        at: CDict,
        usys: OptUSys = None,
    ) -> cxa.AbstractAngle:
        r"""Return the metric angle between two tangent vectors at ``at``.

        This is a thin convenience wrapper over
        ``cxmapi.angle_between(self.metric, chart, uvec, vvec, at=at, usys=usys)``.
        """
        return cxmapi.angle_between(self.metric, chart, uvec, vvec, at=at, usys=usys)  # ty: ignore[invalid-return-type]

    # =====================================================

    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        """Return the string representation.

        >>> import wadler_lindig as wl
        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(3)
        >>> wl.pprint(M)
        Rn(3)

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
