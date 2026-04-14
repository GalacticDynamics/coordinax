"""Manifold definitions and manifold inference helpers."""

__all__ = (
    "AbstractManifold",
    "AbstractAtlas",
)

import abc

from typing import Any

import jax.tree_util
import wadler_lindig as wl

import dataclassish

import coordinax.charts as cxc
from coordinax.internal.custom_types import CDict, OptUSys


@jax.tree_util.register_static
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
    machinery (e.g. {func}`pt_map`).

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
    Cart3D()

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
    def default_chart(self) -> cxc.AbstractChart[Any, Any]:
        """Return a default chart from the atlas.

        >>> import coordinax.manifolds as cxm
        >>> atlas = cxm.EuclideanAtlas(2)
        >>> atlas.default_chart()
        Cart2D()

        """
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def has_chart(self, chart: cxc.AbstractChart[Any, Any], /) -> bool:
        """Return whether the atlas supports the given chart.

        >>> import coordinax.manifolds as cxm
        >>> atlas = cxm.EuclideanAtlas(2)
        >>> atlas.has_chart(cxc.cart2d)
        True

        >>> atlas.has_chart(cxc.cart3d)
        False

        """
        raise NotImplementedError  # pragma: no cover

    def __contains__(self, chart: cxc.AbstractChart[Any, Any]) -> bool:
        """Return whether the atlas supports the given chart.

        >>> import coordinax.manifolds as cxm
        >>> atlas = cxm.EuclideanAtlas(2)
        >>> cxc.cart2d in atlas
        True

        >>> cxc.cart3d in atlas
        False

        """
        return self.has_chart(chart)

    # =====================================================

    def pt_map(self, x: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Transition map for points, checking the atlas.

        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        >>> atlas = cxm.EuclideanAtlas(2)

        >>> x = {"x": 1.0, "y": 1.0}
        >>> atlas.pt_map(x, cxc.cart2d, cxc.polar2d)
        {'r': Array(1.41421356, dtype=float64, ...),
         'theta': Array(0.78539816, dtype=float64, ...)}

        >>> try: atlas.pt_map(x, cxc.cart2d, cxc.sph2)
        ... except ValueError as e: print(e)
        Atlas EuclideanAtlas(ndim=2) does not support chart SphericalTwoSphere()

        """
        return cxc.pt_map(x, self, *args, **kwargs)


@jax.tree_util.register_static
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
    EuclideanManifold(ndim=3)

    The intrinsic dimension is read from the atlas:

    >>> M.ndim
    3

    The atlas itself is accessible and carries the same dimension:

    >>> M.atlas
    EuclideanAtlas(ndim=3)

    The default chart is the canonical Cartesian chart for that dimension:

    >>> M.default_chart
    Cart3D()

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
    Chart Cart2D() is not supported by this manifold atlas.

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
    Atlas EuclideanAtlas(ndim=3) does not support chart SphericalTwoSphere()

    **Cartesian realization and its inverse**

    {meth}`realize_cartesian` maps point coordinates expressed in any chart to
    the canonical ambient Cartesian chart. For a Euclidean manifold this is
    simply a transition map to the Cartesian chart:

    >>> M.realize_cartesian(cxc.sph3d, {"r": 1.0, "theta": 0.0, "phi": 0.0})
    {'x': Array(0., ...), 'y': Array(0., ...), 'z': Array(1., ...)}

    {meth}`unrealize_cartesian` is its inverse --- converting Cartesian
    coordinates back into a target chart:

    >>> M.unrealize_cartesian(cxc.sph3d, {"x": 0.0, "y": 0.0, "z": 1.0})
    {'r': Array(1., ...), 'theta': Array(0., ...), 'phi': Array(0., ...)}

    Both methods validate that the supplied chart belongs to the atlas before
    proceeding:

    >>> try:
    ...     M.realize_cartesian(cxc.sph2, {"theta": 0.0, "phi": 0.0})
    ... except ValueError as e:
    ...     print(e)
    Chart SphericalTwoSphere() is not supported by this manifold atlas.

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

    >>> S2.default_chart
    SphericalTwoSphere()

    """

    atlas: AbstractAtlas
    """Charts compatible with this manifold. This defines the smooth structure."""

    def __post_init__(self) -> None:
        self._check_ndim()

    def _check_ndim(self) -> None:  # noqa: B027
        """Check that the chart and metric dimensions match the manifold dimension."""
        # if self.metric.ndim != self.atlas.ndim:
        #     msg = (
        #         f"Atlas dimension {self.atlas.ndim} "
        #         f"does not match metric dimension {self.metric.ndim}."
        #     )
        #     raise ValueError(msg)

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

    @property
    def default_chart(self) -> cxc.AbstractChart[Any, Any]:
        """Return a default chart from the atlas.

        This is a convenience property that proxies to the atlas default chart.

        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(2)
        >>> M.default_chart
        Cart2D()

        """
        return self.atlas.default_chart()

    def has_chart(self, chart: cxc.AbstractChart[Any, Any], /) -> bool:
        """Return whether ``chart`` belongs to this manifold atlas.

        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(2)
        >>> M.has_chart(cxc.cart2d)
        True
        >>> M.has_chart(cxc.cart3d)
        False

        """
        return self.atlas.has_chart(chart)

    def check_chart(self, chart: cxc.AbstractChart[Any, Any], /) -> None:
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
        Atlas EuclideanAtlas(ndim=2) does not support chart SphericalTwoSphere()

        """
        return cxc.pt_map(x, self, *args, **kwargs)

    def realize_cartesian(
        self,
        chart: cxc.AbstractChart[Any, Any],
        data: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        r"""Realize a manifold point in canonical ambient Cartesian coordinates.

        This is the manifold-level entrypoint for evaluating an **ambient
        realization map** into a canonical ambient Cartesian coordinate system.

        If a manifold has extra geometric embedding data (e.g. a radius for a
        2-sphere embedded in $\mathbb{R}^3$), then the manifold (or an
        associated embedding object) is the natural owner of that data, and this
        method should use it to compute the realization.

        The realization map is conceptually

        $$ X: V \subset \mathbb{R}^n \to \mathbb{R}^m, $$

        evaluated on point-role coordinates ``data`` expressed in ``chart``.

        Notes
        -----
        - This is **point-role only**. It does not apply to tangent-valued roles
          and does not involve Jacobians, metrics, or frame evaluation.
        - Not all manifolds admit a distinguished embedding into an ambient
          Euclidean space; in that case, this method may raise.

        Parameters
        ----------
        chart:
            Chart in which ``data`` is expressed. Must be supported by this
            manifold's atlas.
        data:
            Point coordinates in ``chart``.
        usys:
            Optional unit system used to interpret unitful inputs.

        Returns
        -------
        CDict
            Point coordinates in the canonical ambient Cartesian chart.

        Raises
        ------
        ValueError
            If ``chart`` is not supported by this manifold.
        NotImplementedError
            If no realization map is available via ``self.embedding`` or
            ``chart.realize_cartesian``.

        """
        self.check_chart(chart)
        return chart.realize_cartesian(data, usys=usys)

    def unrealize_cartesian(
        self,
        chart: cxc.AbstractChart[Any, Any],
        data: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        r"""Invert the ambient Cartesian realization on the chart domain.

        This is the manifold-level inverse of {meth}`realize_cartesian`.

        It converts point-role coordinates in the canonical ambient Cartesian
        chart into point-role coordinates in ``chart``. The inverse may be
        undefined or multi-valued globally; this method represents the inverse
        only on the intended domain of ``chart``.

        Notes
        -----
        - This is **point-role only**. It does not apply to tangent-valued roles
          and does not involve Jacobians, metrics, or frame evaluation.

        Parameters
        ----------
        chart:
            Target chart in which the returned coordinates should be expressed.
            Must be supported by this manifold's atlas.
        data:
            Point coordinates in the canonical ambient Cartesian chart.
        usys:
            Optional unit system used to interpret unitful inputs.

        Returns
        -------
        CDict
            Point coordinates in ``chart``.

        Raises
        ------
        ValueError
            If ``chart`` is not supported by this manifold.
        NotImplementedError
            If no inverse realization map is available via ``self.embedding`` or
            ``chart.unrealize_cartesian``.

        """
        self.check_chart(chart)
        return chart.unrealize_cartesian(data, usys=usys)

    # =====================================================

    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        """Return the string representation.

        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(3)

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
