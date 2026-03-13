"""Manifold definitions and manifold inference helpers."""

__all__ = (
    "AbstractManifold",
    "AbstractAtlas",
)

import abc

from typing import Any

import plum

import unxt as u

import coordinax.charts as cxc
import coordinax.metrics as cxm
from coordinax.internal.custom_types import CDict, OptUSys


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
    machinery (e.g. :func:`point_transition_map`).

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
    >>> atlas.supports(cxc.cart3d)
    True

    >>> atlas.supports(cxc.cyl3d)
    True

    Charts with the wrong dimensionality are rejected:

    >>> atlas.supports(cxc.cart2d)
    False

    **Atlas-manifold interaction**

    A manifold object typically owns an atlas describing its smooth
    structure.

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
    def supports(self, chart: cxc.AbstractChart[Any, Any]) -> bool:
        """Return whether the atlas supports the given chart.

        >>> import coordinax.manifolds as cxm
        >>> atlas = cxm.EuclideanAtlas(2)
        >>> atlas.supports(cxc.cart2d)
        True

        >>> atlas.supports(cxc.cart3d)
        False

        """
        raise NotImplementedError  # pragma: no cover

    # =====================================================

    def point_transition_map(self, *args: Any, **kwargs: Any) -> Any:
        """Transition map for points, checking the atlas.

        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        >>> atlas = cxm.EuclideanAtlas(2)

        >>> x = {"x": 1.0, "y": 1.0}
        >>> atlas.point_transition_map(cxc.polar2d, cxc.cart2d, x)
        {'r': Array(1.41421356, dtype=float64, ...),
         'theta': Array(0.78539816, dtype=float64, ...)}

        >>> try: atlas.point_transition_map(cxc.sph2, cxc.cart2d, x)
        ... except ValueError as e: print(e)
        Atlas EuclideanAtlas(ndim=2) does not support chart SphericalTwoSphere()

        """
        return cxc.point_transition_map(self, *args, **kwargs)


class AbstractManifold(metaclass=abc.ABCMeta):
    """Abstract manifold interface."""

    metric: cxm.AbstractMetric
    """Metric on the manifold. This defines the Riemannian structure."""

    atlas: AbstractAtlas
    """Charts compatible with this manifold. This defines the smooth structure."""

    def __post_init__(self) -> None:
        self.check_ndim()

    def check_ndim(self) -> None:
        """Check that the chart and metric dimensions match the manifold dimension."""
        if self.metric.ndim != self.atlas.ndim:
            msg = (
                f"Atlas dimension {self.atlas.ndim} "
                f"does not match metric dimension {self.metric.ndim}."
            )
            raise ValueError(msg)

    @property
    def ndim(self) -> int:
        """Return the dimension of the manifold."""
        return self.atlas.ndim

    @property
    def default_chart(self) -> cxc.AbstractChart[Any, Any]:
        """Return a default chart from the atlas."""
        return self.atlas.default_chart()

    def has_chart(self, chart: cxc.AbstractChart[Any, Any], /) -> bool:
        """Return whether ``chart`` belongs to this manifold atlas."""
        return self.atlas.supports(chart)

    def check_chart(self, chart: cxc.AbstractChart[Any, Any], /) -> None:
        """Check that ``chart`` belongs to this manifold atlas."""
        if not self.has_chart(chart):
            msg = f"Chart {chart!r} is not supported by this manifold atlas."
            raise ValueError(msg)

    # =====================================================

    def point_transition_map(self, *args: Any, **kwargs: Any) -> Any:
        """Transition map for points, checking the manifold.

        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        >>> M = cxm.EuclideanManifold(2)

        >>> x = {"x": 1.0, "y": 1.0}
        >>> M.point_transition_map(cxc.polar2d, cxc.cart2d, x)
        {'r': Array(1.41421356, dtype=float64, ...),
         'theta': Array(0.78539816, dtype=float64, ...)}

        >>> try: M.point_transition_map(cxc.sph2, cxc.cart2d, x)
        ... except ValueError as e: print(e)
        Atlas EuclideanAtlas(ndim=2) does not support chart SphericalTwoSphere()

        """
        return cxc.point_transition_map(self, *args, **kwargs)

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
        2-sphere embedded in $\mathbb{R}^3$), then the manifold (or an associated
        embedding object) is the natural owner of that data, and this method
        should use it to compute the realization.

        The realization map is conceptually

        $$ X: V \subset \mathbb{R}^n \to \mathbb{R}^m, $$

        evaluated on point-role coordinates ``data`` expressed in ``chart``.

        Dispatch / resolution order
        ---------------------------
        1. If ``self`` defines an attribute ``embedding`` with a method
           ``realize_cartesian(chart, data, usys=...)``, that method is used.
        2. Otherwise, this method delegates to ``chart.realize_cartesian``.

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

        try:
            return chart.realize_cartesian(data, usys=usys)
        except Exception as e:  # pragma: no cover
            raise NotImplementedError(
                "No ambient realization map is available for this manifold. "
                "Construct an `EmbeddedChart` "
                "or use as chart that defines `realize_cartesian`."
            ) from e

    def unrealize_cartesian(
        self,
        chart: cxc.AbstractChart[Any, Any],
        data: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        r"""Invert the ambient Cartesian realization on the chart domain.

        This is the manifold-level inverse of :meth:`realize_cartesian`.

        It converts point-role coordinates in the canonical ambient Cartesian
        chart into point-role coordinates in ``chart``. The inverse may be
        undefined or multi-valued globally; this method represents the inverse
        only on the intended domain of ``chart``.

        Dispatch / resolution order
        ---------------------------
        1. If ``self`` defines an attribute ``embedding`` with a method
           ``unrealize_cartesian(chart, data, usys=...)``, that method is used.
        2. Otherwise, this method delegates to ``chart.unrealize_cartesian``.

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

        try:
            return chart.unrealize_cartesian(data, usys=usys)
        except Exception as e:  # pragma: no cover
            msg = (
                "No inverse ambient realization map is available for this manifold. "
                "Construct an `EmbeddedChart` "
                "or use as chart that defines `unrealize_cartesian`."
            )
            raise NotImplementedError(msg) from e


# =====================================
# Metric


@plum.dispatch
def norm(
    manifold: AbstractManifold,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    v: CDict,
    /,
    *,
    at: CDict | None = None,
) -> u.AbstractQuantity:
    """Compute the norm of a vector, forwarding to the manifold's metric."""
    return cxm.norm(manifold.metric, chart, v, at=at)
