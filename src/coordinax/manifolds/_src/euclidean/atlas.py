"""Euclidean manifolds."""

__all__ = ("EuclideanAtlas",)

import dataclasses

from typing import Any, ClassVar, TypeVar, final

import jax

import coordinax.charts as cxc
from coordinax.manifolds._src.base import AbstractAtlas

CT = TypeVar("CT", bound=type[cxc.AbstractChart[Any, Any]])

EUCLIDEAN_ATLAS_DEFAULT_CHARTS: dict[int, cxc.AbstractChart[Any, Any]] = {
    0: cxc.cart0d,
    1: cxc.cart1d,
    2: cxc.cart2d,
    3: cxc.cart3d,
}


def _can_common_point_transition(chart: cxc.AbstractChart, dim: int, /) -> bool:
    expect = EUCLIDEAN_ATLAS_DEFAULT_CHARTS.get(dim)
    if expect is None:
        return False
    try:
        got = chart.cartesian
    except cxc.NoGlobalCartesianChartError:
        return False

    return got == expect


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class EuclideanAtlas(AbstractAtlas):
    r"""Atlas of coordinate charts for the Euclidean manifold $\mathbb{R}^n$.

    An **atlas** $\mathcal{A}$ is the collection of compatible charts that
    together cover a smooth manifold. For the Euclidean manifold $\mathbb{R}^n$,
    the atlas $\mathcal{A}_{\mathbb{R}^n}$ admits a chart $C = (U, \varphi)$ if
    and only if both of the following hold:

    1. The chart dimensionality matches $n$ (i.e. $\varphi$ maps into
       $\mathbb{R}^n$).
    2. The chart class is **explicitly registered** with
       {class}`~coordinax.manifolds.EuclideanAtlas` via {meth}`register`, *or*
       the chart has a compatible transition map to the default Cartesian chart
       (detected via ``chart.cartesian``).

    **Built-in charts.** The following chart classes are registered
    automatically:

    - *0-D*: `Cart0D`
    - *1-D*: `Cart1D`, `Radial1D`
    - *2-D*: `Cart2D`, `Polar2D`
    - *3-D*: `Cart3D`, `Cylindrical3D`, `Spherical3D`, `LonLatSpherical3D`,
      `LonCosLatSpherical3D`, `MathSpherical3D`, `ProlateSpheroidal3D`
    - *N-D*: `CartND`

    **Default chart.** The atlas provides a canonical Cartesian chart for each
    dimension: `Cart0D` through `Cart3D` for $n \leq 3$, and `CartND` for $n >
    3$.

    Parameters
    ----------
    ndim : int
        Dimension $n \geq 0$ of the Euclidean manifold that this atlas covers.

    Examples
    --------
    **Construction**

    >>> import coordinax.manifolds as cxmd
    >>> atlas = cxmd.EuclideanAtlas(3)
    >>> atlas
    EuclideanAtlas(ndim=3)

    >>> atlas.ndim
    3

    **Default chart**

    The atlas provides a canonical Cartesian chart for each dimension:

    >>> atlas.default_chart()
    Cart3D()

    >>> cxmd.EuclideanAtlas(2).default_chart()
    Cart2D()

    >>> cxmd.EuclideanAtlas(1).default_chart()
    Cart1D()

    For $n > 3$ the fallback is `CartND`:

    >>> cxmd.EuclideanAtlas(10).default_chart()
    CartND()

    **Chart membership**

    Use the ``in`` operator (via {meth}`~AbstractAtlas.__contains__`) to test
    whether a chart instance belongs to this atlas:

    >>> import coordinax.charts as cxc

    >>> cxc.cart3d in atlas
    True

    >>> cxc.sph3d in atlas
    True

    Charts with the wrong dimensionality are rejected:

    >>> cxc.cart2d in atlas
    False

    **Point transition maps**

    Convert point coordinates between any two supported charts. The atlas
    validates that both charts belong to it before delegating to the chart-level
    transition map. Here the point $(1, 0, 0)$ in Cartesian — lying on the
    positive $x$-axis — maps to $(r, \theta, \phi) = (1, \pi/2, 0)$ in
    spherical:

    >>> x = {"x": 1.0, "y": 0.0, "z": 0.0}
    >>> atlas.pt_map(x, cxc.cart3d, cxc.sph3d)
    {'r': Array(1., ...), 'theta': Array(1.57079633, ...), 'phi': Array(0., ...)}

    Attempting a transition to a chart not in the atlas raises:

    >>> try:
    ...     atlas.pt_map(x, cxc.cart3d, cxc.cart2d)
    ... except ValueError as e:
    ...     print(e)
    Atlas EuclideanAtlas(ndim=3) does not support chart Cart2D()

    """

    ndim: int
    """Dimension of the Euclidean manifold."""

    _ELIGIBLE_CHARTS: ClassVar[set[type[cxc.AbstractChart[Any, Any]]]] = set()

    def default_chart(self) -> cxc.AbstractChart[Any, Any]:
        """Return the default chart for this atlas.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxmd

        >>> cxmd.EuclideanAtlas(0).default_chart()
        Cart0D()

        >>> cxmd.EuclideanAtlas(1).default_chart()
        Cart1D()

        >>> cxmd.EuclideanAtlas(2).default_chart()
        Cart2D()

        >>> cxmd.EuclideanAtlas(3).default_chart()
        Cart3D()

        For higher dimensions, the default is CartND:

        >>> cxmd.EuclideanAtlas(100).default_chart()
        CartND()

        """
        return EUCLIDEAN_ATLAS_DEFAULT_CHARTS.get(self.ndim, cxc.cartnd)

    def has_chart(self, chart: cxc.AbstractChart[Any, Any], /) -> bool:
        """Return whether the atlas supports the given chart.

        This checks if the chart has the same dimension as the atlas and is
        either explicitly registered or has a common point transition map to the
        default Cartesian chart.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxmd

        >>> atlas = cxmd.EuclideanAtlas(2)

        >>> atlas.has_chart(cxc.cart2d)
        True

        >>> atlas.has_chart(cxc.polar2d)
        True

        >>> atlas.has_chart(cxc.cart3d)
        False

        """
        return chart.ndim == self.ndim and (
            type(chart) in self._ELIGIBLE_CHARTS
            or _can_common_point_transition(chart, self.ndim)
        )

    @classmethod
    def register(cls, registrant: CT, /) -> CT:
        """Register a class for Euclidean Atlas eligibility.

        This does not guarantee that a particular atlas will support the chart,
        since support also depends on dimensionality and transition maps. It
        simply makes the chart class eligible for support if those other
        conditions are met.

        """
        cls._ELIGIBLE_CHARTS.add(registrant)
        return registrant


for chart_cls in (
    # 0-D
    cxc.Cart0D,
    # 1-D
    cxc.Cart1D,
    cxc.Radial1D,
    # 2-D
    cxc.Cart2D,
    cxc.Polar2D,
    # 3-D
    cxc.Cart3D,
    cxc.Cylindrical3D,
    cxc.Spherical3D,
    cxc.LonLatSpherical3D,
    cxc.LonCosLatSpherical3D,
    cxc.MathSpherical3D,
    cxc.ProlateSpheroidal3D,
    # N-D
    cxc.CartND,
):
    _ = EuclideanAtlas.register(chart_cls)
