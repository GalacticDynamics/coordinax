"""Euclidean manifolds."""

__all__ = ("EuclideanAtlas",)

import dataclasses
import weakref

from typing import Any, Final, TypeVar, final

import jax

import coordinax.charts as cxc
from coordinax._src.base import AbstractAtlas, AbstractChart
from coordinax._src.exceptions import NoGlobalCartesianChartError

CT = TypeVar("CT", bound=type[AbstractChart[Any, Any]])

EUCLIDEAN_ATLAS_DEFAULT_CHARTS: Final[dict[int, AbstractChart[Any, Any]]] = {}

EUCLIDEAN_ATLAS_ELIGIBLE_CHARTS: weakref.WeakSet[type[AbstractChart[Any, Any]]] = (
    weakref.WeakSet()
)


def _can_common_point_transition(chart: AbstractChart, dim: int, /) -> bool:
    """Return whether ``chart`` maps to the default Cartesian chart for ``dim``."""
    expect = EUCLIDEAN_ATLAS_DEFAULT_CHARTS.get(dim)
    if expect is None:
        return False
    try:
        got = chart.cartesian
    except NoGlobalCartesianChartError:
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
    Cart3D(M=Rn(3))

    >>> cxmd.EuclideanAtlas(2).default_chart()
    Cart2D(M=Rn(2))

    >>> cxmd.EuclideanAtlas(1).default_chart()
    Cart1D(M=Rn(1))

    For $n > 3$ the fallback is `CartND`:

    >>> cxmd.EuclideanAtlas(10).default_chart()
    CartND(M=Rn(True))

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

    """

    ndim: int
    """Dimension of the Euclidean manifold."""

    def default_chart(self) -> AbstractChart[Any, Any]:
        """Return the default chart for this atlas.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxmd

        >>> cxmd.EuclideanAtlas(0).default_chart()
        Cart0D(M=Rn(0))

        >>> cxmd.EuclideanAtlas(1).default_chart()
        Cart1D(M=Rn(1))

        >>> cxmd.EuclideanAtlas(2).default_chart()
        Cart2D(M=Rn(2))

        >>> cxmd.EuclideanAtlas(3).default_chart()
        Cart3D(M=Rn(3))

        For higher dimensions, the default is CartND:

        >>> cxmd.EuclideanAtlas(100).default_chart()
        CartND(M=Rn(True))

        """
        return EUCLIDEAN_ATLAS_DEFAULT_CHARTS.get(self.ndim, cxc.cartnd)

    def has_chart(self, chart: AbstractChart[Any, Any], /) -> bool:
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
            type(chart) in EUCLIDEAN_ATLAS_ELIGIBLE_CHARTS
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
        EUCLIDEAN_ATLAS_ELIGIBLE_CHARTS.add(registrant)
        return registrant
