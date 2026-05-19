"""Two-sphere manifold."""

__all__ = ("HyperSphericalAtlas",)

import dataclasses
import weakref

from typing import Any, Final, TypeVar, final

import jax

from coordinax._src.base import AbstractAtlas, AbstractChart

CT = TypeVar("CT", bound=type[AbstractChart[Any, Any, Any]])


SPHERICAL_ATLAS_DEFAULT_CHARTS: Final[dict[int, AbstractChart[Any, Any, Any]]] = {}
SPHERICAL_ATLAS_ELIGIBLE_CHARTS: Final[
    weakref.WeakSet[type[AbstractChart[Any, Any, Any]]]
] = weakref.WeakSet()


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class HyperSphericalAtlas(AbstractAtlas):
    """Atlas for spherical manifolds (e.g. the circle or 2-sphere).

    E.g. the 2-sphere contains charts:

    - `~coordinax.charts.SphericalTwoSphere`,
    - `~coordinax.charts.LonLatSphericalTwoSphere`,
    - `~coordinax.charts.LonCosLatSphericalTwoSphere`, and
    - `~coordinax.charts.MathSphericalTwoSphere`.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> atlas = cxm.HyperSphericalAtlas()
    >>> atlas.ndim
    2

    >>> cxc.sph2 in atlas
    True

    >>> cxc.lonlat_sph2 in atlas
    True

    >>> cxc.cart2d in atlas
    False

    >>> atlas.default_chart()
    SphericalTwoSphere(M=Sn(2))

    """

    ndim: int = 2
    """Dimension of the two-sphere."""

    def default_chart(self) -> AbstractChart[Any, Any, Any]:
        """Return the default chart (SphericalTwoSphere) for this atlas."""
        return SPHERICAL_ATLAS_DEFAULT_CHARTS[self.ndim]

    def has_chart(self, chart: AbstractChart[Any, Any, Any], /) -> bool:
        """Return whether the atlas supports the given chart.

        Examples
        --------
        >>> import coordinax.manifolds as cxm
        >>> import coordinax.charts as cxc

        >>> atlas = cxm.HyperSphericalAtlas()
        >>> atlas.has_chart(cxc.sph2)
        True

        >>> atlas.has_chart(cxc.cart3d)
        False

        """
        return (chart.ndim == self.ndim) and (
            type(chart) in SPHERICAL_ATLAS_ELIGIBLE_CHARTS
        )

    @classmethod
    def register(cls, registrant: CT, /) -> CT:
        """Register a chart class for HyperSphericalAtlas eligibility.

        Examples
        --------
        >>> import coordinax.manifolds as cxm
        >>> import coordinax.charts as cxc

        >>> cxc.SphericalTwoSphere in SPHERICAL_ATLAS_ELIGIBLE_CHARTS
        True

        """
        SPHERICAL_ATLAS_ELIGIBLE_CHARTS.add(registrant)
        return registrant
