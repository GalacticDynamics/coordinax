"""Two-sphere manifold."""

__all__ = ("TwoSphereManifold", "TwoSphereAtlas")

from dataclasses import dataclass

from typing import Any, ClassVar, TypeVar, final

import coordinax.charts as cxc
import coordinax.metrics as cxm
from .base import AbstractAtlas, AbstractManifold

CT = TypeVar("CT", bound=type[cxc.AbstractChart[Any, Any]])


@final
@dataclass(frozen=True, slots=True)
class TwoSphereAtlas(AbstractAtlas):
    """Atlas for the two-sphere manifold.

    The two-sphere atlas supports charts that cover the 2-sphere, such as
    :class:`~coordinax.charts.SphericalTwoSphere`,
    :class:`~coordinax.charts.LonLatSphericalTwoSphere`,
    :class:`~coordinax.charts.LonCosLatSphericalTwoSphere`, and
    :class:`~coordinax.charts.MathSphericalTwoSphere`.

    Examples
    --------
    >>> import coordinax.manifolds as cxma
    >>> import coordinax.charts as cxc

    >>> atlas = cxma.TwoSphereAtlas()
    >>> atlas.ndim
    2

    >>> atlas.supports(cxc.sph2)
    True

    >>> atlas.supports(cxc.lonlat_sph2)
    True

    >>> atlas.supports(cxc.cart2d)
    False

    >>> atlas.default_chart()
    SphericalTwoSphere...

    """

    ndim: int = 2
    """Dimension of the two-sphere (always 2)."""

    _ELIGIBLE_CHARTS: ClassVar[set[type[cxc.AbstractChart[Any, Any]]]] = set()

    def default_chart(self) -> cxc.AbstractChart[Any, Any]:
        """Return the default chart (SphericalTwoSphere) for this atlas."""
        return cxc.sph2

    def supports(self, chart: cxc.AbstractChart[Any, Any]) -> bool:
        """Return whether the atlas supports the given chart."""
        return type(chart) in self._ELIGIBLE_CHARTS

    @classmethod
    def register(cls, registrant: CT, /) -> CT:
        """Register a chart class for TwoSphereAtlas eligibility.

        Examples
        --------
        >>> import coordinax.manifolds as cxma
        >>> import coordinax.charts as cxc

        >>> cxc.SphericalTwoSphere in cxma.TwoSphereAtlas._ELIGIBLE_CHARTS
        True

        """
        cls._ELIGIBLE_CHARTS.add(registrant)
        return registrant


for _chart_cls in (
    cxc.SphericalTwoSphere,
    cxc.LonLatSphericalTwoSphere,
    cxc.LonCosLatSphericalTwoSphere,
    cxc.MathSphericalTwoSphere,
):
    _ = TwoSphereAtlas.register(_chart_cls)


@final
@dataclass(frozen=True, slots=True)
class TwoSphereManifold(AbstractManifold):
    r"""The unit two-sphere $S^2$ as a Riemannian manifold.

    This manifold has the intrinsic metric
    $g = \mathrm{diag}(1,\;\sin^2\!\theta)$ in standard $(\theta,\phi)$
    coordinates (see :class:`~coordinax.metrics.SphereMetric`), and an atlas
    containing :class:`~coordinax.charts.SphericalTwoSphere`.

    Examples
    --------
    >>> import coordinax.manifolds as cxma
    >>> import coordinax.charts as cxc

    >>> S2 = cxma.TwoSphereManifold()
    >>> S2.ndim
    2

    >>> S2.has_chart(cxc.sph2)
    True

    >>> S2.default_chart
    SphericalTwoSphere...

    """

    def __init__(self) -> None:
        object.__setattr__(self, "metric", cxm.SphereMetric())
        object.__setattr__(self, "atlas", TwoSphereAtlas())
