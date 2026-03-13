"""Euclidean manifolds."""

__all__ = ("EuclideanManifold", "EuclideanAtlas")

from dataclasses import dataclass

from typing import Any, ClassVar, TypeVar, final

import coordinax.charts as cxc
import coordinax.metrics as cxm
from .base import AbstractAtlas, AbstractManifold

CT = TypeVar("CT", bound=type[cxc.AbstractChart[Any, Any]])


@final
@dataclass(frozen=True, slots=True)
class EuclideanAtlas(AbstractAtlas):
    """Atlas for Euclidean manifolds."""

    ndim: int
    """Dimension of the Euclidean manifold."""

    _ELIGIBLE_CHARTS: ClassVar[set[type[cxc.AbstractChart[Any, Any]]]] = set()

    def default_chart(self) -> cxc.AbstractChart[Any, Any]:
        # TODO: make this autodetect
        chart: cxc.AbstractChart[Any, Any]
        match self.ndim:
            case 0:
                chart = cxc.cart0d
            case 1:
                chart = cxc.cart1d
            case 2:
                chart = cxc.cart2d
            case 3:
                chart = cxc.cart3d
            case 6:
                chart = cxc.poincarepolar6d
            case _:
                msg = f"Euclidean({self.ndim}) is unsupported for now."
                raise ValueError(msg)
        return chart

    def supports(self, chart: cxc.AbstractChart[Any, Any]) -> bool:
        return chart.ndim == self.ndim and type(chart) in self._ELIGIBLE_CHARTS

    @classmethod
    def register(cls, registrant: CT, /) -> CT:
        """Register a class for Euclidean Atlas eligibility."""
        cls._ELIGIBLE_CHARTS.add(registrant)
        return registrant


for chart_cls in (
    cxc.Cart0D,
    cxc.Cart1D,
    cxc.Radial1D,
    cxc.Cart2D,
    cxc.Polar2D,
    cxc.Cart3D,
    cxc.Cylindrical3D,
    cxc.Spherical3D,
    cxc.LonLatSpherical3D,
    cxc.LonCosLatSpherical3D,
    cxc.MathSpherical3D,
    cxc.ProlateSpheroidal3D,
    cxc.CartND,
):
    _ = EuclideanAtlas.register(chart_cls)


# ===================================================================


@final
@dataclass(frozen=True, slots=True)
class EuclideanManifold(AbstractManifold):
    """Euclidean manifold with identity metric."""

    dim: int
    """Intrinsic dimension of the manifold."""

    def __init__(self, dim: int, /) -> None:
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "metric", cxm.EuclideanMetric(dim))
        object.__setattr__(self, "atlas", EuclideanAtlas(self.dim))
