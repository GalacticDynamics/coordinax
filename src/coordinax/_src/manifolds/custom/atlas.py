"""Customs manifolds."""

__all__ = ("CustomAtlas",)

from dataclasses import dataclass

from typing import Any, final

import jax

import coordinax.charts as cxc
from coordinax._src.manifolds.base import AbstractAtlas


@jax.tree_util.register_static
@final
@dataclass(frozen=True, slots=True)
class CustomAtlas(AbstractAtlas):
    r"""Atlas of explicitly registered charts for a custom manifold.

    ``CustomAtlas`` is an explicit atlas: chart membership is determined only
    by the set of chart classes provided at construction time.

    A chart belongs to the atlas iff:

    1. Its class is in ``charts``.
    2. Its dimensionality matches the atlas ``ndim``.

    The default chart must be one of the registered classes and defines the
    atlas dimension.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> atlas = cxm.CustomAtlas(
    ...     charts=(cxc.Cart2D, cxc.Polar2D),
    ...     chart_default=cxc.cart2d,
    ... )
    >>> atlas.ndim
    2
    >>> atlas.default_chart()
    Cart2D()
    >>> atlas.has_chart(cxc.polar2d)
    True
    >>> atlas.has_chart(cxc.cart3d)
    False

    """

    charts: tuple[type[cxc.AbstractChart[Any, Any]], ...]
    """Explicitly registered chart classes for this atlas."""

    chart_default: cxc.AbstractChart[Any, Any]
    """Stored default chart instance provided at construction."""

    def __post_init__(self) -> None:
        if len(set(self.charts)) != len(self.charts):
            raise ValueError("CustomAtlas chart classes must be unique.")

        if type(self.chart_default) not in self.charts:
            raise ValueError(
                "Default chart class "
                f"{type(self.chart_default).__name__} is not registered in this atlas."
            )

        # Check all the charts have the right dimensionality
        for chart_cls in self.charts:
            try:
                chart = chart_cls()
            except TypeError as e:
                raise ValueError(
                    "CustomAtlas chart classes must be zero-argument "
                    f"constructible, got {chart_cls.__name__}."
                ) from e

            if chart.ndim != self.ndim:
                raise ValueError(
                    f"Chart class {chart_cls.__name__} has ndim={chart.ndim} "
                    f"but expected {self.ndim}."
                )

    def default_chart(self) -> cxc.AbstractChart[Any, Any]:
        """Return the default chart for this atlas."""
        return self.chart_default

    @property
    def ndim(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self.chart_default.ndim

    def has_chart(self, chart: cxc.AbstractChart[Any, Any]) -> bool:
        """Return whether the atlas supports the given chart."""
        return type(chart) in self.charts and chart.ndim == self.ndim
