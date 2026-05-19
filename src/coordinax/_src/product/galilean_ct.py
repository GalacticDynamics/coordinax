"""Galilean spacetime manifold, atlas, metric, and chart."""

__all__ = ("galilean_spacetime", "GalileanCT", "galileanct")

from dataclasses import KW_ONLY, field, replace

from jaxtyping import Float
from typing import Any, ClassVar, Final, cast, final
from typing_extensions import override

import jax.tree_util as jtu
import numpy as np

import unxt as u

from .chart import AbstractFlatCartesianProductChart
from .manifold import CartesianProductManifold
from coordinax._src.base import (
    AbstractChart,
    AbstractFixedComponentsChart,
    AbstractManifold,
    chart_dataclass_decorator,
)
from coordinax._src.charts.d1 import time1d
from coordinax._src.charts.d3 import cart3d
from coordinax._src.custom_types import CDict, Ds, Ks
from coordinax._src.euclidean.manifold import R1, R3

galilean_spacetime: Final = CartesianProductManifold(
    factors=(R1, R3),
    factor_names=("ct", "space"),
)
r"""The 4-dimensional Galilean spacetime manifold, $\mathbb{R} \times \mathbb{R}^3$."""


C_DEFAULT = u.StaticQuantity(np.array(299_792.458), "km/s")


@jtu.register_static
@final
@chart_dataclass_decorator
class GalileanCT(AbstractFlatCartesianProductChart[Ks, Ds]):
    r"""4D spacetime rep with components ``(ct, x, y, z)`` and Minkowski metric.

    This is a Cartesian product chart: GalileanCT(spatial_chart) ≡ time1d x
    spatial_chart

    The time component is always the canonical 1D time chart `time1d` with
    component "t". The time coordinate is automatically converted to ct using
    the speed of light.

    Mathematical definition:
    $$
       x^0 = ct,\quad x^i = \text{spatial components}
       \\
       g = \mathrm{diag}(-1, 1, 1, 1) \quad \text{(signature } - + + +)
    $$

    Parameters
    ----------
    spatial_chart
        Spatial position rep supplying component names and dimensions.
    c
        Speed of light used to form ``ct`` from ``t`` (defaults to
        ``Quantity(299_792.458, "km/s")``).

    Returns
    -------
    Rep
        Representation with components ``("ct", *spatial_chart.components)`` and
        dimensions ``("length", *spatial_chart.coord_dimensions)``.

    Notes
    -----
    - The first factor is always `time1d`; the time chart is not user-selectable.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc

    >>> cxc.GalileanCT()
    GalileanCT()

    >>> cxc.GalileanCT(cxc.sph3d)
    GalileanCT(spatial_chart=Spherical3D(M=Rn(3)))

    """

    spatial_chart: AbstractFixedComponentsChart[Any, Any, Any] = field(default=cart3d)  # pylint: disable=invalid-field-call
    """Spatial part of the representation. Defaults: `coordinax.charts.cart3d`."""

    _: KW_ONLY
    c: Float[u.StaticQuantity["speed"], ""] = field(default=C_DEFAULT)  # pylint: disable=invalid-field-call
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""

    M: ClassVar[AbstractManifold]

    @property
    def M(self) -> AbstractManifold:
        """The manifold this chart belongs to, derived from the spatial chart."""
        return galilean_spacetime

    # ===============================================================
    # Product Chart API

    @property
    def time_chart(self) -> AbstractChart[Any, Any, Any]:
        """Time factor chart (always `time1d`)."""
        return time1d

    @override
    @property
    def factors(self) -> tuple[AbstractChart[Any, Any, Any], ...]:
        """Return (time1d, spatial_chart) as required by product chart spec."""
        return (self.time_chart, self.spatial_chart)

    @override
    @property
    def factor_names(self) -> tuple[str, ...]:
        """Factor names are ('time', 'space')."""
        return ("time", "space")

    @override
    def split_components(self, p: CDict) -> tuple[CDict, CDict]:
        """Split CDict by factors, keeping 'ct' for time factor.

        GalileanCT uses 'ct' for the time component. The split returns
        factor dicts with their native keys ('ct' for time, spatial keys for space).
        """
        time_dict = {"ct": p["ct"]}
        spatial_dict = {k: p[k] for k in self.spatial_chart.components}
        return (time_dict, spatial_dict)

    @override
    def merge_components(self, parts: tuple[CDict, CDict], /) -> CDict:  # ty: ignore[invalid-method-override]
        """Merge factor CDicts back into GalileanCT components.

        Expects time factor dict with 'ct' key, spatial factor dict with spatial keys.
        """
        return {**parts[0], **parts[1]}

    # ===============================================================
    # Chart API

    @property
    def components(self) -> Ks:
        # Override to use "ct" instead of "t" for the time component
        return cast("Ks", ("ct", *self.spatial_chart.components))

    @property
    def coord_dimensions(self) -> Ds:
        # Override to use "length" for ct dimension
        return cast("Ds", ("length", *self.spatial_chart.coord_dimensions))

    @override
    @property
    def cartesian(self) -> "GalileanCT[Any, Any]":
        """Get a Cartesian-chart version of the given spacetime chart.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> rep = cxc.GalileanCT(cxc.sph3d)
        >>> rep
        GalileanCT(spatial_chart=Spherical3D(M=Rn(3)))

        >>> rep.cartesian  # default is Cart3D
        GalileanCT(spatial_chart=Cart3D(M=Rn(3)))

        """
        spatial_cart = self.spatial_chart.cartesian
        # Return same object if already cartesian
        if spatial_cart == self.spatial_chart:
            return self
        return replace(self, spatial_chart=spatial_cart)


galileanct = GalileanCT(spatial_chart=cart3d)
"""Default GalileanCT with Cartesian spatial chart (i.e. Cartesian 4D spacetime).

>>> import coordinax.charts as cxc
>>> cxc.galileanct
GalileanCT()

>>> cxc.galileanct.cartesian is cxc.galileanct
True

"""
