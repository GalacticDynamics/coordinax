"""Spacetime representations with non-Euclidean metrics."""

__all__ = ("SpaceTimeCT", "spacetimect")

from dataclasses import KW_ONLY, field, replace

from jaxtyping import Float
from typing import Any, cast, final
from typing_extensions import override

import jax.tree_util as jtu

import unxt as u

from .base import AbstractChart, AbstractFixedComponentsChart, chart_dataclass_decorator
from .custom_types import CDict, Ds, Ks
from .d1 import time1d
from .d3 import cart3d
from .product import AbstractFlatCartesianProductChart

C_DEFAULT = u.StaticQuantity(299_792.458, "km/s")


@jtu.register_static
@final
@chart_dataclass_decorator
class SpaceTimeCT(AbstractFlatCartesianProductChart[Ks, Ds]):
    r"""4D spacetime rep with components ``(ct, x, y, z)`` and Minkowski metric.

    This is a Cartesian product chart: SpaceTimeCT(spatial_chart) ≡ time1d x
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

    >>> cxc.SpaceTimeCT()
    SpaceTimeCT()

    >>> cxc.SpaceTimeCT(cxc.sph3d)
    SpaceTimeCT(spatial_chart=Spherical3D())

    """

    spatial_chart: AbstractFixedComponentsChart[Any, Any] = field(default=cart3d)  # pylint: disable=invalid-field-call
    """Spatial part of the representation. Defaults: `coordinax.charts.cart3d`."""

    _: KW_ONLY
    c: Float[u.StaticQuantity["speed"], ""] = field(default=C_DEFAULT)  # pylint: disable=invalid-field-call
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""

    @property
    def time_chart(self) -> AbstractChart[Any, Any]:
        """Time factor chart (always `time1d`)."""
        return time1d

    @override
    @property
    def factors(self) -> tuple[AbstractChart[Any, Any], ...]:
        """Return (time1d, spatial_chart) as required by product chart spec."""
        return (self.time_chart, self.spatial_chart)

    @override
    @property
    def factor_names(self) -> tuple[str, ...]:
        """Factor names are ('time', 'space')."""
        return ("time", "space")

    @property
    def components(self) -> Ks:
        # Override to use "ct" instead of "t" for the time component
        return cast("Ks", ("ct", *self.spatial_chart.components))

    @property
    def coord_dimensions(self) -> Ds:
        # Override to use "length" for ct dimension
        return cast("Ds", ("length", *self.spatial_chart.coord_dimensions))

    @override
    def split_components(self, p: CDict) -> tuple[CDict, CDict]:
        """Split CDict by factors, keeping 'ct' for time factor.

        SpaceTimeCT uses 'ct' for the time component. The split returns
        factor dicts with their native keys ('ct' for time, spatial keys for space).
        """
        time_dict = {"ct": p["ct"]}
        spatial_dict = {k: p[k] for k in self.spatial_chart.components}
        return (time_dict, spatial_dict)

    @override
    def merge_components(self, parts: tuple[CDict, CDict], /) -> CDict:  # ty: ignore[invalid-method-override]
        """Merge factor CDicts back into SpaceTimeCT components.

        Expects time factor dict with 'ct' key, spatial factor dict with spatial keys.
        """
        return {**parts[0], **parts[1]}

    @override
    @property
    def cartesian(self) -> "SpaceTimeCT":
        """Get a Cartesian-chart version of the given spacetime chart.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> rep = cxc.SpaceTimeCT(cxc.sph3d)
        >>> rep
        SpaceTimeCT(spatial_chart=Spherical3D())

        >>> rep.cartesian  # default is Cart3D
        SpaceTimeCT()

        """
        spatial_cart = self.spatial_chart.cartesian
        # Return same object if already cartesian
        if spatial_cart == self.spatial_chart:
            return self
        return replace(self, spatial_chart=spatial_cart)


spacetimect = SpaceTimeCT(spatial_chart=cart3d)
"""Default SpaceTimeCT with Cartesian spatial chart (i.e. Cartesian 4D spacetime).

>>> import coordinax.charts as cxc
>>> cxc.spacetimect
SpaceTimeCT()

>>> cxc.spacetimect.cartesian is cxc.spacetimect
True

"""
