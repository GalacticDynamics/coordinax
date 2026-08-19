"""Galilean spacetime manifold, atlas, metric, and chart."""

__all__ = ("galilean_spacetime", "GalileanCT", "galileanct")

from dataclasses import KW_ONLY, field, replace

from jaxtyping import Float
from typing import Any, ClassVar, Final, cast, final, override

import numpy as np

import unxt as u

from .chart import AbstractFlatCartesianProductChart
from .manifold import CartesianProductManifold
from coordinax._src.base import (
    AbstractChart,
    AbstractManifold,
    AbstractStaticFixedComponentsChart,
    chart_dataclass_decorator,
)
from coordinax._src.charts.d1 import time1d
from coordinax._src.charts.d3 import cart3d
from coordinax._src.custom_types import Ds, Ks
from coordinax._src.euclidean.manifold import R1, R3
from coordinaxs.api.custom_types import CDict

galilean_spacetime: Final = CartesianProductManifold(
    factors=(R1, R3), factor_names=("ct", "space")
)
r"""The 4-dimensional Galilean spacetime manifold, $\mathbb{R} \times \mathbb{R}^3$."""


C_DEFAULT = u.StaticQuantity(np.array(299_792.458), "km/s")


_MSG_TIME_DEPENDENT_SPATIAL = (
    "GalileanCT is a product, `time1d x spatial_chart`, but this spatial "
    "chart is time-dependent, so there is no single spatial factor to "
    "multiply by: its coordinates label different points of space at "
    "different times. That is a fibre bundle over the time axis rather than a "
    "product, and the datum a product cannot carry is the connection -- the "
    "velocity of the coordinate frame, equivalently which spatial point at "
    "`t` counts as the same point at `t'`. Galilean spacetime supplies no "
    "canonical choice (that is what it means to have no absolute space), so "
    "it has to be stated rather than assumed. Evaluate the chart on one time "
    "slice, e.g. with `AtTime`, if a product is what you want."
)


@final
@chart_dataclass_decorator
class GalileanCT(AbstractFlatCartesianProductChart[Ks, Ds]):
    r"""4D chart with components ``(ct, x, y, z)`` on Galilean spacetime.

    This is a Cartesian product chart: GalileanCT(spatial_chart) ≡ time1d x
    spatial_chart

    **The product is a choice, not a given.** Galilean spacetime is
    intrinsically a *fibre bundle* over the time axis: the time function is
    canonical, and the fibres are the simultaneity slices, but there is no
    canonical identification between the slices -- that absence is what "no
    absolute space" means. Writing the chart as a product therefore fixes one:
    a notion of which spatial point at ``t`` is the same point at ``t'``,
    i.e. a rest frame. A Galilean boost ``x -> x + vt`` acts precisely by
    changing it, so the factorisation this class asserts is not boost
    invariant.

    That is legitimate and useful for an inertial frame with a spatial chart
    that does not move, which is what this class is for. It is *not* the
    general case: a spatial chart whose coordinates move with ``t`` needs the
    connection the product cannot carry -- the frame velocity -- and is
    rejected in ``__post_init__``.

    Contrast Minkowski, where a boost mixes ``ct`` with ``x`` so that neither
    projection is canonical and no preferred foliation exists at all; that is
    why `coordinax.charts.MinkowskiCT` is a single chart rather than a
    product. The two spacetimes differ in *which* factorisation is unavailable,
    not merely in the metric signature.

    The time axis coordinate is ``x^0 = ct`` — a *length* — stored directly on
    the chart (component ``"ct"``, dimension ``"length"``). The underlying time
    factor is the canonical 1D chart ``time1d`` (native component ``"t"``); this
    chart carries the length-valued ``ct`` on its axis rather than a physical
    time, consistent with the Euclidean ``R1`` time factor and the
    ``diag(1, 1, 1, 1)`` product metric. Because ``ct`` *is* the coordinate,
    ``split_components``/``merge_components`` simply re-key ``"ct" <-> "t"``
    without a runtime ``c`` multiply. ``c`` is retained to form ``ct`` from a
    physical time at construction/conversion boundaries.

    Mathematical definition:
    $$
       x^0 = ct,\quad x^i = \text{spatial components}
    $$

    The underlying manifold is ``galilean_spacetime = R1 x R3`` (both Euclidean),
    so the induced product metric is ``diag(1, 1, 1, 1)`` — Galilean spacetime
    carries no invariant Lorentzian spacetime metric. (This is *not* Minkowski
    ``diag(-1, 1, 1, 1)``; a Lorentzian variant would need a different manifold.)

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

    spatial_chart: AbstractStaticFixedComponentsChart[Any, Any, Any] = field(
        default=cart3d
    )  # pylint: disable=invalid-field-call
    """Spatial part of the representation (defaults to `coordinax.charts.cart3d`)."""

    _: KW_ONLY
    c: Float[u.StaticQuantity, ""] = field(default=C_DEFAULT)  # pylint: disable=invalid-field-call
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""

    M: ClassVar[AbstractManifold]

    def __post_init__(self) -> None:
        """Reject a *time-dependent* spatial chart, which is not a factor.

        The array check inherited from `AbstractStaticChart` is about JAX
        safety, not geometry: it stops a live array hiding inside a static
        node. It does not answer this question, and a time-dependent chart can
        pass it -- a `TubularChart` over a two-argument curve holds only the
        curve, a non-array leaf.

        The geometric question is separate. `GalileanCT` asserts a *product*,
        ``time1d x spatial_chart``, and a product asserts that the spatial
        factor is the same at every time. A chart whose coordinates move with
        ``t`` makes that false: what it describes is a **fibre bundle** over
        the time axis, whose extra datum is a connection -- the frame
        velocity. There is no canonical choice of one, so it cannot be
        supplied silently here (see the class docstring).
        """
        super().__post_init__()
        if getattr(self.spatial_chart, "is_time_dependent", False):
            raise TypeError(_MSG_TIME_DEPENDENT_SPATIAL)

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
        """Factor names are ('ct', 'space'), matching ``galilean_spacetime``."""
        return ("ct", "space")

    @override
    def split_components(self, p: CDict) -> tuple[CDict, CDict]:
        """Split CDict by factors, re-keying the ``"ct"`` axis to ``time1d``'s ``"t"``.

        The returned factor dicts use each factor's native keys: ``"t"`` for the
        time factor (holding the length-valued ``x^0 = ct``) and the spatial
        keys for the space factor. This is a pure re-key — ``ct`` is already the
        coordinate, so no ``c`` conversion is applied (see the class docstring).
        """
        # The time factor is `time1d`, whose native component is "t"; map the
        # chart's "ct" value onto it so factor operations (metric, pt_map)
        # validate against the real factor chart.
        time_dict = {"t": p["ct"]}
        spatial_dict = {k: p[k] for k in self.spatial_chart.components}
        return (time_dict, spatial_dict)

    @override
    def merge_components(self, parts: tuple[CDict, CDict], /) -> CDict:  # ty: ignore[invalid-method-override]
        """Merge factor CDicts back into GalileanCT components.

        The time factor dict uses `time1d`'s native "t" key (holding the
        length-valued ``x^0 = ct``); re-key it back to "ct". Pure re-key — no
        ``c`` conversion (see the class docstring).
        """
        return {"ct": parts[0]["t"], **parts[1]}

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
