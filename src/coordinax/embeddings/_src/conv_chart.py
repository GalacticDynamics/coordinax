"""Representations for embedded manifolds."""

__all__ = ("EmbeddedChart",)

from dataclasses import dataclass

from typing import Any

import coordinax.charts as cxc
from .core import AbstractEmbedding
from coordinax.internal.custom_types import CDict, Ds, Ks, OptUSys


@dataclass(frozen=True)
class EmbeddedChart(cxc.AbstractChart[Ks, Ds]):
    """Chart for intrinsic coordinates on an embedded manifold.

    This is a convenience wrapper that combines an intrinsic chart with an
    embedding to an ambient Cartesian chart. It provides the same component and
    dimension information as the intrinsic chart, but also provides a
    realization map to Cartesian coordinates via the embedding.

    The more correct way to represent an embedded manifold is with
    {class}`~coordinax.embeddings.EmbeddedManifold`.

    Examples
    --------
    Embed/project `{class}`~coordinax.charts.SphericalTwoSphere` through an ambient
    `{class}`~coordinax.charts.Spherical3D` chart::

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u

    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(2.0, "km"), ambient=cxc.sph3d)
    >>> chart = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb)
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> sph = cxe.embed_point(chart, p)
    >>> sph
    {'r': Quantity(Array(2., dtype=float64, ...), unit='km'),
     'theta': Angle(Array(1.57079633, dtype=float64, ...), unit='rad'),
     'phi': Angle(Array(0., dtype=float64, ...), unit='rad')}

    >>> p2 = cxe.project_point(chart, sph)
    >>> p2
    {'theta': Angle(Array(1.57079633, dtype=float64, ...), unit='rad'),
     'phi': Angle(Array(0., dtype=float64, ...), unit='rad')}
    >>> jnp.allclose(p2["theta"].value, p["theta"].value)
    Array(True, dtype=bool)

    Embed/project through an ambient `{class}`~coordinax.charts.Cart3D` chart
    (routing via `{class}`~coordinax.charts.Spherical3D` internally)::

    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(2.0, "km"), ambient=cxc.cart3d)
    >>> chart = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb)
    >>> xyz = cxe.embed_point(chart, p)
    >>> p3 = cxe.project_point(chart, xyz)
    >>> p3
    {'theta': Quantity(Array(1.57079633, dtype=float64), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    >>> bool(jnp.allclose(u.ustrip("rad", p3["phi"]), u.ustrip("rad", p["phi"])))
    True

    """

    intrinsic: cxc.AbstractChart[Ks, Ds]  # e.g. SphericalTwoSphere
    """The intrinsic chart whose coordinates are being embedded."""

    embedding: AbstractEmbedding  # holds ambient and and any embedding params
    """The embedding that defines the map to the ambient chart."""

    @property
    def ambient(self) -> cxc.AbstractChart[Any, Any]:
        return self.embedding.ambient

    # ===================================================
    # Chart API

    @property
    def components(self) -> Ks:
        return self.intrinsic.components

    @property
    def coord_dimensions(self) -> Ds:
        return self.intrinsic.coord_dimensions

    @property
    def cartesian(self) -> cxc.AbstractChart:  # type: ignore[type-arg]
        return self.ambient.cartesian

    def realize_cartesian(self, q: CDict, *, usys: OptUSys = None) -> CDict:
        return self.embedding.realize_cartesian(self.intrinsic, q, usys=usys)

    def unrealize_cartesian(self, x: CDict, *, usys: OptUSys = None) -> CDict:
        return self.embedding.unrealize_cartesian(self.intrinsic, x, usys=usys)

    def __hash__(self) -> int:
        return hash((self.__class__, self.intrinsic, self.embedding))
