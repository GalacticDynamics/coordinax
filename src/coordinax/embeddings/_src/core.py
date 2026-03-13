"""Representations for embedded manifolds."""

__all__ = ("AbstractEmbedding", "EmbeddedManifold")

import abc
from dataclasses import dataclass

from typing import Any, TypeVar, final
from typing_extensions import override

import coordinax.charts as cxc
import coordinax.manifolds as cxma
import coordinax.metrics as cxm
from coordinax.internal.custom_types import CDict, OptUSys

AmbT = TypeVar("AmbT", bound=cxc.AbstractChart[Any, Any])


class AbstractEmbedding(metaclass=abc.ABCMeta):
    r"""Abstract base class representing a smooth embedding.

    An embedding represents a smooth injective map $$ \iota : M \hookrightarrow
    N $$ of an intrinsic manifold (with charts in {mod}`coordinax.charts`) into
    an ambient manifold represented by ``ambient`` (an
    {class}`~coordinax.charts.AbstractChart`).

    Conceptually, an embedding provides:

    - A smooth map from intrinsic coordinates ``q^i`` to ambient coordinates
      ``x^a = x^a(q)`` via :meth:`embed`.
    - A (possibly local) inverse or projection map from ambient coordinates back
      to intrinsic coordinates via :meth:`project`.
    - A realization map to the ambient Cartesian chart, typically used to induce
      a metric on the intrinsic manifold via

        $$ g_{ij}(q) = J^a{}_i J^b{}_j G_{ab}, $$

      where ``J^a{}_i = ∂x^a / ∂q^i`` and ``G_{ab}`` is the ambient metric.

    The attribute ``ambient`` owns the ambient chart in which the intrinsic
    manifold is embedded. In particular, ``realize_cartesian`` and
    ``unrealize_cartesian`` should delegate to the ambient chart's Cartesian
    realization when appropriate.

    Examples
    --------
    A concrete example is the embedding of ``SphericalTwoSphere`` into ``Spherical3D``:
    the intrinsic coordinates may be ``(θ, φ)`` on the unit 2-sphere, while the
    ambient coordinates are ``(r, θ, φ)`` with fixed radius ``r = R``. A
    concrete subclass can therefore:

    - Map ``(θ, φ) ↦ (R, θ, φ)`` in ``Spherical3D`` via :meth:`embed`.
    - Drop the radial component via :meth:`project`.
    - Realize to Cartesian coordinates by first embedding into ``Spherical3D``
      and then delegating to its Cartesian realization.

    Subclasses are responsible for implementing the coordinate-level maps;
    higher-level metric machinery (e.g. induced metrics) can be built on top of
    this interface.

    """

    ambient: cxc.AbstractChart[Any, Any]  # e.g. Cart3D

    def realize_cartesian(
        self,
        intrinsic: cxc.AbstractChart[Any, Any],
        q: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        raise NotImplementedError("TODO")

    def unrealize_cartesian(
        self,
        intrinsic: cxc.AbstractChart[Any, Any],
        x: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        raise NotImplementedError("TODO")


# ============================================================================


@final
@dataclass(frozen=True)
class EmbeddedManifold(cxma.AbstractManifold):
    r"""Embedded manifold.

    Examples
    --------
    Embed/project `{class}`~coordinax.charts.SphericalTwoSphere` through an ambient
    `{class}`~coordinax.charts.Spherical3D` chart::

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import coordinax.manifolds as cxma
    >>> import unxt as u

    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(2.0, "km"), ambient=cxc.sph3d)
    >>> manifold = cxe.EmbeddedManifold(
    ...     intrinsic=cxma.TwoSphereManifold(), embedding=emb,
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> sph = cxe.embed_point(manifold, p)
    >>> sph
    {'r': Quantity(Array(2., dtype=float64, ...), unit='km'),
     'theta': Angle(Array(1.57079633, dtype=float64, ...), unit='rad'),
     'phi': Angle(Array(0., dtype=float64, ...), unit='rad')}

    >>> p2 = cxe.project_point(manifold, sph)
    >>> p2
    {'theta': Angle(Array(1.57079633, dtype=float64, ...), unit='rad'),
     'phi': Angle(Array(0., dtype=float64, ...), unit='rad')}
    >>> jnp.allclose(p2["theta"].value, p["theta"].value)
    Array(True, dtype=bool)

    Embed/project through an ambient `{class}`~coordinax.charts.Cart3D` chart
    (routing via `{class}`~coordinax.charts.Spherical3D` internally)::

    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(2.0, "km"), ambient=cxc.cart3d)
    >>> manifold = cxe.EmbeddedManifold(
    ...     intrinsic=cxma.TwoSphereManifold(), embedding=emb,
    ... )
    >>> xyz = cxe.embed_point(manifold, p)
    >>> p3 = cxe.project_point(manifold, xyz)
    >>> p3
    {'theta': Quantity(Array(1.57079633, dtype=float64), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    >>> bool(jnp.allclose(u.ustrip("rad", p3["phi"]), u.ustrip("rad", p["phi"])))
    True

    """

    intrinsic: cxma.AbstractManifold
    embedding: AbstractEmbedding  # owns ambient + parameters

    @override
    def check_ndim(self) -> None:
        pass  # FIXME: implement the metric, then remove this shim.

    @override
    @property
    def metric(self) -> cxm.AbstractMetric:  # type: ignore[override]
        # induced metric from embedding (J^T G J)
        # return InducedMetric(self.embedding, ambient_metric=EuclideanMetric(...))
        raise NotImplementedError("TODO")

    @override
    @property
    def atlas(self) -> cxma.AbstractAtlas:  # type: ignore[override]
        return self.intrinsic.atlas

    def realize_cartesian(
        self,
        chart: cxc.AbstractChart[Any, Any],
        data: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        self.check_chart(chart)
        return self.embedding.realize_cartesian(chart, data, usys=usys)

    def unrealize_cartesian(
        self,
        chart: cxc.AbstractChart[Any, Any],
        data: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        self.check_chart(chart)
        return self.embedding.unrealize_cartesian(chart, data, usys=usys)
