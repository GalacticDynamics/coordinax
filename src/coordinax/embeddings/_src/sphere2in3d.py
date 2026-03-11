"""Embedding of the 2-sphere in 3D."""

__all__ = ("TwoSphereIn3D", "embedded_twosphere")

from dataclasses import dataclass, field

from typing import Any

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxma
from .core import AbstractEmbedding, EmbeddedManifold
from coordinax.internal.custom_types import CDict, OptUSys


@dataclass(frozen=True, slots=True)
class TwoSphereIn3D(AbstractEmbedding):
    r"""Embedding of ``cxc.SphericalTwoSphere`` as a 2-sphere in a 3D ambient chart.

    This embedding models a 2-sphere of fixed radius $R$ as the hypersurface $r
    = R$ in 3D spherical coordinates $(r, \theta, \phi)$. The intrinsic chart is
    therefore expected to have components $(\theta, \phi)$.

    The key design choice is that **all** coordinate-level embedding and
    projection operations are defined via an intermediate 3D spherical chart
    ({class}`~coordinax.charts.Spherical3D`), regardless of which ambient chart
    is selected. In particular:

    - If ``ambient`` is `{class}`~coordinax.charts.Spherical3D`, then
      :meth:`embed` returns spherical coordinates ``(r, theta, phi)`` and
      :meth:`project` expects the same.
    - If ``ambient`` is `{class}`~coordinax.charts.Cart3D`, then :meth:`embed`
      performs ``SphericalTwoSphere -> Spherical3D -> Cart3D`` and returns Cartesian
      coordinates ``(x, y, z)``; :meth:`project` performs ``Cart3D ->
      Spherical3D -> SphericalTwoSphere``.

    Parameters
    ----------
    radius
        Sphere radius ``R``.
    ambient
        Ambient chart. Defaults to `{class}`~coordinax.charts.Spherical3D`.

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

    radius: float | u.AbstractQuantity = field()
    ambient: cxc.AbstractChart[Any, Any] = field(default=cxc.sph3d)

    @property
    def intrinsic(self) -> cxc.AbstractChart[Any, Any]:
        """The intrinsic chart for this embedding is always SphericalTwoSphere."""
        return cxc.sph2

    def check_intrinsic_chart(self, chart: cxc.AbstractChart[Any, Any], /) -> None:
        if not isinstance(chart, cxc.SphericalTwoSphere):
            msg = (
                "TwoSphereIn3D expects an intrinsic chart of type SphericalTwoSphere; "
                f"received {type(chart).__name__}."
            )
            raise TypeError(msg)

    def _embed_sph3d(
        self,
        intrinsic: cxc.AbstractChart[Any, Any],
        q: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        """Embed ``SphericalTwoSphere`` intrinsic coords into ``Spherical3D`` coords."""
        del usys
        self.check_intrinsic_chart(intrinsic)
        return {"r": self.radius, "theta": q["theta"], "phi": q["phi"]}

    def _project_sph3d(
        self,
        intrinsic: cxc.AbstractChart[Any, Any],
        x_sph: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        """Project ``Spherical3D`` onto ``SphericalTwoSphere`` intrinsic coords."""
        del usys
        self.check_intrinsic_chart(intrinsic)
        return {"theta": x_sph["theta"], "phi": x_sph["phi"]}

    def realize_cartesian(
        self,
        intrinsic: cxc.AbstractChart[Any, Any],
        q: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        """Realize intrinsic coords into ambient Cartesian coordinates."""
        # Embed into Spherical3D, then realize to Cartesian.
        x_sph = self._embed_sph3d(intrinsic, q, usys=usys)
        return cxc.sph3d.realize_cartesian(x_sph, usys=usys)

    def unrealize_cartesian(
        self,
        intrinsic: cxc.AbstractChart[Any, Any],
        x: CDict,
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        """Unrealize ambient Cartesian coordinates into intrinsic coords."""
        x_sph = cxc.sph3d.unrealize_cartesian(x, usys=usys)
        return self._project_sph3d(intrinsic, x_sph, usys=usys)


# ============================================================================


def embedded_twosphere(
    radius: float | u.AbstractQuantity,
    ambient: cxc.AbstractChart[Any, Any] = cxc.sph3d,
) -> EmbeddedManifold:
    """Create an {class}`coordinax.embeddings.EmbeddedManifold` for the two-sphere.

    This is a convenience helper that constructs an
    {class}`coordinax.embeddings.EmbeddedManifold` with
    ``intrinsic=TwoSphereManifold()`` and ``embedding=TwoSphereIn3D(radius,
    ambient)``.

    Parameters
    ----------
    radius
        Sphere radius.
    ambient
        Ambient chart for the embedding.  Defaults to
        `{class}`~coordinax.charts.Spherical3D`.

    Returns
    -------
    EmbeddedManifold
        An embedded manifold pairing the two-sphere manifold with the
        {class}`coordinax.embeddings.TwoSphereIn3D` embedding.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u

    Default ambient (Spherical3D)::

    >>> manifold = cxe.embedded_twosphere(radius=u.Q(2.0, "km"))
    >>> manifold.embedding.ambient
    Spherical3D...

    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> sph = cxe.embed_point(manifold, p)
    >>> sph
    {'r': Quantity(Array(2., dtype=float64, ...), unit='km'),
     'theta': Angle(Array(1.57079633, dtype=float64, ...), unit='rad'),
     'phi': Angle(Array(0., dtype=float64, ...), unit='rad')}

    With Cartesian ambient::

    >>> manifold = cxe.embedded_twosphere(
    ...     radius=u.Q(2.0, "km"), ambient=cxc.cart3d,
    ... )
    >>> xyz = cxe.embed_point(manifold, p)
    >>> p3 = cxe.project_point(manifold, xyz)
    >>> p3
    {'theta': Quantity(Array(1.57079633, dtype=float64), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    """
    return EmbeddedManifold(
        intrinsic=cxma.TwoSphereManifold(),
        embedding=TwoSphereIn3D(radius=radius, ambient=ambient),
    )
