"""Two-sphere manifold."""

__all__ = ("TwoSphereIn3D", "embedded_twosphere")

import dataclasses

from typing import Any, final

import unxt as u

import coordinax.charts as cxc
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.embedded import (
    AbstractEmbeddingMap,
    AmbientT,
    EmbeddedManifold,
    IntrinsicT,
)
from coordinax._src.euclidean.manifold import R3
from coordinax._src.spherical.manifold import S2


@final
@dataclasses.dataclass(frozen=True, slots=True)
class TwoSphereIn3D(AbstractEmbeddingMap[IntrinsicT, AmbientT]):
    r"""Embedding of ``cxc.SphericalTwoSphere`` as a 2-sphere in a 3D ambient chart.

    This embedding models a 2-sphere of fixed radius $R$ as the hypersurface $r
    = R$ in 3D spherical coordinates $(r, \theta, \phi)$. The intrinsic chart is
    therefore expected to have components $(\theta, \phi)$.

    The key design choice is that **all** coordinate-level embedding and
    projection operations are defined via an intermediate 3D spherical chart
    ({class}`~coordinax.charts.Spherical3D`), regardless of which ambient chart
    is selected. In particular:

    - If ``ambient`` is :class:`~coordinax.charts.Spherical3D`, then
      {meth}`embed` returns spherical coordinates ``(r, theta, phi)`` and
      {meth}`project` expects the same.
    - If ``ambient`` is :class:`~coordinax.charts.Cart3D`, then {meth}`embed`
      performs ``SphericalTwoSphere -> Spherical3D -> Cart3D`` and returns Cartesian
      coordinates ``(x, y, z)``; {meth}`project` performs ``Cart3D ->
      Spherical3D -> SphericalTwoSphere``.

    Parameters
    ----------
    radius
        Sphere radius ``R``.
    ambient
        Ambient chart. Defaults to :class:`~coordinax.charts.Spherical3D`.

    Examples
    --------
    Embed/project :class:`~coordinax.charts.SphericalTwoSphere` through an ambient
    :class:`~coordinax.charts.Spherical3D` chart:

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> sph = cxm.pt_embed(p, chart)
    >>> sph
    {'r': Q(2., 'km'), 'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    >>> p2 = cxm.pt_project(sph, chart)
    >>> p2
    {'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}
    >>> jnp.allclose(p2["theta"].value, p["theta"].value)
    Array(True, dtype=bool)

    Embed/project through an ambient :class:`~coordinax.charts.Cart3D` chart
    (routing via :class:`~coordinax.charts.Spherical3D` internally):

    >>> emb = cxm.TwoSphereIn3D(radius=u.Q(2.0, "km"), ambient=cxc.cart3d)
    >>> chart = cxm.EmbeddedChart(emb)
    >>> xyz = cxm.pt_embed(p, chart)
    >>> sorted(xyz)
    ['x', 'y', 'z']
    >>> bool(jnp.allclose(u.ustrip("km", xyz["x"]), 2.0, atol=1e-6))
    True

    >>> p3 = cxm.pt_project(xyz, chart)
    >>> bool(jnp.allclose(u.ustrip("rad", p3["phi"]), u.ustrip("rad", p["phi"])))
    True

    """

    radius: u.AbstractQuantity | float | int = dataclasses.field()
    ambient: cxc.AbstractChart[Any, Any, Any] = dataclasses.field(default=cxc.sph3d)

    @property
    def intrinsic(self) -> cxc.AbstractChart[Any, Any, Any]:
        """The intrinsic chart is always `coordinax.charts.SphericalTwoSphere`."""
        return cxc.sph2

    def embed(self, q: CDict, /, *, usys: OptUSys = None) -> CDict:
        """Embed ``SphericalTwoSphere`` intrinsic coords into the ambient chart."""
        x_sph: CDict = {"r": self.radius, "theta": q["theta"], "phi": q["phi"]}
        # A ``Spherical3D`` ambient (any instance — the chart is not a singleton)
        # already uses ``(r, theta, phi)``, so no coordinate transform is needed.
        if isinstance(self.ambient, cxc.Spherical3D):
            return x_sph
        out: CDict = cxc.pt_map(  # ty: ignore[invalid-assignment]
            x_sph, cxc.sph3d, self.ambient, usys=usys
        )
        return out

    def project(self, x: CDict, /, *, usys: OptUSys = None) -> CDict:
        """Project ambient coords onto ``SphericalTwoSphere`` intrinsic coords."""
        x_sph: CDict = x
        if not isinstance(self.ambient, cxc.Spherical3D):
            x_sph = cxc.pt_map(  # ty: ignore[invalid-assignment]
                x, self.ambient, cxc.sph3d, usys=usys
            )
        return {"theta": x_sph["theta"], "phi": x_sph["phi"]}


def embedded_twosphere(
    radius: float | u.AbstractQuantity,
    ambient: cxc.AbstractChart[Any, Any, Any] = cxc.sph3d,
) -> EmbeddedManifold:
    """Create an `coordinax.manifolds.EmbeddedManifold` for the two-sphere.

    This is a convenience helper that constructs an
    `coordinax.manifolds.EmbeddedManifold` with
    ``intrinsic=HyperSphericalManifold()`` and ``embedding=TwoSphereIn3D(radius,
    ambient)``.

    Parameters
    ----------
    radius
        Sphere radius.
    ambient
        Ambient chart for the embedding.  Defaults to
        `coordinax.charts.Spherical3D`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    Default ambient (Spherical3D):

    >>> M = cxm.embedded_twosphere(radius=u.Q(2.0, "km"))
    >>> M
    EmbeddedManifold(intrinsic=HyperSphericalManifold(...),
                     ambient=Rn(3),
                     embed_map=TwoSphereIn3D(radius=Q(2., 'km'),
                                             ambient=Spherical3D(M=Rn(3))))

    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> sph = cxm.pt_embed(p, M)
    >>> sph
    {'r': Q(2., 'km'), 'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    With Cartesian ambient the embedding returns ``(x, y, z)``:

    >>> M = cxm.embedded_twosphere(radius=u.Q(2.0, "km"), ambient=cxc.cart3d)
    >>> xyz = cxm.pt_embed(p, M)
    >>> sorted(xyz)
    ['x', 'y', 'z']
    >>> bool(jnp.allclose(u.ustrip("km", xyz["x"]), 2.0, atol=1e-6))
    True

    """
    return EmbeddedManifold(
        intrinsic=S2,
        ambient=R3,
        embed_map=TwoSphereIn3D(radius=radius, ambient=ambient),
    )
