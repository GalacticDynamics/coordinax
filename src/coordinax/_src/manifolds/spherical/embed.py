"""Two-sphere manifold."""

__all__ = ("TwoSphereIn3D", "embedded_twosphere")

from dataclasses import dataclass, field

from typing import Any, final

import unxt as u

import coordinax.charts as cxc
from .manifold import HyperSphericalManifold
from coordinax._src.manifolds.custom_types import CDict, OptUSys
from coordinax._src.manifolds.embedded import (
    AbstractEmbeddingMap,
    AmbientT,
    EmbeddedManifold,
    IntrinsicT,
)
from coordinax._src.manifolds.euclidean import EuclideanManifold


@final
@dataclass(frozen=True, slots=True)
class TwoSphereIn3D(AbstractEmbeddingMap[IntrinsicT, AmbientT]):
    r"""Embedding of ``cxc.SphericalTwoSphere`` as a 2-sphere in a 3D ambient chart.

    This embedding models a 2-sphere of fixed radius $R$ as the hypersurface $r
    = R$ in 3D spherical coordinates $(r, \theta, \phi)$. The intrinsic chart is
    therefore expected to have components $(\theta, \phi)$.

    The key design choice is that **all** coordinate-level embedding and
    projection operations are defined via an intermediate 3D spherical chart
    ({class}`~coordinax.charts.Spherical3D`), regardless of which ambient chart
    is selected. In particular:

    - If ``ambient`` is `{class}`~coordinax.charts.Spherical3D`, then
      {meth}`embed` returns spherical coordinates ``(r, theta, phi)`` and
      {meth}`project` expects the same.
    - If ``ambient`` is `{class}`~coordinax.charts.Cart3D`, then {meth}`embed`
      performs ``SphericalTwoSphere -> Spherical3D -> Cart3D`` and returns Cartesian
      coordinates ``(x, y, z)``; {meth}`project` performs ``Cart3D ->
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

    Embed/project through an ambient `{class}`~coordinax.charts.Cart3D` chart
    (routing via `{class}`~coordinax.charts.Spherical3D` internally)::

    >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
    >>> xyz = cxm.pt_embed(p, chart)
    >>> p3 = cxm.pt_project(xyz, chart)
    >>> p3
    {'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    >>> bool(jnp.allclose(u.ustrip("rad", p3["phi"]), u.ustrip("rad", p["phi"])))
    True

    """

    radius: u.AbstractQuantity | float | int = field()

    @property
    def intrinsic(self) -> cxc.AbstractChart[Any, Any]:
        """The intrinsic chart is always `coordinax.charts.SphericalTwoSphere`."""
        return cxc.sph2

    @property
    def ambient(self) -> cxc.AbstractChart[Any, Any]:
        """The ambient chart is always `coordinax.charts.Spherical3D`."""
        return cxc.sph3d

    def embed(self, q: CDict, /, *, usys: OptUSys = None) -> CDict:
        """Embed ``SphericalTwoSphere`` intrinsic coords into ``Spherical3D`` coords."""
        del usys
        return {"r": self.radius, "theta": q["theta"], "phi": q["phi"]}

    def project(self, x_sph: CDict, /, *, usys: OptUSys = None) -> CDict:
        """Project ``Spherical3D`` onto ``SphericalTwoSphere`` intrinsic coords."""
        del usys
        return {"theta": x_sph["theta"], "phi": x_sph["phi"]}


def embedded_twosphere(
    radius: float | u.AbstractQuantity,
    ambient: cxc.AbstractChart[Any, Any] = cxc.sph3d,
) -> EmbeddedManifold:
    """Create an {class}`coordinax.manifolds.EmbeddedManifold` for the two-sphere.

    This is a convenience helper that constructs an
    {class}`coordinax.manifolds.EmbeddedManifold` with
    ``intrinsic=HyperSphericalManifold()`` and ``embedding=TwoSphereIn3D(radius,
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
        {class}`coordinax.manifolds.TwoSphereIn3D` embedding.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    Default ambient (Spherical3D)::

    >>> manifold = cxm.embedded_twosphere(radius=u.Q(2.0, "km"))
    >>> manifold
    EmbeddedManifold(intrinsic=HyperSphericalManifold(...),
                     ambient=EuclideanManifold(ndim=3),
                     embed_map=TwoSphereIn3D(radius=Q(2., 'km')))

    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> sph = cxm.pt_embed(p, manifold)
    >>> sph
    {'r': Q(2., 'km'), 'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    With Cartesian ambient::

    >>> manifold = cxm.embedded_twosphere(
    ...     radius=u.Q(2.0, "km"), ambient=cxc.cart3d,
    ... )
    >>> xyz = cxm.pt_embed(p, manifold)
    >>> p3 = cxm.pt_project(xyz, manifold)
    >>> p3
    {'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    """
    return EmbeddedManifold(
        intrinsic=HyperSphericalManifold(2),
        ambient=EuclideanManifold(3),
        embed_map=TwoSphereIn3D(radius=radius),
    )
