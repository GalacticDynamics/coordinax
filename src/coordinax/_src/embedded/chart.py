"""Representations for embedding manifolds."""

__all__ = ("EmbeddedChart",)


from typing import ClassVar, Generic, cast, final
from typing_extensions import override

import jax
import plum

import coordinax.api.charts as cxcapi
import coordinax.api.manifolds as cxmapi
from .embedmap import AbstractEmbeddingMap, AmbientT, IntrinsicT
from .manifold import EmbeddedManifold
from coordinax._src.base import (
    AbstractChart,
    AbstractManifold,
    chart_dataclass_decorator,
)
from coordinax._src.custom_types import CDict, Ds, Ks, OptUSys


@jax.tree_util.register_static
@final
@chart_dataclass_decorator
class EmbeddedChart(AbstractChart[Ks, Ds], Generic[IntrinsicT, AmbientT, Ks, Ds]):
    r"""Chart for intrinsic coordinates on an embedding manifold.

    This is a convenience wrapper that combines an intrinsic chart with an
    embedding to an ambient Cartesian chart. It provides the same component and
    dimension information as the intrinsic chart, but also provides a
    realization map to Cartesian coordinates via the embedding.

    The more correct way to represent an embedding manifold is with
    {class}`~coordinax.manifolds.EmbeddedManifold`.

    Examples
    --------
    Embed/project `{class}`~coordinax.charts.SphericalTwoSphere` through an
    ambient `{class}`~coordinax.charts.Spherical3D` chart::

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

    """

    embed_map: AbstractEmbeddingMap[IntrinsicT, AmbientT]
    """The embedding that defines the map to the ambient chart.

    This is the core data of the EmbeddedChart, as it defines the ambient chart
    and the embedding parameters (e.g., radius for a sphere). The intrinsic
    chart is determined by the embedding's ``intrinsic`` property, and the
    ambient chart is determined by the embedding's ``ambient`` property.

    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
    >>> chart.embed_map
    TwoSphereIn3D(radius=Q(2., 'km'))

    """

    M: ClassVar[AbstractManifold]

    @override
    @property
    def M(self) -> EmbeddedManifold:
        """The manifold associated with this chart.

        This is an EmbeddedManifold that combines the intrinsic and ambient
        manifolds defined by the embedding map.

        >>> import coordinax.manifolds as cxm
        >>> import unxt as u
        >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
        >>> chart.M
        EmbeddedManifold(intrinsic=HyperSphericalManifold(ndim=2),
                         ambient=Rn(3),
                         embed_map=TwoSphereIn3D(radius=Q(2., 'km')))

        """
        return EmbeddedManifold(
            intrinsic=self.intrinsic.M, ambient=self.ambient.M, embed_map=self.embed_map
        )

    @property
    def intrinsic(self) -> IntrinsicT:
        """The intrinsic chart.

        >>> import coordinax.manifolds as cxm
        >>> import unxt as u
        >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
        >>> chart.intrinsic
        SphericalTwoSphere(M=Sn(2))

        """
        return self.embed_map.intrinsic

    @property
    def ambient(self) -> AmbientT:
        """The ambient chart.

        >>> import coordinax.manifolds as cxm
        >>> import unxt as u
        >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
        >>> chart.ambient
        Spherical3D(M=Rn(3))

        """
        return self.embed_map.ambient

    # ===================================================
    # Chart API

    @property
    def components(self) -> Ks:
        """Return the components of the intrinsic chart.

        >>> import coordinax.manifolds as cxm
        >>> import unxt as u
        >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
        >>> chart.components
        ('theta', 'phi')

        """
        return self.intrinsic.components

    @property
    def coord_dimensions(self) -> Ds:
        """Return the coordinate dimensions of the intrinsic chart.

        >>> import coordinax.manifolds as cxm
        >>> import unxt as u
        >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
        >>> chart.coord_dimensions
        ('angle', 'angle')

        """
        return self.intrinsic.coord_dimensions

    @property
    def cartesian(self) -> AbstractChart:
        """The ambient Cartesian chart for the embedding.

        >>> import coordinax.manifolds as cxm
        >>> import unxt as u
        >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
        >>> chart.cartesian
        Cart3D(M=Rn(3))

        """
        return self.ambient.cartesian

    def __hash__(self) -> int:
        """Hash based on the class and the embedding map.

        >>> import coordinax.manifolds as cxm
        >>> import unxt as u
        >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.StaticQuantity(2.0, "km")))
        >>> isinstance(hash(chart), int)
        True

        """  # noqa: E501
        return hash((self.__class__, self.embed_map))


# ===================================================================


@plum.dispatch
def pt_embed(
    p_intrinsic: CDict, embedding: EmbeddedChart, /, *, usys: OptUSys = None
) -> CDict:
    r"""Embed intrinsic point coordinates into ambient coordinates.

    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
    >>> p_intrinsic = {"theta": u.Q(45, "deg"), "phi": u.Q(30, "deg")}
    >>> cxm.pt_embed(p_intrinsic, chart)
    {'r': Q(2., 'km'), 'theta': Q(45, 'deg'), 'phi': Q(30, 'deg')}

    """
    # Redispatch to the more general pt_embed that handles chart
    # transitions in both the ambient and intrinsic charts.
    out = cxmapi.pt_embed(
        p_intrinsic,
        embedding.intrinsic,
        embedding.ambient,
        embedding.embed_map,
        usys=usys,
    )
    return cast("CDict", out)


# ===================================================================


@plum.dispatch
def pt_project(
    p_ambient: CDict, embedding: EmbeddedChart, /, *, usys: OptUSys = None
) -> CDict:
    r"""Project ambient coordinates onto intrinsic chart coordinates.

    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
    >>> p_amb = {"r": u.Q(2.0, "km"), "theta": u.Q(45, "deg"), "phi": u.Q(30, "deg")}
    >>> cxm.pt_project(p_amb, chart)
    {'theta': Q(45, 'deg'), 'phi': Q(30, 'deg')}

    """
    # Redispatch to the more general pt_project that handles chart
    # transitions in both the ambient and intrinsic charts.
    out = cxmapi.pt_project(
        p_ambient,
        embedding.ambient,
        embedding.intrinsic,
        embedding.embed_map,
        usys=usys,
    )
    return cast("CDict", out)


@plum.dispatch
def pt_project(
    p_ambient: CDict,
    from_chart: AbstractChart,
    embedding: EmbeddedChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""Project ambient coordinates onto intrinsic chart coordinates.

    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
    >>> p_amb = {"r": u.Q(2.0, "km"), "theta": u.Q(45, "deg"), "phi": u.Q(30, "deg")}
    >>> cxm.pt_project(p_amb, cxc.sph3d, chart)
    {'theta': Q(45, 'deg'), 'phi': Q(30, 'deg')}

    """
    p_ambient: CDict = cxcapi.pt_map(  # ty: ignore[invalid-assignment]
        p_ambient, from_chart, embedding.ambient, usys=usys
    )
    # Redispatch to the more general pt_project that handles chart
    # transitions in both the ambient and intrinsic charts.
    out = cxmapi.pt_project(
        p_ambient,
        embedding.ambient,
        embedding.intrinsic,
        embedding.embed_map,
        usys=usys,
    )
    return cast("CDict", out)


# ===================================================================


@plum.dispatch
def pt_map(
    p: CDict,
    from_chart: EmbeddedChart,
    to_chart: EmbeddedChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Convert between embedded manifolds with a shared ambient space.

    This function transforms intrinsic coordinates from one embedded manifold
    to another by:
    1. Embedding the point into the ambient space of the source manifold
    2. Transforming in the ambient space (if the ambient charts differ)
    3. Projecting back to the intrinsic coordinates of the target manifold

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    **Example 1: Two spheres with different radii**

    Both spheres use the same intrinsic SphericalTwoSphere chart but have different
    radii:

    >>> sphere1 = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(1.0, "km")))
    >>> sphere2 = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))

    A point on sphere1 (theta=pi/4, phi=0):

    >>> p = {"theta": u.Q(45, "deg"), "phi": u.Q(0, "deg")}
    >>> p2 = cxc.pt_map(p, sphere1, sphere2)
    >>> {k: v.uconvert("deg") for k, v in p2.items()}
    {'theta': Q(45, 'deg'), 'phi': Q(0, 'deg')}

    The angular coordinates are preserved (both spheres share the same
    angular parameterization via projection through the shared ambient space).

    """
    if type(to_chart.ambient) is not type(from_chart.ambient):
        msg = "EmbeddedChart ambient kinds must match for conversion."
        raise ValueError(msg)

    p_ambient = cxmapi.pt_embed(p, from_chart, usys=usys)
    p_ambient = cxcapi.pt_map(
        p_ambient,
        from_chart.ambient.M,
        from_chart.ambient,
        to_chart.ambient.M,
        to_chart.ambient,
        usys=usys,
    )
    out = cxmapi.pt_project(p_ambient, to_chart, usys=usys)
    return cast("CDict", out)


@plum.dispatch
def pt_map(
    p: CDict,
    from_chart: AbstractChart,
    to_chart: EmbeddedChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Project an ambient position into an embedded chart.

    This transforms coordinates from an ambient chart (e.g., Cartesian or
    Spherical) into the intrinsic coordinates of an embedded manifold.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    **From Cartesian ambient to SphericalTwoSphere intrinsic:**

    >>> sphere = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(1.0, "m")))

    A point on the unit sphere in Cartesian coords (on equator, x-axis):

    >>> p_cart = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxc.pt_map(p_cart, cxc.cart3d, sphere)
    {'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    **From Spherical ambient to SphericalTwoSphere intrinsic:**

    The ambient spherical coords (r, theta, phi) project to intrinsic
    (theta, phi), discarding the radial component:

    >>> p_sph = {"r": 5, "theta": 1, "phi": 0.5}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.pt_map(p_sph, cxc.sph3d, sphere, usys=usys)
    {'theta': 1, 'phi': 0.5}

    """
    p_ambient = cxcapi.pt_map(
        p, from_chart.M, from_chart, to_chart.ambient.M, to_chart.ambient, usys=usys
    )
    out = cxmapi.pt_project(p_ambient, to_chart, usys=usys)
    return cast("CDict", out)


@plum.dispatch
def pt_map(
    p: CDict,
    from_chart: EmbeddedChart,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Embed intrinsic coordinates into an ambient representation.

    This transforms intrinsic coordinates of an embedded manifold into
    coordinates of an ambient chart, which may differ from the embedding's
    native ambient chart.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    From SphericalTwoSphere intrinsic to Cartesian ambient:

    >>> sphere = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(1.0, "m")))

    A point on the unit sphere (on equator, x-axis):

    >>> p_sph = {"theta": u.Q(1.0, "rad"), "phi": u.Q(0.0, "rad")}
    >>> cxc.pt_map(p_sph, sphere, cxc.cart3d)
    {'x': Q(0.84147098, 'm'), 'y': Q(0., 'm'), 'z': Q(0.54030231, 'm')}

    From SphericalTwoSphere intrinsic to Spherical ambient:

    >>> p_sph = {"theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
    >>> cxc.pt_map(p_sph, sphere, cxc.sph3d)
    {'r': Q(1., 'm'), 'theta': Q(1., 'rad'), 'phi': Q(0.5, 'rad')}

    """
    p_ambient = cxmapi.pt_embed(p, from_chart, usys=usys)
    out = cxcapi.pt_map(
        p_ambient,
        from_chart.ambient.M,
        from_chart.ambient,
        to_chart.M,
        to_chart,
        usys=usys,
    )
    return cast("CDict", out)
