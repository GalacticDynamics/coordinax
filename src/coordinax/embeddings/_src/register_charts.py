"""Point-roled transformations."""

__all__: tuple[str, ...] = ()


import plum

import coordinax.api.embeddings as cxeapi
import coordinax.charts as cxc
from .conv_chart import EmbeddedChart
from coordinax.internal.custom_types import CDict, OptUSys


@plum.dispatch
def point_realization_map(
    to_chart: EmbeddedChart,  # type: ignore[type-arg]
    from_chart: EmbeddedChart,  # type: ignore[type-arg]
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Convert between embedded manifolds with a shared ambient space.

    This function transforms intrinsic coordinates from one embedded manifold
    to another by:
    1. Embedding the point into the ambient space of the source manifold
    2. Transforming in the ambient space (if the ambient charts differ)
    3. Projecting back to the intrinsic coordinates of the target manifold

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    **Example 1: Two spheres with different radii**

    Both spheres use the same intrinsic SphericalTwoSphere chart but have different
    radii:

    >>> emb1 = cxe.TwoSphereIn3D(radius=u.Q(1.0, "km"), ambient=cxc.cart3d)
    >>> sphere1 = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb1)
    >>> emb2 = cxe.TwoSphereIn3D(radius=u.Q(2.0, "km"), ambient=cxc.cart3d)
    >>> sphere2 = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb2)

    A point on sphere1 (theta=pi/4, phi=0):

    >>> p = {"theta": u.Q(45, "deg"), "phi": u.Q(0, "deg")}
    >>> p2 = cxc.point_realization_map(sphere2, sphere1, p)
    >>> {k: v.uconvert("deg") for k, v in p2.items()}
    {'theta': Quantity(Array(45., dtype=float64), unit='deg'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='deg')}

    The angular coordinates are preserved (both spheres share the same
    angular parameterization via projection through the shared ambient space).

    """
    if type(to_chart.ambient) is not type(from_chart.ambient):
        msg = "EmbeddedChart ambient kinds must match for conversion."
        raise ValueError(msg)

    p_ambient = cxeapi.embed_point(from_chart, p)  # TODO: support usys
    p_ambient = cxc.point_realization_map(
        to_chart.ambient, from_chart.ambient, p_ambient, usys=usys
    )
    return cxeapi.project_point(to_chart, p_ambient)  # TODO: support usys


@plum.dispatch
def point_realization_map(
    to_chart: EmbeddedChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Project an ambient position into an embedded chart.

    This transforms coordinates from an ambient chart (e.g., Cartesian or
    Spherical) into the intrinsic coordinates of an embedded manifold.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    **From Cartesian ambient to SphericalTwoSphere intrinsic:**

    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(1.0, "m"), ambient=cxc.cart3d)
    >>> sphere = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb)

    A point on the unit sphere in Cartesian coords (on equator, x-axis):

    >>> p_cart = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxc.point_realization_map(sphere, cxc.cart3d, p_cart)
    {'theta': Quantity(Array(1.57079633, dtype=float64), unit='rad'),
     'phi': Quantity(Array(0., dtype=float64, ...), unit='rad')}

    **From Spherical ambient to SphericalTwoSphere intrinsic:**

    The ambient spherical coords (r, theta, phi) project to intrinsic
    (theta, phi), discarding the radial component:

    >>> p_sph = {"r": 5, "theta": 1, "phi": 0.5}  # No units
    >>> usys = u.unitsystem("m", "rad")
    >>> cxc.point_realization_map(sphere, cxc.sph3d, p_sph, usys=usys)
    {'theta': Array(1., dtype=float64), 'phi': Array(0.5, dtype=float64, ...)}

    """
    p_ambient = cxc.point_realization_map(to_chart.ambient, from_chart, p, usys=usys)
    return cxeapi.project_point(to_chart, p_ambient)


@plum.dispatch
def point_realization_map(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: EmbeddedChart,  # type: ignore[type-arg]
    p: CDict,
    /,
    usys: OptUSys = None,
) -> CDict:
    """Embed intrinsic coordinates into an ambient representation.

    This transforms intrinsic coordinates of an embedded manifold into
    coordinates of an ambient chart, which may differ from the embedding's
    native ambient chart.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.embeddings as cxe
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    **From SphericalTwoSphere intrinsic to Cartesian ambient:**

    >>> emb = cxe.TwoSphereIn3D(radius=u.Q(1.0, "m"), ambient=cxc.cart3d)
    >>> sphere = cxe.EmbeddedChart(intrinsic=cxc.sph2, embedding=emb)

    A point on the unit sphere (on equator, x-axis):

    >>> p_cart = {"theta": u.Q(1.0, "rad"), "phi": u.Q(0.0, "rad")}
    >>> cxc.point_realization_map(cxc.cart3d, sphere, p_cart)
    {'x': Quantity(Array(0.84147098, dtype=float64, ...), unit='m'),
     'y': Quantity(Array(0., dtype=float64, ...), unit='m'),
     'z': Quantity(Array(0.54030231, dtype=float64, ...), unit='m')}

    **From SphericalTwoSphere intrinsic to Spherical ambient:**

    >>> p_sph = {"theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
    >>> cxc.point_realization_map(cxc.sph3d, sphere, p_sph)
    {'r': Quantity(Array(1., dtype=float64, weak_type=True), unit='m'),
     'theta': Quantity(Array(1., dtype=float64), unit='rad'),
     'phi': Quantity(Array(0.5, dtype=float64, weak_type=True), unit='rad')}

    """
    p_ambient = cxeapi.embed_point(from_chart, p)
    return cxc.point_realization_map(to_chart, from_chart.ambient, p_ambient, usys=usys)
