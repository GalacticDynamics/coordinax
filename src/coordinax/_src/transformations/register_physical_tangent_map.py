"""Conversion functions for vector representations."""

__all__: tuple[str, ...] = ()

import jax.tree as jtu
import plum

import quaxed.numpy as jnp

from .utils import pack_uniform_unit, unpack_with_unit
from coordinax._src import api, charts, roles
from coordinax._src.custom_types import CsDict, OptUSys
from coordinax._src.embed import EmbeddedManifold

# ===================================================================
# Support for the higher-level `vconvert` function


@plum.dispatch
def vconvert(
    role: roles.PhysDisp | roles.PhysVel | roles.PhysAcc,
    to_chart: charts.AbstractChart,  # type: ignore[type-arg]
    from_chart: charts.AbstractChart,  # type: ignore[type-arg]
    p_dif: CsDict,
    p_pos: CsDict,
    /,
    *_: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    """Convert a differential vector from one chart to another.

    For example, the following roles transform via physical_tangent_transform
    (physical vectors in orthonormal frame):

    - Pos: physical vector with uniform length units
    - Vel: physical vector with uniform velocity units (length/time)
    - PhysAcc: physical vector with uniform acceleration units (length/time²)

    All components must have homogeneous physical dimension. For example, in
    cylindrical coordinates, a Pos has components (rho[m], phi[m], z[m]) where
    phi is the physical length along the tangential direction, NOT an angular
    increment.

    """
    # Convert using the tangent-space transformation
    return api.physical_tangent_transform(
        to_chart, from_chart, p_dif, at=p_pos, usys=usys
    )


# ===================================================================


@plum.dispatch
def physical_tangent_transform(
    to_chart: charts.AbstractChart,  # type: ignore[type-arg]
    from_chart: charts.AbstractChart,  # type: ignore[type-arg]
    v_phys: CsDict,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    """Transform **physical** differential components by orthonormal-frame change.

    This treats `v_phys` as components of a physical vector (velocity or acceleration)
    expressed in the orthonormal frame associated with `from_chart`, evaluated at the
    position `at` (given in `from_chart` coordinates).

    The same geometric vector is re-expressed in the orthonormal frame associated with
    `to_chart` at the same physical point, using the target metric.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt

    >>> point = {"r": u.Q(5, "m"), "theta": u.Q(45, "deg"), "phi": u.Q(30, "deg")}
    >>> v_sph = {"r": u.Q(1, "m/s"), "theta": u.Q(0.1, "m/s"),
    ...          "phi": u.Q(0.2, "m/s")}
    >>> v_cart = cxt.physical_tangent_transform(
    ...     cxc.cart3d, cxc.sph3d, v_sph, at=point)
    >>> v_cart
    {'x': Quantity(Array(0.57360968, dtype=float64), unit='m / s'),
     'y': Quantity(Array(0.56211381, dtype=float64), unit='m / s'),
     'z': Quantity(Array(0.6363961, dtype=float64), unit='m / s')}

    We can also transform raw arrays without units, by providing a unit system:

    >>> point = {"r": 5, "theta": 45, "phi": 30}
    >>> v_sph = {"r": 1, "theta": 0.1, "phi": 0.2}  # m/s
    >>> usys = u.unitsystem("m", "deg", "s")
    >>> v_cart = cxt.physical_tangent_transform(
    ...     cxc.cart3d, cxc.sph3d, v_sph, at=point, usys=usys)
    >>> v_cart
    {'x': Array(0.57360968, dtype=float64), 'y': Array(0.56211381, dtype=float64),
     'z': Array(0.6363961, dtype=float64)}

    Notes
    -----
    This is *not* the correct rule for transforming coordinate time-derivatives
    (e.g. dtheta/dt).

    """
    # Convert the position into the target rep so we can evaluate its frame at
    # the same point.
    p_pos_to = api.point_transform(to_chart, from_chart, at, usys=usys)

    # Orthonormal frames in Cartesian components at the same physical point.
    B_from = api.frame_cart(from_chart, at=at, usys=usys)
    B_to = api.frame_cart(to_chart, at=p_pos_to, usys=usys)

    # Pack vector components (uniform unit: speed or acceleration)
    # in rep component order.
    v_from, unit = pack_uniform_unit(v_phys, from_chart.components)

    # v_cart = B_from @ v_from
    v_cart = api.pushforward(B_from, v_from)

    # Pull back into the target frame using the target metric.
    v_to = api.pullback(api.metric_of(to_chart), B_to, v_cart)

    # Unpack to dict with shared unit
    out = unpack_with_unit(v_to, unit, to_chart.components)

    # Reshape outputs to broadcast shape of inputs
    shape = jnp.broadcast_shapes(*[jnp.shape(v) for v in v_phys.values()])
    return jtu.map(lambda x: jnp.reshape(x, shape), out)


@plum.dispatch
def physical_tangent_transform(
    to_chart: charts.AbstractChart,  # type: ignore[type-arg]
    from_embedded: EmbeddedManifold,  # type: ignore[type-arg]
    v_phys: CsDict,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    """Transform physical tangent components from an embedded manifold to ambient chart.

    This handles the embedding of intrinsic physical tangent components into the
    ambient space. The ``at`` position is given in intrinsic coordinates.

    Parameters
    ----------
    to_chart : charts.AbstractChart
        Target ambient chart (e.g., ``Cart3D``).
    from_embedded : EmbeddedManifold
        The embedded manifold defining the intrinsic chart and embedding.
    v_phys : CsDict
        Physical tangent components in the intrinsic orthonormal frame.
    at : CsDict
        Intrinsic coordinates where the tangent frame is evaluated.
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CsDict
        Physical tangent components in the target ambient chart's orthonormal frame.

    """
    # Get the orthonormal frame for the embedded manifold at the given point
    B_from = api.frame_cart(from_embedded, at=at, usys=usys)

    # Pack vector components (uniform unit: speed or acceleration)
    to_keys = tuple(to_chart.components)

    v_from, unit = pack_uniform_unit(v_phys, from_embedded.components)

    # Push forward to Cartesian: v_cart = B_from @ v_from
    v_cart = api.pushforward(B_from, v_from)

    # Pull back into the target frame using the target metric
    # First embed the point to get position in ambient chart
    p_ambient = api.embed_point(from_embedded, at, usys=usys)
    # Then transform to target chart coordinates
    p_to = api.point_transform(
        to_chart, from_embedded.ambient_chart, p_ambient, usys=usys
    )
    # Get target frame
    B_to = api.frame_cart(to_chart, at=p_to, usys=usys)
    # Pull back
    v_to = api.pullback(api.metric_of(to_chart), B_to, v_cart)
    out = unpack_with_unit(v_to, unit, to_keys)

    # Reshape outputs to broadcast shape of inputs
    shape = jnp.broadcast_shapes(*[v.shape for v in v_phys.values()])
    return jtu.map(lambda x: jnp.reshape(x, shape), out)


@plum.dispatch
def physical_tangent_transform(
    to_chart: charts.Cart1D | charts.Cart2D | charts.Cart3D,
    from_embedded: EmbeddedManifold,  # type: ignore[type-arg]
    v_phys: CsDict,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    """Transform physical tangent components from an embedded manifold to ambient chart.

    This handles the embedding of intrinsic physical tangent components into the
    ambient space. The ``at`` position is given in intrinsic coordinates.

    Parameters
    ----------
    to_chart : charts.AbstractChart
        Target ambient chart (e.g., ``Cart3D``).
    from_embedded : EmbeddedManifold
        The embedded manifold defining the intrinsic chart and embedding.
    v_phys : CsDict
        Physical tangent components in the intrinsic orthonormal frame.
    at : CsDict
        Intrinsic coordinates where the tangent frame is evaluated.
    usys
        Unit system for the transformation. This is sometimes required for
        transformations that depend on physical constants (e.g., speed of light
        or ``Delta`` in {class}`~coordinax.charts.ProlateSpheroidal3D`) but `p`
        is raw values without units.

    Returns
    -------
    CsDict
        Physical tangent components in the target ambient chart's orthonormal frame.

    """
    # Get the orthonormal frame for the embedded manifold at the given point
    B_from = api.frame_cart(from_embedded, at=at, usys=usys)

    # Pack vector components (uniform unit: speed or acceleration)
    from_keys = tuple(from_embedded.components)
    to_keys = tuple(to_chart.components)

    v_from, unit = pack_uniform_unit(v_phys, from_keys)

    # Push forward to Cartesian: v_cart = B_from @ v_from
    v_cart = api.pushforward(B_from, v_from)

    # If target is Cartesian, we're done (just unpack)
    out = unpack_with_unit(v_cart, unit, to_keys)

    # Reshape outputs to broadcast shape of inputs
    shape = jnp.broadcast_shapes(*[v.shape for v in v_phys.values()])
    return jtu.map(lambda x: jnp.reshape(x, shape), out)
