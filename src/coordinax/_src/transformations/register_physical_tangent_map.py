"""Conversion functions for vector representations."""

__all__: tuple[str, ...] = ()

from collections.abc import Callable
from typing import Any

import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u

from .utils import pack_uniform_unit, unpack_with_unit
from coordinax._src import api, charts as cxc, roles as cxr
from coordinax._src.api import physical_tangent_transform
from coordinax._src.custom_types import CsDict, OptUSys
from coordinax._src.embed import EmbeddedManifold
from coordinax_api import vconvert

# ===================================================================
# Support for the higher-level `vconvert` function


@plum.dispatch
def vconvert(
    role: cxr.AbstractPhysRole,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    p_tan: CsDict,
    p_pnt: CsDict,
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
        to_chart, from_chart, p_tan, at=p_pnt, usys=usys
    )


#####################################################################
# Physical tangent transformations

# ===================================================================
# Partial application


@plum.dispatch
def physical_tangent_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    /,
) -> Callable[..., Any]:
    """Return a partial function for physical tangent transformation.

    When called with only ``to_chart`` and ``from_chart``, returns a callable
    that can be used to transform physical tangent vectors (velocities,
    accelerations, etc.) between the orthonormal frames of the two charts.

    This is useful for creating reusable transformation functions or for
    use in higher-order functions like ``jax.vmap``.

    Returns
    -------
    Callable
        A function with signature ``(v_phys, *, at, usys=None) -> CsDict``
        that transforms physical tangent components.

    Examples
    --------
    >>> import jax
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxt
    >>> import unxt as u

    Create a reusable transformation function:

    >>> sph_to_cart = cxt.physical_tangent_transform(cxc.cart3d, cxc.sph3d)

    Use it to transform a velocity vector:

    >>> point = {"r": u.Q(5, "m"), "theta": u.Q(90, "deg"), "phi": u.Q(0, "deg")}
    >>> v_sph = {"r": u.Q(1, "m/s"), "theta": u.Q(0, "m/s"), "phi": u.Q(0, "m/s")}
    >>> v_cart = sph_to_cart(v_sph, at=point)
    >>> jax.tree.map(lambda x: x.round(2), v_cart)
    {'x': Quantity(Array(1., dtype=float64), unit='m / s'),
     'y': Quantity(Array(0., dtype=float64), unit='m / s'),
     'z': Quantity(Array(0., dtype=float64), unit='m / s')}

    The same function can be reused for different points and vectors:

    >>> v_sph2 = {"r": u.Q(0, "m/s"), "theta": u.Q(1, "m/s"), "phi": u.Q(0, "m/s")}
    >>> v_cart2 = sph_to_cart(v_sph2, at=point)
    >>> jax.tree.map(lambda x: x.round(2), v_cart2)
    {'x': Quantity(Array(0., dtype=float64), unit='m / s'),
     'y': Quantity(Array(0., dtype=float64), unit='m / s'),
     'z': Quantity(Array(-1., dtype=float64), unit='m / s')}

    """
    # NOT: lambda is much faster than ft.partial here
    return lambda *args, **kw: api.physical_tangent_transform(
        to_chart, from_chart, *args, **kw
    )


# ===================================================================


@plum.dispatch
def physical_tangent_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
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
    p_pnt_to = api.point_transform(to_chart, from_chart, at, usys=usys)

    # Orthonormal frames in Cartesian components at the same physical point.
    B_from = api.frame_cart(from_chart, at=at, usys=usys)
    B_to = api.frame_cart(to_chart, at=p_pnt_to, usys=usys)

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
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
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
    to_chart: cxc.Cart1D | cxc.Cart2D | cxc.Cart3D,
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
    to_chart : cxc.AbstractChart
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
        or ``Delta`` in {class}`~coordinax.cxc.ProlateSpheroidal3D`) but `p`
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


# ===================================================================
# Tangent Transform a Quantity


@plum.dispatch
def physical_tangent_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    q: u.AbstractQuantity,
    /,
    *,
    at: u.AbstractQuantity | CsDict,
    usys: OptUSys = None,
) -> u.AbstractQuantity:
    """Transform a physical tangent Quantity between charts.

    This is a convenience wrapper around `physical_tangent_transform` for
    transforming a single Quantity representing a physical tangent vector.

    """
    # Pack input Quantity into component dict
    q_dict = api.cdict(q, from_chart)
    at_dict = api.cdict(at, from_chart) if isinstance(at, u.AbstractQuantity) else at

    # Transform components
    qto_dict = api.physical_tangent_transform(
        to_chart, from_chart, q_dict, at=at_dict, usys=usys
    )

    # Reassemble output Quantity
    qto: u.AbstractQuantity = jnp.stack(
        [qto_dict[comp] for comp in to_chart.components], axis=-1
    )
    return qto


# =================================================================
# Tangent Transform a an Array


@plum.dispatch
def physical_tangent_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    v: jnp.ndarray,
    /,
    *,
    at: jnp.ndarray | CsDict,
    usys: OptUSys = None,
) -> jnp.ndarray:
    """Transform a physical tangent vector array between charts.

    This is a convenience wrapper around `physical_tangent_transform` for
    transforming a single array representing a physical tangent vector.

    The input array `v` should have shape (..., N) where N is the number of
    components in `from_chart`. The output array will have shape (..., M)
    where M is the number of components in `to_chart`.

    """
    raise NotImplementedError("TODO")
