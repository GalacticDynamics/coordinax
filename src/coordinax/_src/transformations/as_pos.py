"""Role transformation: as_pos (to Pos role)."""

import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u

from coordinax._src import api, charts as cxc, embed as cxe
from coordinax._src.custom_types import CsDict, OptUSys

# ============================================================================


@plum.dispatch
def as_pos(
    point: CsDict,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    origin: None,
    /,
    *,
    usys: OptUSys = None,
) -> tuple[CsDict, cxc.AbstractChart]:  # type: ignore[type-arg]
    r"""Convert a point coordinate dictionary to a displacement.

    Using the chart's coordinate origin.

    Mathematical Meaning:

    Interprets ``point`` as coordinates of $p \in \mathbb{R}^n$ and returns
    the displacement vector

    $$ d = p - 0. $$

    This operation is only defined for charts with a Euclidean metric.

    Parameters
    ----------
    point
        Coordinate dictionary representing a point.
    chart
        A Euclidean chart.
    origin
        Must be ``None``.
    usys
        Optional unit system for the output displacement.

    Raises
    ------
    NotImplementedError
        If the chart does not have a Euclidean metric.

    Examples
    --------
    >>> import coordinax as cx
    >>> cx.as_pos({"x": 1, "y": 2, "z": 3}, cx.charts.cart3d, None)
    ({'x': 1, 'y': 2, 'z': 3}, Cart3D())

    """
    # Check point, chart, and origin are compatible
    chart.check_data(point)

    # TODO: for other metrics
    if not chart.is_euclidean:
        msg = "as_pos is currently only implemented for charts with Euclidean metric."
        raise NotImplementedError(msg)

    # Convert to Cartesian chart
    cart_chart = chart.cartesian
    point_cart = api.point_transform(cart_chart, chart, point, usys=usys)

    # Compute displacement in Cartesian. For origin = 0 that's just the point
    # itself.
    disp_data = point_cart

    return disp_data, cart_chart


@plum.dispatch
def as_pos(
    point: CsDict,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    origin: CsDict,
    /,
    *,
    usys: OptUSys = None,
) -> tuple[CsDict, cxc.AbstractChart]:  # type: ignore[type-arg]
    r"""Convert a point coordinate dictionary to a displacement.

    Relative to an explicit origin.

    Mathematical Meaning:
    Computes $$ d = p - o $$

    in a shared Cartesian chart.

    Parameters
    ----------
    point
        Coordinates of the point $p$.
    chart
        A Euclidean chart.
    origin
        Coordinates of the origin $o$.
    usys
        Optional unit system for the output displacement.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.as_pos({"x": 2}, cx.charts.cart1d, {"x": 0.5})
    ({'x': Array(1.5, dtype=float64, weak_type=True)}, Cart1D())

    """
    # Check point, chart, and origin are compatible
    chart.check_data(point)
    chart.check_data(origin)

    # TODO: for other metrics
    if not chart.is_euclidean:
        msg = "as_pos is currently only implemented for charts with Euclidean metric."
        raise NotImplementedError(msg)

    # Convert both to shared Cartesian chart, subtract
    cart_chart = chart.cartesian

    # Point Transform
    point_cart = api.point_transform(cart_chart, chart, point, usys=usys)
    origin_cart = api.point_transform(cart_chart, chart, origin, usys=usys)

    # Compute displacement in Cartesian
    disp_data = jtu.map(
        jnp.subtract, point_cart, origin_cart, is_leaf=u.quantity.is_any_quantity
    )

    return disp_data, cart_chart


@plum.dispatch
def as_pos(
    point: CsDict,
    chart: cxe.EmbeddedManifold,  # type: ignore[type-arg]
    origin: None,
    /,
    *,
    usys: OptUSys = None,
) -> tuple[CsDict, cxc.AbstractChart]:  # type: ignore[type-arg]
    """Convert a Point coordinate to a Pos-valued displacement on an embedded manifold.

    Raises
    ------
    ValueError
        For embedded manifolds, there is no canonical coordinate-origin point on
        the manifold. Therefore `origin` must be provided explicitly.

    """
    msg = "Embedded manifolds require an explicit origin for as_pos."
    raise ValueError(msg)


@plum.dispatch
def as_pos(
    point: CsDict,
    chart: cxe.EmbeddedManifold,  # type: ignore[type-arg]
    origin: CsDict,
    /,
    *,
    usys: OptUSys = None,
) -> tuple[CsDict, cxc.AbstractChart]:  # type: ignore[type-arg]
    r"""Convert point on an embedded manifold to an ambient Cartesian displacement.

    Mathematical Meaning:
    Let $\iota : M \hookrightarrow \mathbb{R}^N$ be the embedding. This
    computes

    $$ d = \iota(p) - \iota(o) \in \mathbb{R}^N. $$

    The result lives in the ambient Cartesian chart.

    Parameters
    ----------
    point
        Coordinates of $p \in M$.
    chart
        An embedded manifold.
    origin
        Coordinates of $o \in M$.
    usys
        Optional unit system for the output displacement.

    Returns
    -------
    (CsDict, AbstractChart)
        Ambient displacement and the ambient Cartesian chart.

    Examples
    --------
    Example: TwoSphere embedded in 3D Cartesian space.

    In this example, ``point`` and ``origin`` are given in a chart on the
    manifold (e.g. longitude/latitude). ``as_pos`` embeds both into the ambient
    chart (typically ``Cart3D``) and subtracts there.

    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> import wadler_lindig as wl

    >>> embed_twosphere = cxc.EmbeddedManifold(cxc.twosphere, cxc.cart3d,
    ...                                        params={"R": u.Q(2.0, "km")})

    Define two points on the sphere:

    >>> p = {"theta": u.Q(10, "deg"), "phi": u.Q(30, "deg")}
    >>> o = {"theta": u.Q(0, "deg"),  "phi": u.Q(0, "deg")}

    Convert to an ambient displacement:

    >>> d, ambient = cx.as_pos(p, embed_twosphere, o)
    >>> wl.pprint(d, short_arrays='compact', use_short_names=True, named_units=False)
    { 'x': Quantity(0.30076747, unit='km'),
      'y': Quantity(0.17364818, unit='km'),
      'z': Quantity(-0.03038449, unit='km') }

    The returned displacement dictionary uses the *ambient chart* components.

    """
    chart.check_data(point)
    chart.check_data(origin)

    # Embed both points to the ambient chart
    point_ambient = api.embed_point(chart, point, usys=usys)
    origin_ambient = api.embed_point(chart, origin, usys=usys)

    # Use ambient chart as basis for as_pos
    disp_data, disp_chart = api.as_pos(
        point_ambient, chart.ambient_chart, origin_ambient, usys=usys
    )

    return disp_data, disp_chart
