"""Vector Conversion."""

__all__ = ("add", "cmap", "subtract")

from collections.abc import Callable, Mapping
from typing import Any, no_type_check

import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt.quantity as uq

import coordinax.api.representations as cxrapi
import coordinax.charts as cxc
from .custom_types import CDict, OptUSys
from .geom import PointGeometry, TangentGeometry
from .rep import Representation

# =======================================================================
# CMap


@plum.dispatch
def cmap(*fixed_args: Any, **fixed_kw: Any) -> Any:
    """Return a partial function for vector conversion.

    Examples
    --------
    Convert a point from Cartesian coordinates to spherical coordinates:

    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Define a map to convert a point from Cartesian coordinates to spherical coordinates:

    >>> map = cxr.cmap(cxc.cart3d, cxr.point, cxc.sph3d)

    Apply the map to a point:

    >>> q = {"x": 1, "y": 2, "z": 3}
    >>> map(q)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    >>> q = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> map(q)
    {'r': Q(3.74165739, 'm'), 'theta': Q(0.64052231, 'rad'),
     'phi': Q(1.10714872, 'rad')}

    Apply to a Point:

    >>> import coordinax.vectors as cxv
    >>> vec = cxv.Point.from_(q, cxc.cart3d)
    >>> map(vec)
    Point(
      {'r': Q(3.74165739, 'm'), 'theta': Q(0.64052231, 'rad'),
       'phi': Q(1.10714872, 'rad')},
      chart=Spherical3D(), manifold=EuclideanManifold(ndim=3)
    )

    """
    return lambda x, *args, **kwargs: cxrapi.cconvert(
        x, *fixed_args, *args, **fixed_kw, **kwargs
    )


# =======================================================================
# Partial dispatches


# Need to set precedence=1 to avoid ambiguity with the (Any, to_chart,
# from_chat) -> pt_map dispatch.
@plum.dispatch(precedence=1)  # ty: ignore[no-matching-overload]
def cconvert(obj: None, /, *fixed_args: Any, **fixed_kw: Any) -> Any:
    r"""Return a partial function for vector conversion.

    Examples
    --------
    Convert a point from Cartesian coordinates to spherical coordinates:

    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc

    Define a point in Cartesian coordinates:

    >>> q = {"x": 1.0, "y": 2.0, "z": 3.0}

    Convert it to spherical coordinates:

    >>> map = cxr.cconvert(None, cxc.cart3d, cxr.point, cxc.sph3d)
    >>> map(q)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    """
    del obj  # unused
    return cmap(*fixed_args, **fixed_kw)


# =======================================================================
# General dispatches


@plum.dispatch
def cconvert(
    x: Any,
    # from-*
    from_chart: cxc.AbstractChart,
    from_rep: Representation,
    # to-*
    to_chart: cxc.AbstractChart,
    to_rep: Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> Any:
    r"""Convert point data between charts.

    Examples
    --------
    Convert a point from Cartesian coordinates to spherical coordinates:

    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc

    Define a point in Cartesian coordinates:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}

    Convert it to spherical coordinates:

    >>> q = cxr.cconvert(p, cxc.cart3d, cxr.point, cxc.sph3d, cxr.point)
    >>> q
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output `q` represents the same geometric point but expressed in the
    target chart.

    The representation remains unchanged; only the chart changes:

    >>> cxr.cconvert(q, cxc.sph3d, cxr.point, cxc.cart3d, cxr.point)
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    Let's work through more examples.

    **Cartesian to Spherical (with units):**

    >>> import unxt as u
    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxr.cconvert(p, cxc.cart3d, cxr.point, cxc.sph3d, cxr.point)
    {'r': Q(1., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    **Cylindrical to Cartesian (without units):**

    >>> p = {"rho": 3.0, "phi": 0, "z": 4.0}
    >>> cxr.cconvert(p, cxc.cyl3d, cxr.point, cxc.cart3d, cxr.point)
    {'x': Array(3., dtype=float64, ...), 'y': Array(0., dtype=float64, ...),
     'z': 4.0}

    **Polar to Cartesian (2D):**

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cxr.cconvert(p, cxc.polar2d, cxr.point, cxc.cart2d, cxr.point)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    **Between Spherical variants (Spherical to LonLatSpherical):**

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(45, "deg"), "phi": u.Q(0, "deg")}
    >>> cxr.cconvert(p, cxc.sph3d, cxr.point, cxc.lonlat_sph3d, cxr.point)
    {'lon': Q(0, 'deg'), 'lat': Q(45., 'deg'), 'distance': Q(1., 'm')}

    **Identity conversion (same chart):**

    >>> p = {"x": u.Q(2.0, "m"), "y": u.Q(3.0, "m")}
    >>> cxr.cconvert(p, cxc.cart2d, cxr.point, cxc.cart2d, cxr.point) is p
    True

    """
    # redispatch on the combination of GeometryKind
    return cxrapi.cconvert(
        x,
        # from-*
        from_chart,
        from_rep.geom_kind,
        from_rep,
        # to-*
        to_chart,
        to_rep.geom_kind,
        to_rep,
        # extra
        **({"at": at} if at is not None else {}),
        usys=usys,
    )


@plum.dispatch
def cconvert(
    x: Any,
    # from-*
    from_chart: cxc.AbstractChart,
    from_rep: Representation,
    # to-*
    to_chart: cxc.AbstractChart,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> Any:
    r"""Convert point data between charts.

    Examples
    --------
    Convert a point from Cartesian coordinates to spherical coordinates:

    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc

    Define a point in Cartesian coordinates:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}

    Convert it to spherical coordinates:

    >>> q = cxr.cconvert(p, cxc.cart3d, cxr.point, cxc.sph3d)
    >>> q
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output `q` represents the same geometric point but expressed in the
    target chart.

    The representation remains unchanged; only the chart changes:

    >>> cxr.cconvert(q, cxc.sph3d, cxr.point, cxc.cart3d)
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    """
    # redispatch on the combination of GeometryKind
    return cxrapi.cconvert(
        x,
        # from-*
        from_chart,
        from_rep.geom_kind,
        from_rep,
        # to-*
        to_chart,
        from_rep.geom_kind,
        from_rep,
        # extra
        **({"at": at} if at is not None else {}),
        usys=usys,
    )


# =======================================================================
# Geometry-specific dispatches


@plum.dispatch
def cconvert(
    x: Any,
    # from-*
    from_chart: cxc.AbstractChart,
    from_geom: PointGeometry,
    from_rep: Representation,
    # to-*
    to_chart: cxc.AbstractChart,
    to_geom: PointGeometry,
    to_rep: Representation,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    r"""Convert point data between charts.

    This function delegates to `coordinax.charts.pt_map`.
    The representation arguments are checked to ensure they correspond to
    canonical point data:

    $$(\mathrm{PointGeometry},\, \mathrm{NoBasis},\, \mathrm{Location}).$$

    Examples
    --------
    Convert a point from Cartesian coordinates to spherical coordinates:

    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc

    Define a point in Cartesian coordinates:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}

    Convert it to spherical coordinates:

    >>> cxr.cconvert(p, cxc.cart3d, cxr.point_geom, cxr.point,
    ...                 cxc.sph3d, cxr.point_geom, cxr.point)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    """
    return cxc.pt_map(
        x, from_chart, from_geom, from_rep, to_chart, to_geom, to_rep, usys=usys
    )


@plum.dispatch
def cconvert(
    x: Any,
    # from-*
    from_chart: cxc.AbstractChart,
    from_geom: TangentGeometry,
    from_rep: Representation,
    # to-*
    to_chart: cxc.AbstractChart,
    to_geom: TangentGeometry,
    to_rep: Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> Any:
    r"""Convert tangent data between charts via Jacobian pushforward.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = {"r": jnp.array(5.0), "theta": jnp.array(1.0), "phi": jnp.array(2.0)}
    >>> at = {"r": jnp.array(3.0), "theta": jnp.array(0.5), "phi": jnp.array(0.0)}
    >>> cxr.cconvert(v, cxc.sph3d, cxr.tangent_geom, cxr.coord_disp,
    ...              cxc.sph3d, cxr.tangent_geom, cxr.phys_disp, at=at)
    {'r': Array(5., dtype=float64, ...),
     'theta': Array(3., dtype=float64, ...),
     'phi': Array(..., dtype=float64, ...)}

    >>> v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> cxr.cconvert(v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, cxr.coord_disp, at=at)
    {'r': Array(1., ...), 'theta': Array(0., ...)}

    """
    del from_geom, to_geom
    return cxrapi.tangent_map(
        x, from_chart, from_rep, to_chart, to_rep, at=at, usys=usys
    )


# =======================================================================
# Leaf binary operation helper


@no_type_check
def _leaf_binop(
    op: Callable, a: Mapping[str, Any], b: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply a binary op to matching leaves in two component dicts."""
    return jtu.map(op, a, b, is_leaf=uq.is_any_quantity)


# =======================================================================
# Add


def _binop_via_cartesian(
    op: Callable,
    lhs: Any,
    lhs_chart: cxc.AbstractChart,
    rhs: Any,
    rhs_chart: cxc.AbstractChart,
) -> tuple[CDict, cxc.AbstractChart]:
    """Run *op* on two CDicts, round-tripping through Cartesian if needed.

    Returns ``(result_data, result_chart)``.
    """
    target_chart = lhs_chart

    try:
        ambient_cart = target_chart.cartesian
    except cxc.NoGlobalCartesianChartError:
        ambient_cart = None

    if ambient_cart is None or ambient_cart == target_chart:
        rhs_data = (
            rhs
            if rhs_chart == target_chart
            else cxc.pt_map(rhs, rhs_chart, target_chart)
        )
        return _leaf_binop(op, lhs, rhs_data), target_chart

    # Curvilinear: round-trip through Cartesian.
    lhs_cart = cxc.pt_map(lhs, lhs_chart, ambient_cart)
    rhs_cart = cxc.pt_map(rhs, rhs_chart, ambient_cart)
    result_cart = _leaf_binop(op, lhs_cart, rhs_cart)
    out: CDict = cxc.pt_map(result_cart, ambient_cart, target_chart)  # ty: ignore[invalid-assignment]
    return out, target_chart


@plum.dispatch
def add(
    lhs: Any,
    lhs_chart: cxc.AbstractChart,
    lhs_rep: Representation,
    rhs: Any,
    rhs_chart: cxc.AbstractChart,
    rhs_rep: Representation,
    /,
) -> Any:
    """Add two coordinate data objects via Cartesian round-trip.

    Both operands are converted to the ambient Cartesian chart of
    ``lhs_chart``, added component-wise, then converted back to
    ``lhs_chart``.  If ``lhs_chart`` is already Cartesian (or has no
    global Cartesian), ``rhs`` is converted into ``lhs_chart`` and added
    directly.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p1 = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> p2 = {"x": u.Q(4, "m"), "y": u.Q(5, "m"), "z": u.Q(6, "m")}
    >>> cxr.add(p1, cxc.cart3d, cxr.point, p2, cxc.cart3d, cxr.point)
    {'x': Q(5, 'm'), 'y': Q(7, 'm'), 'z': Q(9, 'm')}

    """
    del lhs_rep, rhs_rep  # unused
    result_data, _ = _binop_via_cartesian(jnp.add, lhs, lhs_chart, rhs, rhs_chart)
    return result_data


# =======================================================================
# Subtract


@plum.dispatch
def subtract(
    lhs: Any,
    lhs_chart: cxc.AbstractChart,
    lhs_rep: Representation,
    rhs: Any,
    rhs_chart: cxc.AbstractChart,
    rhs_rep: Representation,
    /,
) -> Any:
    """Subtract two coordinate data objects via Cartesian round-trip.

    Both operands are converted to the ambient Cartesian chart of
    ``lhs_chart``, subtracted component-wise, then converted back to
    ``lhs_chart``.  If ``lhs_chart`` is already Cartesian (or has no
    global Cartesian), ``rhs`` is converted into ``lhs_chart`` and
    subtracted directly.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> p1 = {"x": u.Q(4, "m"), "y": u.Q(5, "m"), "z": u.Q(6, "m")}
    >>> p2 = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.subtract(p1, cxc.cart3d, cxr.point, p2, cxc.cart3d, cxr.point)
    {'x': Q(3, 'm'), 'y': Q(3, 'm'), 'z': Q(3, 'm')}

    """
    del lhs_rep, rhs_rep  # unused
    result_data, _ = _binop_via_cartesian(jnp.subtract, lhs, lhs_chart, rhs, rhs_chart)
    return result_data
