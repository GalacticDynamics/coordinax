"""Vector Conversion."""

__all__ = ("cmap",)

from typing import Any

import plum

import coordinax.api.representations as api
import coordinax.charts as cxc
from .geom import PointGeometry
from .rep import Representation
from coordinax.internal.custom_types import OptUSys

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


    """
    return lambda x, *args, **kwargs: api.cconvert(
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
    return api.cconvert(
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
    return api.cconvert(
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
        usys=usys,
    )


# =======================================================================
# Point dispatches


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
