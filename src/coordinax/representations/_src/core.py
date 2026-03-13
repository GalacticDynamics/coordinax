"""Vector Conversion."""

__all__ = ("vconvert",)

from typing import Any

import plum

import coordinax.api.representations as api
import coordinax.charts as cxc
from .geom import PointGeometry
from .rep import Representation
from coordinax.internal.custom_types import OptUSys

# =======================================================================
# General dispatches


@plum.dispatch
def vconvert(
    to_chart: cxc.AbstractChart,
    to_rep: Representation,
    from_chart: cxc.AbstractChart,
    from_rep: Representation,
    x: Any,
    /,
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

    >>> q = cxr.vconvert(cxc.sph3d, cxr.point, cxc.cart3d, cxr.point, p)
    >>> q
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output `q` represents the same geometric point but expressed in the
    target chart.

    The representation remains unchanged; only the chart changes:

    >>> cxr.vconvert(cxc.cart3d, cxr.point, cxc.sph3d, cxr.point, q)
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    Let's work through more examples.

    **Cartesian to Spherical (with units):**

    >>> import unxt as u
    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxr.vconvert(cxc.sph3d, cxr.point, cxc.cart3d, cxr.point, p)
    {'r': Q(1., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    **Cylindrical to Cartesian (without units):**

    >>> p = {"rho": 3.0, "phi": 0, "z": 4.0}
    >>> cxr.vconvert(cxc.cart3d, cxr.point, cxc.cyl3d, cxr.point, p)
    {'x': Array(3., dtype=float64, ...), 'y': Array(0., dtype=float64, ...),
     'z': 4.0}

    **Polar to Cartesian (2D):**

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cxr.vconvert(cxc.cart2d, cxr.point, cxc.polar2d, cxr.point, p)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    **Between Spherical variants (Spherical to LonLatSpherical):**

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(45, "deg"), "phi": u.Q(0, "deg")}
    >>> cxr.vconvert(cxc.lonlat_sph3d, cxr.point, cxc.sph3d, cxr.point, p)
    {'lon': Q(0, 'deg'), 'lat': Q(45., 'deg'), 'distance': Q(1., 'm')}

    **Identity conversion (same chart):**

    >>> p = {"x": u.Q(2.0, "m"), "y": u.Q(3.0, "m")}
    >>> cxr.vconvert(cxc.cart2d, cxr.point, cxc.cart2d, cxr.point, p) is p
    True

    """
    # redispatch on the combination of GeometryKind
    return api.vconvert(
        # to-*
        to_chart,
        to_rep.geom_kind,
        to_rep,
        # from-*
        from_chart,
        from_rep.geom_kind,
        from_rep,
        # data
        x,
        usys=usys,
    )


@plum.dispatch
def vconvert(
    to_chart: cxc.AbstractChart,
    from_chart: cxc.AbstractChart,
    from_rep: Representation,
    x: Any,
    /,
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

    >>> q = cxr.vconvert(cxc.sph3d, cxc.cart3d, cxr.point, p)
    >>> q
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output `q` represents the same geometric point but expressed in the
    target chart.

    The representation remains unchanged; only the chart changes:

    >>> cxr.vconvert(cxc.cart3d, cxc.sph3d, cxr.point, q)
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    """
    # redispatch on the combination of GeometryKind
    return api.vconvert(
        # to-*
        to_chart,
        from_rep.geom_kind,
        from_rep,
        # from-*
        from_chart,
        from_rep.geom_kind,
        from_rep,
        # data
        x,
        usys=usys,
    )


# =======================================================================
# Point dispatches


@plum.dispatch
def vconvert(
    to_chart: cxc.AbstractChart,
    to_geom: PointGeometry,
    to_rep: Representation,
    from_chart: cxc.AbstractChart,
    from_geom: PointGeometry,
    from_rep: Representation,
    x: Any,
    /,
    usys: OptUSys = None,
) -> Any:
    r"""Convert point data between charts.

    This function delegates to `coordinax.charts.point_realization_map`.
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

    >>> cxr.vconvert(cxc.sph3d, cxr.point_geom, cxr.point,
    ...              cxc.cart3d, cxr.point_geom, cxr.point, p)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    """
    return cxc.point_realization_map(
        to_chart, to_geom, to_rep, from_chart, from_geom, from_rep, x, usys=usys
    )
