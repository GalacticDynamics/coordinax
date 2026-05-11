"""Vector Conversion."""

__all__: tuple[str, ...] = ()

from typing import Any, Final

import plum

import coordinax.charts as cxc
from .custom_types import OptUSys
from .geom import PointGeometry
from .rep import Representation, point

# =======================================================================
# General dispatches


@plum.dispatch
def pt_map(
    x: Any,
    # from-*
    from_chart: cxc.AbstractChart,
    from_rep: Representation,
    # to-*
    to_chart: cxc.AbstractChart,
    to_rep: Representation,
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

    >>> q = cxc.pt_map(p, cxc.cart3d, cxr.point, cxc.sph3d, cxr.point)
    >>> q
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output `q` represents the same geometric point but expressed in the
    target chart.

    The representation remains unchanged; only the chart changes:

    >>> cxc.pt_map(q, cxc.sph3d, cxr.point, cxc.cart3d, cxr.point)
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    Let's work through more examples.

    **Cartesian to Spherical (with units):**

    >>> import unxt as u
    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxc.pt_map(p, cxc.cart3d, cxr.point, cxc.sph3d, cxr.point)
    {'r': Q(1., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    **Cylindrical to Cartesian (without units):**

    >>> p = {"rho": 3.0, "phi": 0, "z": 4.0}
    >>> cxc.pt_map(p, cxc.cyl3d, cxr.point, cxc.cart3d, cxr.point)
    {'x': Array(3., dtype=float64, ...), 'y': Array(0., dtype=float64, ...),
     'z': 4.0}

    **Polar to Cartesian (2D):**

    >>> p = {"r": u.Q(5.0, "m"), "theta": u.Q(90, "deg")}
    >>> cxc.pt_map(p, cxc.polar2d, cxr.point, cxc.cart2d, cxr.point)
    {'x': Q(3.061617e-16, 'm'), 'y': Q(5., 'm')}

    **Between Spherical variants (Spherical to LonLatSpherical):**

    >>> p = {"r": u.Q(1.0, "m"), "theta": u.Q(45, "deg"), "phi": u.Q(0, "deg")}
    >>> cxc.pt_map(p, cxc.sph3d, cxr.point, cxc.lonlat_sph3d, cxr.point)
    {'lon': Q(0, 'deg'), 'lat': Q(45., 'deg'), 'distance': Q(1., 'm')}

    **Identity conversion (same chart):**

    >>> p = {"x": u.Q(2.0, "m"), "y": u.Q(3.0, "m")}
    >>> cxc.pt_map(p, cxc.cart2d, cxr.point, cxc.cart2d, cxr.point) is p
    True

    """
    # redispatch on the combination of GeometryKind
    return cxc.pt_map(
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
def pt_map(
    x: Any,
    from_chart: cxc.AbstractChart,
    from_rep: Representation,
    to_chart: cxc.AbstractChart,
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

    >>> q = cxc.pt_map(p, cxc.cart3d, cxr.point, cxc.sph3d)
    >>> q
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output `q` represents the same geometric point but expressed in the
    target chart.

    The representation remains unchanged; only the chart changes:

    >>> cxc.pt_map(q, cxc.sph3d, cxr.point, cxc.cart3d)
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    """
    # redispatch on the combination of GeometryKind
    return cxc.pt_map(
        x,
        # from-*
        from_chart,
        from_rep.geom_kind,
        from_rep,
        # to-*
        to_chart,
        from_rep.geom_kind,
        from_rep,
        # data
        usys=usys,
    )


# =======================================================================
# Point dispatches

_TO_PTM_MSG: Final = (
    "PointGeometry cconvert requires `to_rep` to be the canonical point "
    "representation `point = Representation(point_geom, no_basis, loc)`."
)
_FROM_PTM_MSG: Final = (
    "PointGeometry cconvert requires `from_rep` to be the canonical point "
    "representation `point = Representation(point_geom, no_basis, loc)`."
)


@plum.dispatch
def pt_map(
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

    >>> cxc.pt_map(p, cxc.cart3d, cxr.point_geom, cxr.point,
    ...                             cxc.sph3d, cxr.point_geom, cxr.point)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    """
    del to_geom, from_geom  # only used for dispatch

    # Check that the representations are compatible with the geometry kind.
    if to_rep != point:
        raise ValueError(_TO_PTM_MSG)
    if from_rep != point:
        raise ValueError(_FROM_PTM_MSG)

    return cxc.pt_map(x, from_chart, to_chart, usys=usys)
