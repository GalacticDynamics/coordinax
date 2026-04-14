"""``import coordinax.main as cx``."""

__all__ = (  # distances
    "Distance",
    # angles
    "Angle",
    # charts
    "NoGlobalCartesianChartError",
    "CartesianProductChart",
    "cartesian_chart",
    "guess_chart",
    "cdict",
    "pt_map",
    "jacobian_pt_map",
    "realize_cartesian",
    "cart1d",
    "radial1d",
    "time1d",
    "cart2d",
    "polar2d",
    "cart3d",
    "cyl3d",
    "sph3d",
    "lonlat_sph3d",
    "loncoslat_sph3d",
    "math_sph3d",
    "cartnd",
    # representations
    "cconvert",
    "add",
    "subtract",
    "PointGeometry",
    "point_geom",
    "NoBasis",
    "no_basis",
    "Location",
    "loc",
    "Representation",
    "point",
    # vectors
    "Point",
    "ToUnitsOptions",
)

from coordinax.angles import Angle
from coordinax.charts import (
    CartesianProductChart,
    NoGlobalCartesianChartError,
    cart1d,
    cart2d,
    cart3d,
    cartesian_chart,
    cartnd,
    cdict,
    cyl3d,
    guess_chart,
    jacobian_pt_map,
    loncoslat_sph3d,
    lonlat_sph3d,
    math_sph3d,
    polar2d,
    pt_map,
    radial1d,
    realize_cartesian,
    sph3d,
    time1d,
)
from coordinax.distances import Distance
from coordinax.representations import (
    Location,
    NoBasis,
    PointGeometry,
    Representation,
    add,
    cconvert,
    loc,
    no_basis,
    point,
    point_geom,
    subtract,
)
from coordinax.vectors import Point, ToUnitsOptions

try:  # noqa: SIM105
    import coordinax.interop.astropy as _  # noqa: F401  # ty: ignore[unresolved-import]
except ImportError:
    pass
