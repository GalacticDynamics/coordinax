"""Guesses for the Representation and its components."""

__all__: tuple[str, ...] = ()

from typing import Any, cast

import plum

import unxt as u

import coordinax.api.representations as cxrapi
import coordinax.charts as cxc
from .basis import AbstractBasis, coord_basis, no_basis
from .constants import (
    ACCELERATION,
    ANGLE,
    ANGULAR_ACCELERATION,
    ANGULAR_SPEED,
    AREA,
    LENGTH,
    SPEED,
)
from .custom_types import CDict
from .geom import (
    AbstractGeometry,
    PointGeometry,
    TangentGeometry,
    point_geom,
    tangent_geom,
)
from .rep import Representation, point
from .semantics import AbstractSemanticKind, Location, acc, dpl, loc, vel

# ===================================================================


@plum.dispatch
def guess_geometry_kind(obj: AbstractGeometry, /) -> AbstractGeometry:
    """Infer geometry kind from an AbstractGeometry object.

    Examples
    --------
    >>> import coordinax.representations as cxr

    >>> geom = cxr.PointGeometry()
    >>> cxr.guess_geometry_kind(geom) is geom
    True

    """
    return obj


# Mapping from dimension to geometry flag
DIM_TO_GEOM_MAP: dict[
    u.AbstractDimension | tuple[u.AbstractDimension, ...], AbstractGeometry
] = {
    LENGTH: point_geom,  # Length → point (affine location, per spec)
    ANGLE: point_geom,  # Angles → affine (mixed with other coords)
    (ANGLE, LENGTH): point_geom,
    SPEED: tangent_geom,
    ANGULAR_SPEED: tangent_geom,
    (ANGULAR_SPEED, SPEED): tangent_geom,
    (SPEED, ANGULAR_SPEED): tangent_geom,
    ACCELERATION: tangent_geom,
    ANGULAR_ACCELERATION: tangent_geom,
    (ACCELERATION, ANGULAR_ACCELERATION): tangent_geom,
    (ANGULAR_ACCELERATION, ACCELERATION): tangent_geom,
}


@plum.dispatch
def guess_geometry_kind(
    dim: u.AbstractDimension | tuple[u.AbstractDimension, ...], /
) -> AbstractGeometry:
    """Infer geometry kind from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> cxr.guess_geometry_kind(u.dimension("length"))
    PointGeometry()

    >>> cxr.guess_geometry_kind((u.dimension("angle"), u.dimension("length")))
    PointGeometry()

    >>> cxr.guess_geometry_kind(u.dimension("speed"))
    TangentGeometry()

    >>> cxr.guess_geometry_kind(u.dimension("angular speed"))
    TangentGeometry()

    >>> cxr.guess_geometry_kind(u.dimension("acceleration"))
    TangentGeometry()

    >>> cxr.guess_geometry_kind(u.dimension("angular acceleration"))
    TangentGeometry()

    """
    try:
        geom = DIM_TO_GEOM_MAP[dim]
    except KeyError as e:
        msg = f"Cannot infer geometry kind from dimension {dim}"
        raise ValueError(msg) from e
    return geom


@plum.dispatch
def guess_geometry_kind(obj: u.AbstractQuantity, /) -> AbstractGeometry:
    """Infer geometry kind from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> cxr.guess_geometry_kind(u.Q(1.0, "m"))
    PointGeometry()

    >>> cxr.guess_geometry_kind(u.Q(2.0, "m / s"))
    TangentGeometry()

    >>> cxr.guess_geometry_kind(u.Q(3.0, "m / s ** 2"))
    TangentGeometry()

    """
    dim = u.dimension_of(obj)
    out = cxrapi.guess_geometry_kind(dim)
    return cast("AbstractGeometry", out)


dim_name = lambda dim: dim._physical_type[0]


@plum.dispatch
def guess_geometry_kind(obj: CDict, /) -> AbstractGeometry:
    """Infer geometry kind from the physical dimensions of a component dictionary.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> d1 = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> cxr.guess_geometry_kind(d1)
    PointGeometry()

    >>> d2 = {"lon": u.Q(1.0, "deg"), "lat": u.Q(2.0, "deg")}
    >>> cxr.guess_geometry_kind(d2)
    PointGeometry()

    >>> d3 = {"x": u.Q(1.0, "m / s"), "y": u.Q(2.0, "m / s")}
    >>> cxr.guess_geometry_kind(d3)
    TangentGeometry()

    """
    # Find the `dim` for determining the geometry kind
    dims = {u.dimension_of(v) for v in obj.values()}
    if len(dims) == 0:
        msg = "Cannot infer geometry kind without dimensions"
        raise ValueError(msg)

    # Get down to a single dimension or tuple of dimensions for the lookup
    dim = dims.pop() if len(dims) == 1 else tuple(sorted(dims, key=dim_name))

    # Now we can infer geometry kind from the dimension.
    out = cxrapi.guess_geometry_kind(dim)
    return cast("AbstractGeometry", out)


@plum.dispatch
def guess_geometry_kind(obj: CDict, chart: cxc.AbstractChart, /) -> AbstractGeometry:
    """Infer geometry kind from the physical dimensions of a component dictionary and chart.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc

    >>> d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> cxr.guess_geometry_kind(d, cxc.cart2d)
    PointGeometry()

    """  # noqa: E501
    # Confirm data is compatible with the chart (e.g., no "x" component for
    # spherical coords) and has the right dimensions
    chart.check_data(obj, keys=True, values=False)
    out = cxrapi.guess_geometry_kind(obj)
    return cast("AbstractGeometry", out)


@plum.dispatch
def guess_geometry_kind(
    obj: CDict, chart: cxc.ProlateSpheroidal3D, /
) -> AbstractGeometry:
    """Infer geometry kind from the physical dimensions of a component dictionary and chart.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc

    >>> d = {"mu": u.Q(1, "km2"), "nu": u.Q(0.5, "km2"), "phi": u.Q(1, "deg")}
    >>> chart = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(1.0, "km"))
    >>> cxr.guess_geometry_kind(d, chart)
    PointGeometry()

    """  # noqa: E501
    # Confirm data is compatible with the chart (e.g., no "x" component for
    # spherical coords) and has the right dimensions
    chart.check_data(obj, keys=True, values=False)
    # If the above doesn't raise, we can infer geometry kind from the chart and
    # dimensions.
    dims = {u.dimension_of(v) for v in obj.values()}
    if dims == {AREA, ANGLE}:
        # Special case for prolate spheroidal coords, which have mixed length
        # and angle dimensions but are still an affine geometry.
        return point_geom

    raise ValueError(
        f"Cannot infer geometry kind from dimensions {dims} and chart {chart}"
    )


# ===================================================================


@plum.dispatch
def guess_basis_kind(obj: AbstractBasis, /) -> AbstractBasis:
    """Infer basis kind from an AbstractBasis object.

    Examples
    --------
    >>> import coordinax.representations as cxr

    >>> basis = cxr.NoBasis()
    >>> cxr.guess_basis_kind(basis) is basis
    True

    """
    return obj


# Mapping from dimension to basis kind
DIM_TO_BASIS_MAP: dict[
    u.AbstractDimension | tuple[u.AbstractDimension, ...], AbstractBasis
] = {
    LENGTH: no_basis,
    ANGLE: no_basis,
    (ANGLE, LENGTH): no_basis,
}


@plum.dispatch
def guess_basis_kind(
    dim: u.AbstractDimension | tuple[u.AbstractDimension, ...], /
) -> AbstractBasis:
    """Infer basis kind from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> cxr.guess_basis_kind(u.dimension("length"))
    NoBasis()

    >>> cxr.guess_basis_kind((u.dimension("angle"), u.dimension("length")))
    NoBasis()

    """
    try:
        basis = DIM_TO_BASIS_MAP[dim]
    except KeyError as e:
        msg = f"Cannot infer basis kind from dimension {dim}"
        raise ValueError(msg) from e
    return basis


@plum.dispatch
def guess_basis_kind(obj: u.AbstractQuantity, /) -> AbstractBasis:
    """Infer basis kind from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> cxr.guess_basis_kind(u.Q(1.0, "m"))
    NoBasis()

    """
    dim = u.dimension_of(obj)
    out = cxrapi.guess_basis_kind(dim)
    return cast("AbstractBasis", out)


@plum.dispatch
def guess_basis_kind(obj: CDict, /) -> AbstractBasis:
    """Infer basis kind from the physical dimensions of a component dictionary.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> d1 = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> cxr.guess_basis_kind(d1)
    NoBasis()

    >>> d2 = {"lon": u.Q(1.0, "deg"), "lat": u.Q(2.0, "deg")}
    >>> cxr.guess_basis_kind(d2)
    NoBasis()

    """
    dims = {u.dimension_of(v) for v in obj.values()}
    if len(dims) == 0:
        msg = "Cannot infer basis kind without dimensions"
        raise ValueError(msg)

    # Get down to a single dimension or tuple of dimensions for the lookup
    dim = dims.pop() if len(dims) == 1 else tuple(sorted(dims, key=dim_name))

    out = cxrapi.guess_basis_kind(dim)
    return cast("AbstractBasis", out)


# ===================================================================


@plum.dispatch
def guess_semantic_kind(obj: AbstractSemanticKind, /) -> AbstractSemanticKind:
    """Infer semantic kind from an AbstractSemanticKind object.

    Examples
    --------
    >>> import coordinax.representations as cxr

    >>> sem = cxr.Location()
    >>> cxr.guess_semantic_kind(sem) is sem
    True

    """
    return obj


DIM_TO_SEMANTICS_MAP: dict[
    u.AbstractDimension | tuple[u.AbstractDimension, ...], AbstractSemanticKind
] = {
    LENGTH: loc,
    ANGLE: loc,
    (ANGLE, LENGTH): loc,
    SPEED: vel,
    ANGULAR_SPEED: vel,
    (ANGULAR_SPEED, SPEED): vel,
    (SPEED, ANGULAR_SPEED): vel,
    ACCELERATION: acc,
    ANGULAR_ACCELERATION: acc,
    (ACCELERATION, ANGULAR_ACCELERATION): acc,
    (ANGULAR_ACCELERATION, ACCELERATION): acc,
}


@plum.dispatch
def guess_semantic_kind(
    dim: u.AbstractDimension | tuple[u.AbstractDimension, ...], /
) -> AbstractSemanticKind:
    """Infer semantic kind from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> cxr.guess_semantic_kind(u.dimension("length"))
    Location()

    >>> cxr.guess_semantic_kind((u.dimension("angle"), u.dimension("length")))
    Location()

    >>> cxr.guess_semantic_kind(u.dimension("speed"))
    Velocity()

    >>> cxr.guess_semantic_kind(u.dimension("angular speed"))
    Velocity()

    >>> cxr.guess_semantic_kind(u.dimension("acceleration"))
    Acceleration()

    >>> cxr.guess_semantic_kind(u.dimension("angular acceleration"))
    Acceleration()

    """
    try:
        sem = DIM_TO_SEMANTICS_MAP[dim]
    except KeyError as e:
        msg = f"Cannot infer semantic kind from dimension {dim}"
        raise ValueError(msg) from e
    return sem


@plum.dispatch
def guess_semantic_kind(obj: u.AbstractQuantity, /) -> AbstractSemanticKind:
    """Infer semantic kind from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> cxr.guess_semantic_kind(u.Q(1.0, "m"))
    Location()

    """
    dim = u.dimension_of(obj)
    out = cxrapi.guess_semantic_kind(dim)
    return cast("AbstractSemanticKind", out)


@plum.dispatch
def guess_semantic_kind(obj: CDict, /) -> AbstractSemanticKind:
    """Infer semantic kind from the physical dimensions of a component dictionary.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> d1 = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> cxr.guess_semantic_kind(d1)
    Location()

    >>> d2 = {"lon": u.Q(1.0, "deg"), "lat": u.Q(2.0, "deg")}
    >>> cxr.guess_semantic_kind(d2)
    Location()

    """
    # Find the `dim` for determining the semantic kind
    dims = {u.dimension_of(v) for v in obj.values()}
    if len(dims) == 0:
        msg = "Cannot infer semantic kind without dimensions"
        raise ValueError(msg)

    # Get down to a single dimension or tuple of dimensions for the lookup
    dim = dims.pop() if len(dims) == 1 else tuple(sorted(dims, key=dim_name))

    # Now we can infer semantic kind from the dimension.
    out = cxrapi.guess_semantic_kind(dim)
    return cast("AbstractSemanticKind", out)


# ===================================================================


@plum.dispatch
def guess_rep(obj: Representation, /) -> Representation:
    """Infer representation from a Representation object.

    Examples
    --------
    >>> import coordinax.representations as cxr

    >>> rep = cxr.Representation(cxr.PointGeometry(), cxr.NoBasis(), cxr.Location())
    >>> cxr.guess_rep(rep) is rep
    True

    """
    return obj


@plum.dispatch
def guess_rep(obj: PointGeometry, /) -> Representation:
    """Infer representation from a PointGeometry object.

    Examples
    --------
    >>> import coordinax.representations as cxr

    >>> geom = cxr.PointGeometry()
    >>> cxr.guess_rep(geom)
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    return point  # ty: ignore[invalid-return-type]


@plum.dispatch
def guess_rep(
    obj: u.AbstractDimension
    | tuple[u.AbstractDimension, ...]
    | u.AbstractQuantity
    | CDict,
    geom: PointGeometry,
    /,
) -> Representation:
    """Infer point representation from data and an already-inferred PointGeometry.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> rep = cxr.guess_rep(u.dimension("length"), cxr.point_geom)
    >>> rep
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    return point  # ty: ignore[invalid-return-type]


@plum.dispatch
def guess_rep(
    obj: u.AbstractDimension
    | tuple[u.AbstractDimension, ...]
    | u.AbstractQuantity
    | CDict,
    geom: TangentGeometry,
    /,
) -> Representation:
    """Infer tangent representation from data and an already-inferred TangentGeometry.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> rep = cxr.guess_rep(u.dimension("speed"), cxr.tangent_geom)
    >>> rep.geom_kind
    TangentGeometry()
    >>> rep.semantic_kind
    Velocity()

    >>> rep = cxr.guess_rep(u.dimension("acceleration"), cxr.tangent_geom)
    >>> rep.geom_kind
    TangentGeometry()
    >>> rep.semantic_kind
    Acceleration()

    """
    sem = cxrapi.guess_semantic_kind(obj)
    return Representation(geom_kind=geom, basis=coord_basis, semantic_kind=sem)  # ty: ignore[invalid-return-type]


@plum.dispatch
def guess_rep(
    obj: u.AbstractDimension
    | tuple[u.AbstractDimension, ...]
    | u.AbstractQuantity
    | CDict,
    /,
) -> Representation:
    """Infer representation from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> rep = cxr.guess_rep(u.dimension("length"))
    >>> rep
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    >>> rep = cxr.guess_rep(u.Q(1.0, "m"))
    >>> rep
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    >>> rep = cxr.guess_rep(u.dimension("speed"))
    >>> rep.geom_kind
    TangentGeometry()
    >>> rep.semantic_kind
    Velocity()

    >>> rep = cxr.guess_rep(u.Q(1.0, "m / s"))
    >>> rep.geom_kind
    TangentGeometry()
    >>> rep.semantic_kind
    Velocity()

    >>> rep = cxr.guess_rep(u.dimension("acceleration"))
    >>> rep.geom_kind
    TangentGeometry()
    >>> rep.semantic_kind
    Acceleration()

    """
    geom = cxrapi.guess_geometry_kind(obj)
    out = cxrapi.guess_rep(obj, geom)
    return cast("Representation", out)


@plum.dispatch
def guess_rep(obj: Any, chart: cxc.AbstractChart, /) -> Representation:
    """Infer representation from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> rep = cxr.guess_rep(u.dimension("length"))
    >>> rep
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    >>> rep = cxr.guess_rep(u.Q(1.0, "m"))
    >>> rep
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    geom = cxrapi.guess_geometry_kind(obj, chart)
    out = cxrapi.guess_rep(obj, chart, geom)
    return cast("Representation", out)


@plum.dispatch
def guess_rep(
    obj: Any, chart: cxc.AbstractChart, geom: PointGeometry, /
) -> Representation:
    """Infer representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> rep = cxr.guess_rep(u.Q(1.0, "m"), cxc.cart2d, point_geom)
    >>> rep
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    return point  # ty: ignore[invalid-return-type]


@plum.dispatch
def guess_rep(
    obj: Any, chart: cxc.AbstractChart, geom: TangentGeometry, /
) -> Representation:
    """Infer representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    Speed dimensions infer Velocity:

    >>> rep = cxr.guess_rep(u.Q([1.0, 2.0], "m / s"), cxc.cart2d, cxr.TangentGeometry())
    >>> rep
    Representation(geom_kind=TangentGeometry(), basis=CoordinateBasis(),
                   semantic_kind=Velocity())

    Length dimensions would infer Location in general, but TangentGeometry
    requires a tangent semantic kind, so Displacement is returned instead:

    >>> rep = cxr.guess_rep(u.Q([1.0, 2.0], "m"), cxc.cart2d, cxr.TangentGeometry())
    >>> rep
    Representation(geom_kind=TangentGeometry(), basis=CoordinateBasis(),
                   semantic_kind=Displacement())

    """
    # Infer the semantic kind from the data.
    data = cxc.cdict(obj, chart)
    semantic_kind = cxrapi.guess_semantic_kind(data)

    # Length/angle dimensions map to Location in the dimension→semantic table,
    # but TangentGeometry requires an AbstractTangentSemanticKind. Displacement
    # is the tangent kind at order 0 (same physical dimensions as a position,
    # but living in the tangent space), so use it as the default.
    if isinstance(semantic_kind, Location):
        semantic_kind = dpl

    # TODO: better determine the basis kind
    basis = coord_basis

    return Representation(geom_kind=geom, basis=basis, semantic_kind=semantic_kind)  # ty: ignore[invalid-return-type]
