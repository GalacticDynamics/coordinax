"""Vector."""

__all__ = ("guess_rep",)

import plum

import unxt as u

import coordinax.api.representations as cxrapi
from .constants import ANGLE, LENGTH
from .rep import Representation, point
from coordinax.internal.custom_types import CDict

# Mapping from dimension to role flag
DIM_TO_REP_MAP: dict[u.AbstractDimension, Representation] = {
    LENGTH: point,  # Length → point (affine location, per spec)
    ANGLE: point,  # Angles → affine (mixed with other coords)
}


@plum.dispatch
def guess_rep(obj: u.AbstractDimension, /) -> Representation:
    """Infer representation from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> r1 = cxr.guess_rep(u.dimension("length"))
    >>> r1
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    try:
        rep = DIM_TO_REP_MAP[obj]
    except KeyError as e:
        msg = f"Cannot infer rep from dimension {obj}"
        raise ValueError(msg) from e
    return rep


@plum.dispatch
def guess_rep(x: u.AbstractQuantity, /) -> Representation:
    """Infer representation from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> r1 = cxr.guess_rep(u.Q(1.0, "m"))
    >>> r1
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    dim = u.dimension_of(x)
    return cxrapi.guess_rep(dim)


@plum.dispatch
def guess_rep(obj: CDict, /) -> Representation:
    """Infer representation from the physical dimensions of a component dictionary.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> d1 = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> r1 = cxr.guess_rep(d1)
    >>> r1
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    dims = {u.dimension_of(v) for v in obj.values()}
    if len(dims) != 1:
        msg = f"Cannot infer representation from mixed dimensions: {dims}"
        raise ValueError(msg)
    dim = dims.pop()
    return cxrapi.guess_rep(dim)
