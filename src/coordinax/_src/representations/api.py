"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = ("cartesian_rep", "coord_map", "diff_map")

from typing import TYPE_CHECKING, Any

import plum

from .custom_types import PDict

if TYPE_CHECKING:
    import coordinax.r


@plum.dispatch.abstract
def cartesian_rep(obj: Any, /) -> "coordinax.r.AbstractRep":
    """Return the corresponding Cartesian vector type.

    Examples
    --------
    >>> import coordinax.r as cxr

    >>> cxr.cartesian_rep(cxr.Cart1D)
    coordinax...Cart1D

    >>> cxr.cartesian_rep(cxr.Radial1D)
    coordinax...Cart1D

    >>> cxr.cartesian_rep(cxr.Spherical3D)
    coordinax...Cart3D

    >>> cxr.cartesian_rep(cxr.FourVector)
    coordinax...Cart3D

    >>> cxr.cartesian_rep(cxr.CartND)
    coordinax...CartND

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def coord_map(
    to_rep: "coordinax.r.AbstractRep",
    from_rep: "coordinax.r.AbstractRep",
    p: PDict,
    /,
) -> PDict:
    """Return the coordinate mapping dictionary from one representation to another.

    Examples
    --------
    >>> import coordinax.r as cxr

    >>> cxr.coord_map(cxr.Cart1D, cxr.Radial1D, {})
    {'r': 'x'}

    >>> cxr.coord_map(cxr.Cart3D, cxr.Spherical3D, {})
    {'x': 'r*sin(theta)*cos(phi)', 'y': 'r*sin(theta)*sin(phi)', 'z': 'r*cos(theta)'}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def diff_map(
    to_rep: "coordinax.r.AbstractRep",
    from_rep: "coordinax.r.AbstractRep",
    p_dif: PDict,
    p_pos: PDict,
    /,
) -> PDict:
    """Return the time-differential coordinate representation mapping."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def metric_of(*args: Any) -> "coordinax.r.AbstractMetric":
    """Return the metric associated with a representation type.

    Examples
    --------
    >>> import coordinax as cx
    >>> metric = cx.r.metric_of(cx.r.cart3d)
    >>> metric.metric_matrix(cx.r.cart3d, {}).shape
    (3, 3)

    """
    raise NotImplementedError  # pragma: no cover
