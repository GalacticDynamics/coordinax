"""Representations."""

__all__ = (
    "add",
    "cconvert",
    "change_basis",
    "guess_basis_kind",
    "guess_geometry_kind",
    "guess_rep",
    "guess_semantic_kind",
    "subtract",
)

from typing import Any

import plum


@plum.dispatch.abstract
def change_basis(*args: Any, **kwargs: Any) -> Any:
    """Change the basis of a tangent vector's components.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc
    >>> v = {"x": 1.0, "y": 0.0}
    >>> at = {"x": 1.0, "y": 0.0}
    >>> cxr.change_basis(v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at)
    {'x': 1.0, 'y': 0.0}
    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def cconvert(*args: Any, **kwargs: Any) -> Any:
    """Transform the current vector to the target chart.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc

    Define a point in Cartesian coordinates:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}

    Convert it to spherical coordinates:

    >>> cxr.cconvert(p, cxc.cart3d, cxr.point, cxc.sph3d, cxr.point)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_basis_kind(*args: Any, **kwargs: Any) -> Any:
    """Guess the basis kind of the given data.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> data = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.guess_basis_kind(data)
    NoBasis()

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_geometry_kind(*args: Any, **kwargs: Any) -> Any:
    """Guess the geometry kind of the given data.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> data = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.guess_geometry_kind(data)
    PointGeometry()

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_rep(*args: Any, **kwargs: Any) -> Any:
    """Guess the representation of the given data.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> data = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.guess_rep(data)
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_semantic_kind(*args: Any, **kwargs: Any) -> Any:
    """Guess the semantic kind of the given data.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> data = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.guess_semantic_kind(data)
    Location()
    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def add(*args: Any, **kwargs: Any) -> Any:
    """Add two coordinate data objects.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.
    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def subtract(*args: Any, **kwargs: Any) -> Any:
    """Subtract two coordinate data objects.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.
    """
    raise NotImplementedError  # pragma: no cover
