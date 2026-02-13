"""Vector."""

__all__ = (
    "AbstractRole",
    "AbstractPhysRole",
    "AbstractCoordRole",
    "Point",
    "point",
    "PhysDisp",
    "phys_disp",
    "PhysVel",
    "phys_vel",
    "PhysAcc",
    "phys_acc",
    "CoordDisp",
    "coord_disp",
    "CoordVel",
    "coord_vel",
    "CoordAcc",
    "coord_acc",
    "DIM_TO_ROLE_MAP",
    "guess_role",
)

import abc
import functools as ft

from typing import Any, ClassVar, final, overload

import plum

import unxt as u

import coordinax.api as cxapi
import coordinax.charts as cxc
from coordinax._src.constants import (
    ACCELERATION,
    ANGLE,
    ANGULAR_ACCELERATION,
    ANGULAR_SPEED,
    LENGTH,
    SPEED,
    TIME,
)
from coordinax.api import CsDict


@overload
def d_dt_dim(dim: None, order: int, /) -> None: ...
@overload
def d_dt_dim(dim: u.AbstractDimension | str, order: int, /) -> u.AbstractDimension: ...
def d_dt_dim(
    dim: u.AbstractDimension | str | None, order: int, /
) -> u.AbstractDimension | None:
    """Return the dimension of the time derivative of the given dimension."""
    if dim is None:
        return None
    return u.dimension(dim) / (TIME**order)


class AbstractRole:
    """Flag for vector role (position, velocity, acceleration, etc.).

    Attributes
    ----------
    order : int
        Time-derivative order of the role (e.g. 0=pos, 1=vel, 2=acc, ...).

    """

    order: ClassVar[int]
    """Time-derivative order of the role (e.g. 0=pos, 1=vel, 2=acc, ...)."""

    @classmethod
    @ft.cache
    def dimensions(
        cls, chart: cxc.AbstractChart[Any, Any], /
    ) -> dict[str, u.AbstractDimension | None]:
        """Return the dimensions for this role for the given representation."""
        return {
            c: d_dt_dim(d, cls.order)
            for c, d in zip(chart.components, chart.coord_dimensions, strict=True)
        }

    @abc.abstractmethod
    def derivative(self) -> "AbstractRole":
        """Return role flag for the time derivative of this role."""
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def antiderivative(self) -> "AbstractRole":
        """Return role flag for the time antiderivative of this role."""
        raise NotImplementedError  # pragma: no cover

    def __eq__(self, other: object) -> bool:
        """Check equality between roles."""
        if type(self) is not type(other):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        """Hash a role based on its type."""
        return hash(type(self))


@final
class Point(AbstractRole):
    """Point role flag (affine point data).

    Mathematical Definition
    -----------------------
    A **point** is an element of a manifold or affine space M.
    Points do not form a vector space in general; e.g., on a curved manifold
    you cannot meaningfully add two points.

    - In Euclidean space ℝⁿ, points can be identified with vectors from
      the origin, but this identification is basis-dependent.
    - On manifolds (e.g., a sphere), points are elements of the manifold
      and have no additive structure.
    - Point coordinates may have mixed dimensions (e.g., spherical: length + angles).

    See Also
    --------
    Pos : A tangent vector representing a physical displacement.

    """

    order: ClassVar[int] = 0

    def derivative(self) -> AbstractRole:
        """Return role flag for the time derivative of this role."""
        return phys_vel

    def antiderivative(self) -> AbstractRole:
        """Return role flag for the time antiderivative of this role."""
        return NotImplemented

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}()"


point = Point()


# ===================================================================
# Physical Roles


class AbstractPhysRole(AbstractRole):
    """Abstract base class for physical tangent roles.

    Physical tangent roles have uniform physical dimensions across components
    and transform via `physical_tangent_transform`.
    """

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}()"


@final
class PhysDisp(AbstractPhysRole):
    r"""Displacement role flag: physical displacement vectors with uniform length units.

    Mathematical Definition
    -----------------------
    A **position difference** or **physical displacement** represents a
    **physical vector** in the tangent space T_pM, expressed using
    **orthonormal frame components** at a base point p.

    CRITICAL: All PhysDisp components must have uniform dimension [length].
    This is NOT a coordinate increment (which would have mixed units in curvilinear
    coordinates).

    Physical Components vs. Coordinate Increments
    ---------------------------------------------
    In cylindrical coordinates ($\rho$, $\phi$, z):

    - **Physical Pos** (this class): (rho=1m, phi=2m, z=3m) ✓
      where phi is the physical tangential length component
    - **Coordinate increment** (NOT this class): (Δrho=1m, Δphi=0.5rad, Δz=3m) ✗

    For example, phi=2m means "2 meters in the tangential direction" at the
    base point, NOT "2 radians of angular displacement".

    Transformation Rule
    -------------------
    Pos transforms via **physical_tangent_transform** (pushforward / tangent_transform),
    the SAME rule as Velocity and Acceleration:

    | Role         | Transform via                | Base point needed? |
    |--------------|-----------------------------|--------------------|
    | Point        | point_transform              | No                 |
    | PhysDisp     | physical_tangent_transform  | Sometimes*         |
    | PhysVel      | physical_tangent_transform  | Sometimes*         |
    | PhysAcc      | physical_tangent_transform  | Sometimes*         |

    *Required for embedded/manifold charts; optional for Euclidean spaces.

    PhysDisp, Velocity, and Acceleration are geometrically the same
    (tangent vectors), differing only in physical units:
    - PhysDisp: [length]
    - Velocity: [length/time]
    - Acceleration: [length/time²]

    Addition Rules
    --------------
    - ``PhysDisp + PhysDisp -> PhysDisp`` (same tangent space)
    - ``Point + PhysDisp -> Point`` (affine translation)
    - ``PhysDisp + Point`` is **not** defined (not commutative)
    - ``Point + Point`` is **not** defined

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.roles as cxr
    >>> import unxt as u

    Create a position-difference vector in Cartesian coordinates:

    >>> disp = cx.Vector(
    ...     {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
    ...     cxc.cart3d,
    ...     cxr.phys_disp,
    ... )

    In cylindrical coordinates, ALL components have length units:

    >>> disp_cyl = cx.Vector(
    ...     {"rho": u.Q(1.0, "m"), "phi": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
    ...     cxc.cyl3d,
    ...     cxr.phys_disp,
    ... )

    The phi=2m means "2 meters in the tangential direction", not "2 radians".

    See Also
    --------
    Point : An affine point on a manifold.
    PhysVel : A velocity vector (position-difference per unit time).

    """

    order: ClassVar[int] = 0  # Same dimension as position (length)

    def derivative(self) -> AbstractRole:
        """Return role flag for the time derivative of this role."""
        return phys_vel  # d(pos)/dt has velocity units

    def antiderivative(self) -> AbstractRole:
        """Return role flag for the time antiderivative of this role."""
        return NotImplemented  # No standard meaning


phys_disp = PhysDisp()


@final
class PhysVel(AbstractPhysRole):
    """Velocity role flag (1st time derivative)."""

    order: ClassVar[int] = 1

    def derivative(self) -> AbstractRole:
        """Return role flag for the time derivative of this role."""
        return phys_acc

    def antiderivative(self) -> AbstractRole:
        """Return role flag for the time antiderivative of this role."""
        return phys_disp


phys_vel = PhysVel()


@final
class PhysAcc(AbstractPhysRole):
    """Acceleration role flag (2nd time derivative)."""

    order: ClassVar[int] = 2

    def derivative(self) -> AbstractRole:
        """Return role flag for the time derivative of this role."""
        return NotImplemented

    def antiderivative(self) -> AbstractRole:
        """Return role flag for the time antiderivative of this role."""
        return phys_vel


phys_acc = PhysAcc()


# ==================================================================
# Coordinate-basis Tangent Roles


class AbstractCoordRole(AbstractRole):
    """Abstract base class for coordinate-basis tangent roles.

    Coordinate-basis tangent roles transform via `coord_transform`
    and have per-component physical dimensions according to the chart's
    coordinate dimensions.
    """

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}()"


@final
class CoordDisp(AbstractCoordRole):
    """Coordinate-basis displacement components (contravariant in chart basis)."""

    order: ClassVar[int] = 0

    def derivative(self) -> AbstractCoordRole:
        return coord_vel

    def antiderivative(self) -> AbstractCoordRole:
        return NotImplemented


coord_disp = CoordDisp()


@final
class CoordVel(AbstractCoordRole):
    """Coordinate-basis velocity components (contravariant in chart basis)."""

    order: ClassVar[int] = 1

    def derivative(self) -> AbstractCoordRole:
        return coord_acc

    def antiderivative(self) -> AbstractCoordRole:
        return coord_disp


coord_vel = CoordVel()


@final
class CoordAcc(AbstractCoordRole):
    """Coordinate-basis acceleration components (contravariant in chart basis)."""

    order: ClassVar[int] = 2

    def derivative(self) -> AbstractCoordRole:
        return NotImplemented

    def antiderivative(self) -> AbstractCoordRole:
        return coord_vel


coord_acc = CoordAcc()


# ============================================================================
# Helpers

# Mapping from dimension to role flag
DIM_TO_ROLE_MAP: dict[u.AbstractDimension, type["AbstractRole"]] = {
    LENGTH: Point,  # Length → Point (affine location, per spec)
    ANGLE: Point,  # Angles → affine (mixed with other coords)
    SPEED: PhysVel,  # Length/time → velocity
    ANGULAR_SPEED: PhysVel,  # Angle/time → velocity
    ACCELERATION: PhysAcc,  # Length/time² → acceleration
    ANGULAR_ACCELERATION: PhysAcc,  # Angle/time² → acceleration
}


@plum.dispatch
def guess_role(obj: u.AbstractDimension, /) -> AbstractRole:
    """Infer role flag from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.roles as cxr

    >>> r1 = cxr.guess_role(u.dimension("length"))
    >>> r1
    Point()

    >>> r2 = cxr.guess_role(u.dimension("speed"))
    >>> r2
    PhysVel()

    >>> r3 = cxr.guess_role(u.dimension("acceleration"))
    >>> r3
    PhysAcc()

    """
    try:
        role_cls = DIM_TO_ROLE_MAP[obj]
    except KeyError as e:
        msg = f"Cannot infer role from dimension {obj}"
        raise ValueError(msg) from e
    return role_cls()


@plum.dispatch
def guess_role(x: u.AbstractQuantity, /) -> AbstractRole:
    """Infer role flag from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.roles as cxr

    >>> r1 = cxr.guess_role(u.Q(1.0, "m"))
    >>> r1
    Point()

    >>> r2 = cxr.guess_role(u.Q(2.0, "m / s"))
    >>> r2
    PhysVel()

    >>> r3 = cxr.guess_role(u.Q(3.0, "m / s ** 2"))
    >>> r3
    PhysAcc()

    """
    dim = u.dimension_of(x)
    return cxapi.guess_role(dim)


@plum.dispatch
def guess_role(obj: CsDict, /) -> AbstractRole:
    """Infer role flag from the physical dimensions of a component dictionary.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.roles as cxr

    >>> d1 = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> r1 = cxr.guess_role(d1)
    >>> r1
    Point()

    >>> d2 = {"x": u.Q(1.0, "m / s"), "y": u.Q(2.0, "m / s")}
    >>> r2 = cxr.guess_role(d2)
    >>> r2
    PhysVel()

    """
    dims = {u.dimension_of(v) for v in obj.values()}
    if len(dims) != 1:
        msg = f"Cannot infer role from mixed dimensions: {dims}"
        raise ValueError(msg)
    dim = dims.pop()
    return cxapi.guess_role(dim)
