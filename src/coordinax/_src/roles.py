"""Vector."""

from typing import ClassVar, final, overload

__all__ = (
    "AbstractRole",
    "AbstractPhysicalRole",
    "Point",
    "point",
    "Pos",
    "pos",
    "Vel",
    "vel",
    "Acc",
    "acc",
    "DIM_TO_ROLE_MAP",
)

import abc
import functools as ft

from typing import Any

import unxt as u

import coordinax._src.charts as cxc

Time = u.dimension("time")


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
    return u.dimension(dim) / (Time**order)


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

    @classmethod
    @abc.abstractmethod
    def derivative(cls) -> type["AbstractRole"]:
        """Return role flag for the time derivative of this role."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def antiderivative(cls) -> type["AbstractRole"]:
        """Return role flag for the time antiderivative of this role."""
        raise NotImplementedError  # pragma: no cover


class AbstractPhysicalRole(AbstractRole, abc.ABC):
    """Abstract base class for physical tangent roles.

    Physical tangent roles have uniform physical dimensions across components
    and transform via `physical_tangent_transform`.
    """


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

    @classmethod
    def derivative(cls) -> type[AbstractRole]:
        """Return role flag for the time derivative of this role."""
        return Vel

    @classmethod
    def antiderivative(cls) -> type[AbstractRole]:
        """Return role flag for the time antiderivative of this role."""
        return NotImplemented


point = Point()


@final
class Pos(AbstractPhysicalRole):
    r"""Pos role flag: physical displacement vectors with uniform length units.

    Mathematical Definition
    -----------------------
    A **position difference** or **physical displacement** represents a
    **physical vector** in the tangent space T_pM, expressed using
    **orthonormal frame components** at a base point p.

    CRITICAL: All Pos components must have uniform dimension [length].
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
    | Pos          | physical_tangent_transform  | Sometimes*         |
    | Vel          | physical_tangent_transform  | Sometimes*         |
    | Acc          | physical_tangent_transform  | Sometimes*         |

    *Required for embedded/manifold charts; optional for Euclidean spaces.

    Pos, Velocity, and Acceleration are geometrically the same
    (tangent vectors), differing only in physical units:
    - Pos: [length]
    - Velocity: [length/time]
    - Acceleration: [length/time²]

    Addition Rules
    --------------
    - ``Pos + Pos -> Pos`` (same tangent space)
    - ``Point + Pos -> Point`` (affine translation)
    - ``Pos + Point`` is **not** defined (not commutative)
    - ``Point + Point`` is **not** defined

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Create a position-difference vector in Cartesian coordinates:

    >>> disp = cx.Vector(
    ...     {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
    ...     cx.charts.cart3d,
    ...     cx.roles.pos,
    ... )

    In cylindrical coordinates, ALL components have length units:

    >>> disp_cyl = cx.Vector(
    ...     {"rho": u.Q(1.0, "m"), "phi": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
    ...     cx.charts.cyl3d,
    ...     cx.roles.pos,
    ... )

    The phi=2m means "2 meters in the tangential direction", not "2 radians".

    See Also
    --------
    Point : An affine point on a manifold.
    Vel : A velocity vector (position-difference per unit time).

    """

    order: ClassVar[int] = 0  # Same dimension as position (length)

    @classmethod
    def derivative(cls) -> type[AbstractRole]:
        """Return role flag for the time derivative of this role."""
        return Vel  # d(pos)/dt has velocity units

    @classmethod
    def antiderivative(cls) -> type[AbstractRole]:
        """Return role flag for the time antiderivative of this role."""
        return NotImplemented  # No standard meaning


pos = Pos()


@final
class Vel(AbstractPhysicalRole):
    """Velocity role flag (1st time derivative)."""

    order: ClassVar[int] = 1

    @classmethod
    def derivative(cls) -> type[AbstractRole]:
        """Return role flag for the time derivative of this role."""
        return Acc

    @classmethod
    def antiderivative(cls) -> type[AbstractRole]:
        """Return role flag for the time antiderivative of this role."""
        return Pos  # Now Pos is the physical displacement role


vel = Vel()


@final
class Acc(AbstractPhysicalRole):
    """Acceleration role flag (2nd time derivative)."""

    order: ClassVar[int] = 2

    @classmethod
    def derivative(cls) -> type[AbstractRole]:
        """Return role flag for the time derivative of this role."""
        return NotImplemented

    @classmethod
    def antiderivative(cls) -> type[AbstractRole]:
        """Return role flag for the time antiderivative of this role."""
        return Vel


acc = Acc()


# Mapping from dimension to role flag
DIM_TO_ROLE_MAP: dict[u.AbstractDimension, type["AbstractRole"]] = {
    u.dimension("length"): Point,  # Length → Point (affine location, per spec)
    u.dimension("angle"): Point,  # Angles → affine (mixed with other coords)
    u.dimension("speed"): Vel,  # Length/time → velocity
    u.dimension("angular speed"): Vel,  # Angle/time → velocity
    u.dimension("acceleration"): Acc,  # Length/time² → acceleration
    u.dimension("angular acceleration"): Acc,  # Angle/time² → acceleration
}
