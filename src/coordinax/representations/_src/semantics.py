"""Vector semantic kind."""

__all__ = (
    "AbstractSemanticKind",
    # Location
    "Location",
    "loc",
    # Tangent semantic kinds
    "AbstractTangentSemanticKind",
    "Displacement",
    "dpl",
    "Velocity",
    "vel",
    "Acceleration",
    "acc",
)

import abc
import dataclasses
import functools as ft

from typing import Any, ClassVar, Final, final, overload

import jax.tree_util as jtu
import wadler_lindig as wl

import unxt as u
from dataclassish import field_items

import coordinax.charts as cxc
from .constants import TIME


@jtu.register_static
class AbstractSemanticKind(metaclass=abc.ABCMeta):
    r"""Abstract base class for semantic kind.

    A semantic kind specifies the **meaning** attached to a represented
    geometric object, independent of the underlying geometric type, basis, and
    chart used to express its components.

    In the representation model used by `coordinax`, the full representation is
    determined by three orthogonal pieces of information:

    1. the geometric kind,
    2. the basis, and
    3. the semantic kind.

    This class provides the third of these pieces: it answers the question "what
    does this geometric object represent?" Examples include location for points,
    and later may include displacement, velocity, or acceleration for
    tangent-like objects.

    Mathematical Role:

    The semantic kind refines the interpretation of data within a fixed
    geometric type and basis.

    - For a **point** with `NoBasis`, the semantic kind `Location` indicates
      that the data represents where a point lies on a manifold.
    - For a **tangent vector**, different semantic kinds may distinguish
      displacement from velocity or acceleration, even when the underlying
      transformation law is the same.
    - For a **cotangent object**, semantic kinds may distinguish different dual
      interpretations without changing the underlying covector character.

    Thus semantic kind is distinct from:

    - a **chart**, which specifies how local coordinates are assigned,
    - a **geometry kind**, which specifies what sort of geometric object the
      data represents, and
    - a **basis**, which specifies in what basis the components are written when
      such a choice is meaningful.

    Examples
    --------
    >>> import coordinax.main as cx

    Construct the location semantic object directly:

    >>> semantic = cx.Location()

    With the semantic object, we construct a full representation for point data:

    >>> rep = cx.Representation(cx.point_geom, cx.no_basis, semantic)

    The representation can then be used with `cconvert` to convert point data
    between charts while preserving the fact that the data represents a
    location:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> cx.cconvert(p, cx.cart3d, rep, cx.sph3d, rep)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output is still point data representing the same location, but
    expressed in the target chart.

    Notes
    -----
    This is a static dispatch object and carries no runtime numerical data.
    Concrete subclasses should represent immutable semantic categories.

    """

    canonical_name: ClassVar[str | None] = None
    """Canonical name for the geometric kind."""

    # ===============================================================
    # Dimension API

    @classmethod
    @abc.abstractmethod
    def coord_dimensions(
        cls, chart: cxc.AbstractChart[Any, Any], /
    ) -> tuple[u.AbstractDimension | None, ...]:
        """Return the physical dimensions of the components for this semantic kind.

        Parameters
        ----------
        chart
            The chart defining component names and base coordinate dimensions.

        Returns
        -------
        tuple[u.AbstractDimension | None, ...]
            One entry per component: the physical dimension of that component
            under this semantic kind, or ``None`` if the component is
            dimensionless.

        """
        ...

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, canonical: bool = False, **kw: Any) -> wl.AbstractDoc:
        """Generate a Wadler-Lindig docstring for this Basis.

        Parameters
        ----------
        canonical
            Whether to use the canonical forms of the representation in the
            docstring. E.g. `PointGeometry()` -> `point_geom`.
        **kw
            Additional keyword arguments to pass to the Wadler-Lindig docstring
            formatter.

        Examples
        --------
        >>> import wadler_lindig as wl
        >>> import coordinax.representations as cxr

        >>> semantic = cxr.Location()
        >>> wl.pprint(semantic)
        Location()

        >>> wl.pprint(semantic, canonical=True)
        loc

        """
        if canonical and self.canonical_name is not None:
            return wl.TextDoc(self.canonical_name)

        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=wl.named_objs(field_items(self), **kw),
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kw.get("indent", 4),
        )


# ===================================================================


@final
@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class Location(AbstractSemanticKind):
    r"""Location semantic kind.

    A location semantic kind indicates that the represented data specifies
    **where** a geometric object is, rather than how it is displaced, how fast
    it moves, or how it accelerates.

    Mathematical Definition:

    `Location` is the canonical semantic kind for point data. Let $M$ be a
    smooth manifold. A point is an element $p \in M$, and `Location` indicates
    that the represented data should be interpreted as the coordinates of that
    point in some chart.

    The semantic kind `Location` therefore does not change the underlying
    geometric transformation law: for point data, coordinates still transform by
    the ordinary chart transition map. Instead, it records the interpretation of
    the point-like data as an actual position on the manifold.

    Examples
    --------
    Construct the location semantic object directly:

    >>> import coordinax.representations as cxr
    >>> semantic = cxr.Location()

    Use it inside a full representation for point data:

    >>> rep = cxr.Representation(cxr.point_geom, cxr.no_basis, semantic)

    The representation can then be used with `cconvert` to convert point data
    between charts while preserving the fact that the data represents a
    location:

    >>> import coordinax.charts as cxc
    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> cxr.cconvert(p, cxc.cart3d, rep, cxc.sph3d, rep)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output is still point data representing the same location, but
    expressed in the target chart.

    Notes
    -----
    `Location` does not by itself imply that the represented object is a point,
    but in the current `coordinax` design it is primarily used as the semantic
    kind paired with `PointGeometry`.

    """

    canonical_name: ClassVar = "loc"
    """Canonical name for the location semantic kind."""

    @classmethod
    @ft.cache
    def coord_dimensions(
        cls, chart: cxc.AbstractChart[Any, Any], /
    ) -> tuple[u.AbstractDimension | None, ...]:
        """Return the physical dimensions of each component for a location.

        For a location semantic kind, the dimensions are determined solely by
        the chart: each component's dimension is taken directly from
        ``chart.coord_dimensions``.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> import coordinax.representations as cxr
        >>> cxr.Location.coord_dimensions(cxc.cart3d)
        (PhysicalType('length'), PhysicalType('length'), PhysicalType('length'))

        """
        return tuple(  # ty: ignore[invalid-return-type]
            u.dimension(d) if d is not None else None for d in chart.coord_dimensions
        )


loc = Location()
"""Instance of `Location`."""


# ===================================================================


@overload
def d_dt_dim(dim: None, order: int, /) -> None: ...
@overload
def d_dt_dim(dim: u.AbstractDimension | str, order: int, /) -> u.AbstractDimension: ...
def d_dt_dim(
    dim: u.AbstractDimension | str | None, order: int, /
) -> u.AbstractDimension | None:
    """Return the physical dimension after ``order`` time derivatives.

    Computes ``dimension(dim) / time**order``. If ``dim`` is ``None``, the
    component is dimensionless and ``None`` is returned unchanged.

    Parameters
    ----------
    dim
        The base dimension, given either as a `u.AbstractDimension` instance or
        as a string (e.g. ``"length"``, ``"angle"``). Pass ``None`` to
        indicate a dimensionless component.
    order
        The number of time derivatives to apply. ``0`` leaves the dimension
        unchanged; ``1`` divides by time once (e.g. length → speed); ``2``
        divides by time twice (e.g. length → acceleration).

    Returns
    -------
    u.AbstractDimension | None
        The resulting dimension, or ``None`` if ``dim`` was ``None``.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.representations._src.semantics import d_dt_dim

    Zeroth derivative (no change):

    >>> d_dt_dim("length", 0)
    PhysicalType('length')

    First time derivative of length gives speed:

    >>> d_dt_dim("length", 1)
    PhysicalType({...'speed'...})

    Second time derivative of length gives acceleration:

    >>> d_dt_dim("length", 2)
    PhysicalType('acceleration')

    Angular dimension at first order gives angular speed:

    >>> d_dt_dim("angle", 1)
    PhysicalType({...'angular speed'...})

    Dimensionless components (``None``) pass through unchanged:

    >>> d_dt_dim(None, 1) is None
    True

    """
    if dim is None:
        return None
    return u.dimension(dim) / (TIME**order)  # ty: ignore[unsupported-operator]


@jtu.register_static
class AbstractTangentSemanticKind(AbstractSemanticKind):
    r"""Abstract base class for tangent-vector semantic kinds.

    A tangent semantic kind specifies the **meaning** of a tangent-vector
    object, distinguishing objects that are geometrically identical (both
    elements of a tangent space $T_p M$) but semantically different in how they
    are used or interpreted.

    Examples include:

    - `Displacement`: a finite difference between two nearby points,
    - `Velocity`: the rate of change of position with respect to time,
    - `Acceleration`: the second derivative of position with respect to time.

    All share the same coordinate-transformation law (Jacobian pushforward)
    but differ in physical dimension and physical interpretation.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> isinstance(cxr.Displacement(), cxr.AbstractTangentSemanticKind)
    True
    >>> isinstance(cxr.Velocity(), cxr.AbstractTangentSemanticKind)
    True

    """

    order: ClassVar[int]
    """Time-derivative order of the role (e.g. 0=pos, 1=vel, 2=acc, ...)."""

    @classmethod
    @ft.cache
    def coord_dimensions(
        cls, chart: cxc.AbstractChart[Any, Any], /
    ) -> tuple[u.AbstractDimension | None, ...]:
        """Return the physical dimensions of each component for this tangent kind.

        Each component's base dimension is taken from ``chart.coord_dimensions``
        and scaled by the appropriate power of time according to ``cls.order``:
        dimension / time^order.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> import coordinax.representations as cxr

        >>> [str(x) for x in cxr.Displacement.coord_dimensions(cxc.cart3d)]
        ['length', 'length', 'length']
        >>> [str(x) for x in cxr.Velocity.coord_dimensions(cxc.cart3d)]
        ['speed/...', 'speed/...', 'speed/...']
        >>> [str(x) for x in cxr.Acceleration.coord_dimensions(cxc.cart3d)]
        ['acceleration', 'acceleration', 'acceleration']

        >>> [str(x) for x in cxr.Displacement.coord_dimensions(cxc.sph3d)]
        ['length', 'angle', 'angle']
        >>> [str(x) for x in cxr.Velocity.coord_dimensions(cxc.sph3d)]
        ['speed/...', 'angular frequency/...', 'angular frequency/...']
        >>> [str(x) for x in cxr.Acceleration.coord_dimensions(cxc.sph3d)]
        ['acceleration', 'angular acceleration', 'angular acceleration']

        """
        return tuple(d_dt_dim(d, cls.order) for d in chart.coord_dimensions)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        r"""Register tangent semantic kinds in the internal time-order registry.

        Every concrete subclass must define an ``order`` class variable (an
        integer). On class creation this hook validates that requirement,
        raising `TypeError` for missing or non-integer values, then stores
        ``cls`` in an internal registry under that integer key. The registry is
        then used by `derivative` and `antiderivative` to navigate the
        time-derivative chain.

        The built-in chain is:

        .. code-block:: text

           ... <--> Displacement (0) <--> Velocity (1) <--> Acceleration (2) <--> ...

        New classes can be registered by simply defining a subclass.  For
        example, to add *absement* (the time-integral of displacement) at
        order -1::

            @final
            @jtu.register_static
            @dataclasses.dataclass(frozen=True, slots=True)
            class Absement(AbstractTangentSemanticKind):
                canonical_name = "abs"
                order = -1

        After this definition ``Displacement().antiderivative()`` returns
        ``Absement()`` automatically.

        """
        if hasattr(super(), "__init_subclass__"):
            super().__init_subclass__(**kwargs)

        if not hasattr(cls, "order"):
            raise TypeError(
                f"{cls.__name__!r} must define class variable 'order' as an int "
                "before registration in the tangent time-order ladder."
            )

        order = cls.order
        if not isinstance(order, int):
            raise TypeError(
                f"{cls.__name__!r}.order must be an int, got {type(order).__name__}."
            )

        if order in _TANGENT_TIME_ORDER_LADDER:
            existing = _TANGENT_TIME_ORDER_LADDER[order]
            # Allow re-registration when it is the same class being rebuilt.
            # @dataclasses.dataclass(slots=True) recreates the class via
            # type(cls)(cls.__name__, ...) which fires __init_subclass__ a
            # second time.  The rebuilt class has the same __name__ and
            # __module__ as the original, so we use those to identify it.
            same_class = (
                existing.__name__ == cls.__name__
                and existing.__module__ == cls.__module__
            )
            if not same_class:
                raise TypeError(
                    f"Cannot register {cls.__name__!r} at time-derivative order "
                    f"{order}: order {order} is already occupied by "
                    f"{existing.__name__!r}.  Choose a different order or "
                    f"remove the existing registration first."
                )

        _TANGENT_TIME_ORDER_LADDER[order] = cls

    def derivative(self) -> "AbstractTangentSemanticKind":
        """Return the semantic kind one step up the time-derivative ladder.

        Looks up ``self.order + 1`` in the internal order registry and returns
        a fresh instance of the registered class.  Raises `ValueError` if no
        class is registered at that order.

        This design is open for extension: registering a new
        `AbstractTangentSemanticKind` subclass at order ``N`` automatically
        makes ``kind_at_N_minus_1.derivative()`` return an instance of that
        class.

        Raises
        ------
        ValueError
            If no tangent semantic kind is registered at ``self.order + 1``.

        Examples
        --------
        >>> import coordinax.representations as cxr
        >>> cxr.Displacement().derivative()
        Velocity()
        >>> cxr.Velocity().derivative()
        Acceleration()
        >>> try:
        ...     cxr.Acceleration().derivative()   # no Jerk registered yet
        ... except ValueError as e:
        ...     print(type(e).__name__)
        ValueError

        """
        order = self.order + 1
        if order not in _TANGENT_TIME_ORDER_LADDER:
            raise ValueError(
                f"No tangent semantic kind defined for time derivative of order {order}"
            )
        return _TANGENT_TIME_ORDER_LADDER[order]()

    def antiderivative(self) -> "AbstractTangentSemanticKind":
        """Return the semantic kind one step down the time-derivative ladder.

        Looks up ``self.order - 1`` in the internal order registry and returns
        a fresh instance of the registered class.  Raises `ValueError` if no
        class is registered at that order.

        This design is open for extension: registering a new
        `AbstractTangentSemanticKind` subclass at order ``N`` automatically
        makes ``kind_at_N_plus_1.antiderivative()`` return an instance of that
        class.  For example, once an *Absement* class is registered at order -1,
        ``Displacement().antiderivative()`` will return ``Absement()``.

        Raises
        ------
        ValueError
            If no tangent semantic kind is registered at ``self.order - 1``.

        Examples
        --------
        >>> import coordinax.representations as cxr
        >>> cxr.Acceleration().antiderivative()
        Velocity()
        >>> cxr.Velocity().antiderivative()
        Displacement()
        >>> try:
        ...     cxr.Displacement().antiderivative()   # no Absement registered yet
        ... except ValueError as e:
        ...     print(type(e).__name__)
        ValueError

        """
        order = self.order - 1
        if order not in _TANGENT_TIME_ORDER_LADDER:
            raise ValueError(
                f"No tangent semantic kind defined for time antiderivative of order {order}"  # noqa: E501
            )
        return _TANGENT_TIME_ORDER_LADDER[order]()


_TANGENT_TIME_ORDER_LADDER: Final[dict[int, type[AbstractTangentSemanticKind]]] = {}


@final
@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class Displacement(AbstractTangentSemanticKind):
    r"""Displacement semantic kind.

    A displacement semantic kind indicates that the represented tangent data
    is a **spatial displacement** — a finite difference between two nearby
    points on a manifold, expressed as a tangent vector.

    Mathematical Definition:

    Let $M$ be a smooth manifold and $p, q \in M$ two nearby points. In a
    chart $(U, \varphi)$, the displacement from $p$ to $q$ is the vector
    $\Delta x = \varphi(q) - \varphi(p) \in \mathbb{R}^n$. As a tangent
    vector, this element of $T_p M$ transforms by the Jacobian of the chart
    transition map.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> semantic = cxr.Displacement()
    >>> semantic.canonical_name
    'dpl'

    """

    canonical_name: ClassVar = "dpl"
    """Canonical name for the displacement semantic kind."""

    order: ClassVar[int] = 0
    """Time-derivative order of the role (0 for displacement)."""

    def derivative(self) -> "AbstractTangentSemanticKind":
        """Return the `Velocity` semantic kind.

        Displacement has time-order 0; its time derivative is always `Velocity`
        (order 1).  This override avoids a dict lookup and makes the intent
        explicit.

        Examples
        --------
        >>> import coordinax.representations as cxr
        >>> cxr.Displacement().derivative()
        Velocity()

        """
        return vel


dpl = Displacement()
"""Instance of `Displacement`."""


@final
@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class Velocity(AbstractTangentSemanticKind):
    r"""Velocity semantic kind.

    A velocity semantic kind indicates that the represented tangent data is a
    **velocity vector** — the first time derivative of position on a manifold.

    Mathematical Definition:

    Let $\gamma: \mathbb{R} \to M$ be a smooth curve on manifold $M$. The
    velocity at time $t$ is the tangent vector $\dot{\gamma}(t) \in
    T_{\gamma(t)} M$. In a chart, velocity components $\dot{x}^i =
    d(x^i \circ \gamma)/dt$ transform by the Jacobian of the chart transition
    map.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> semantic = cxr.Velocity()
    >>> semantic.canonical_name
    'vel'

    """

    canonical_name: ClassVar = "vel"
    """Canonical name for the velocity semantic kind."""

    order: ClassVar[int] = 1
    """Time-derivative order of the role (1 for velocity)."""

    def derivative(self) -> AbstractTangentSemanticKind:
        """Return the semantic kind for the time derivative of this velocity.

        Velocity has time-order 1. Its time derivative is acceleration
        (time-order 2).

        Examples
        --------
        >>> import coordinax.representations as cxr
        >>> cxr.Velocity().derivative()
        Acceleration()

        """
        return acc

    def antiderivative(self) -> AbstractTangentSemanticKind:
        """Return the semantic kind for the time antiderivative of this velocity.

        Velocity has time-order 1. Its time antiderivative is displacement
        (time-order 0).

        Examples
        --------
        >>> import coordinax.representations as cxr
        >>> cxr.Velocity().antiderivative()
        Displacement()

        """
        return dpl


vel = Velocity()
"""Instance of `Velocity`."""


@final
@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class Acceleration(AbstractTangentSemanticKind):
    r"""Acceleration semantic kind.

    An acceleration semantic kind indicates that the represented tangent data
    is an **acceleration vector** — the second time derivative of position on
    a manifold.

    Mathematical Definition:

    Let $\gamma: \mathbb{R} \to M$ be a smooth curve on manifold $M$. The
    acceleration at time $t$ is the tangent vector $\ddot{\gamma}(t) \in
    T_{\gamma(t)} M$ (or more precisely, $\nabla_{\dot{\gamma}}\dot{\gamma}$
    in the covariant sense, but in flat space simply the second derivative).

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> semantic = cxr.Acceleration()
    >>> semantic.canonical_name
    'acc'

    """

    canonical_name: ClassVar = "acc"
    """Canonical name for the acceleration semantic kind."""

    order: ClassVar[int] = 2
    """Time-derivative order of the role (2 for acceleration)."""

    def antiderivative(self) -> AbstractTangentSemanticKind:
        """Return the semantic kind for the time antiderivative of this acceleration.

        Acceleration has time-order 2. Its time antiderivative is velocity
        (time-order 1).

        Examples
        --------
        >>> import coordinax.representations as cxr
        >>> cxr.Acceleration().antiderivative()
        Velocity()

        """
        return vel


acc = Acceleration()
"""Instance of `Acceleration`."""
