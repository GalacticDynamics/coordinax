"""Point."""

__all__ = ("Point",)


from dataclasses import replace

from jaxtyping import ArrayLike
from typing import TYPE_CHECKING, Any, Generic, NoReturn, cast, final, override
from typing_extensions import TypeVar

import equinox as eqx
import plum
import quax_blocks

import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .base import AbstractVector, broadcast_and_index_data, check_dimensions
from .custom_types import CKey, HasShape
from .mixins import AstropyRepresentationAPIMixin
from coordinax.frames._src.base import frames_equal

if TYPE_CHECKING:
    from .custom_types import CDict

ChartT = TypeVar(
    "ChartT",
    bound=cxc.AbstractChart[Any, Any, Any],
    default=cxc.AbstractChart[Any, Any, Any],
)
GeomT = TypeVar("GeomT", bound=cxr.AbstractGeometry, default=cxr.AbstractGeometry)
BasisT = TypeVar("BasisT", bound=cxr.AbstractBasis, default=cxr.AbstractBasis)
SemanticT = TypeVar(
    "SemanticT", bound=cxr.AbstractSemanticKind, default=cxr.AbstractSemanticKind
)
V = TypeVar("V", bound=HasShape, default=u.Q)


def refuse_point_scaling(kind: str, /) -> NoReturn:
    """Raise the `TypeError` for scaling a point, which is affine (gh#1025)."""
    if kind == "Coordinate":
        msg = (
            "cannot multiply or divide a Coordinate by a scalar: its base point "
            "is affine, and scaling it has no meaning without an origin. Scale "
            "the fibre tangents individually instead, e.g. "
            "`k * coord['velocity']`."
        )
        raise TypeError(msg)
    msg = (
        f"cannot multiply or divide a {kind} by a scalar: a point is affine, and "
        "scaling it has no meaning without an origin. Scale a displacement "
        "instead: `k * (p - q)`, or `k * p.position_vector()` to scale the "
        "position vector about the origin."
    )
    raise TypeError(msg)


def refuse_point_sum(op: str, /) -> NoReturn:
    """Raise the `TypeError` for an origin-dependent sum of points."""
    if op.endswith(("Coordinate", "Coordinates")):
        msg = (
            f"cannot {op}: its base point is affine, and its fibres live at "
            "different base points. Work with the base points explicitly, "
            "`a.point - b.point`, and with the fibres one by one."
        )
        raise TypeError(msg)
    msg = (
        f"cannot {op}: a point is affine, and this has no meaning without an "
        "origin. `p - q` is the displacement from `q` to `p`, `p + d` "
        "translates `p` by a displacement `d`, and `p.position_vector()` is "
        "the displacement from the origin."
    )
    raise TypeError(msg)


def refuse_scaling_unless_vector(kind: str, other: Any, /) -> Any:
    """Refuse scaling ``kind`` by ``other``, unless ``other`` is itself a vector.

    A vector operand is not a scaling: return `NotImplemented` so Python reports
    the ordinary unsupported-operand error instead of the scalar explanation.
    """
    if isinstance(other, AbstractVector):
        return NotImplemented
    refuse_point_scaling(kind)


def _frame_converter(v: Any, /) -> cxf.AbstractReferenceFrame:
    """Convert a value to an AbstractReferenceFrame, with None -> noframe."""
    if v is None:
        return cxf.noframe
    if isinstance(v, cxf.AbstractReferenceFrame):
        return v
    return cxf.TransformedReferenceFrame.from_(v)  # ty: ignore[invalid-return-type]


@final
class Point(
    # IPythonReprMixin,
    AstropyRepresentationAPIMixin,
    quax_blocks.NumpyInvertMixin[Any],
    quax_blocks.LaxLenMixin,
    AbstractVector[ChartT, cxr.PointGeometry, cxr.NoBasis, cxr.Location, V],
    Generic[ChartT, V],
):
    r"""A coordinate-carrying geometric point.

    A `Point` stores three pieces of information:

    - **data**: a mapping from component name to scalar-like value (typically
      `unxt.Quantity`),
    - **chart**: a chart object describing the coordinate system and component
      schema, and
    - **rep**: a representation describing the *geometric meaning* of the
      components and therefore the correct transformation law.

    The design goal is to make the **public API simple** (construct, convert,
    index) while keeping the **mathematics correct** and the numerical kernels
    JAX-friendly (operate on scalar leaves; rely on `jit`/`vmap`).

    Mathematical background:

    Let $M$ be a manifold and let $(U,\varphi)$ be a chart with coordinate map
    $\varphi: U \to \mathbb{R}^n$. Coordinax distinguishes:

    **Point** (representation ``cxr.point``)
        A point $p \in M$ represented by its chart coordinates $q = \varphi(p)$.
        A point transforms by coordinate change: $q' = (\varphi' \circ
        \varphi^{-1})(q)$.

        In Euclidean charts, point coordinates may have *heterogeneous physical
        dimensions* (e.g. spherical $(r,\theta,\phi)$ mixes length and angle).
        This is expected.

    Parameters
    ----------
    data
        Mapping from chart component name to scalar value. Each leaf may be a
        `unxt.Quantity` (recommended) or an array-like. Components are expected
        to be *scalar leaves*; batching happens via broadcasting of these
        leaves.
    chart
        A chart instance (e.g. `cxc.cart3d`, `cxc.sph3d`) that defines component
        names and per-component physical dimensions.

    Examples
    --------
    Construct a **point** in Cartesian 3D and convert to spherical:

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> cart = cx.Point.from_({"x": u.Q(1, "m"), "y": u.Q(1, "m"), "z": u.Q(1, "m")},
    ...                  cxc.cart3d)
    >>> sph = cart.cconvert(cxc.sph3d)
    >>> sph["r"]
    Q(1.73205081, 'm')

    Notes
    -----
    Notes on units and array shape:

    - A `Point` does **not** require that all components share one unit. This
      is essential for charts like spherical coordinates where point components
      naturally mix dimensions.
    - Batching is represented by broadcasting the component leaves; the
      conceptual shape of the `Point` is `broadcast_shapes(*(v.shape for v in
      data.values()))`.

    Core operations:

    - Indexing: ``vec["x"]`` returns a component leaf.
    - Conversion: ``vec.cconvert(target_chart, at=...)`` converts the vector to
      `target_chart`. For ``Point`` this is a coordinate transform.

    """

    data: dict[CKey, Any]  # TODO: data: dict[CKey, V]
    """The data for each component."""

    chart: ChartT = eqx.field(static=True)
    """The chart of the vector, e.g. `cxc.cart3d`."""

    frame: cxf.AbstractReferenceFrame = eqx.field(
        default=cxf.noframe, converter=_frame_converter
    )
    """The reference frame of the point. Defaults to ``cxf.noframe``."""

    def __check_init__(self) -> None:
        self.M.check_chart(self.chart)
        self.chart.check_data(self.data, keys=True)

    @property
    def rep(self) -> cxr.Representation[cxr.PointGeometry, cxr.NoBasis, cxr.Location]:
        """The representation of the vector."""
        return cxr.point

    @override
    def __getitem__(self, key: Any) -> "V | Point":  # ty: ignore[invalid-method-override]
        if isinstance(key, str):
            return self.data[key]
        data = broadcast_and_index_data(self.data, self.shape, key)
        return replace(self, data=data)  # ty: ignore[invalid-return-type]

    # A point is affine, so scaling one is refused (gh#1025). The quax rules
    # (`mul_p_scalar_point`, ...) raise the same error, but the quax-blocks
    # operators turn a rule's `TypeError` into `NotImplemented`, and Python then
    # reports a bare "unsupported operand" that drops the explanation. (`-> Any`,
    # not `NoReturn`: jaxtyping's method wrapper rejects `NoReturn` there.)
    def __mul__(self, other: Any) -> Any:
        """Refuse ``point * k``: a point is affine.

        >>> import coordinax as cx
        >>> cx.Point.from_([1.0, 2.0, 3.0], "km") * 2
        Traceback (most recent call last):
        ...
        TypeError: cannot multiply or divide a Point by a scalar: a point is affine, ...

        """
        return refuse_scaling_unless_vector("Point", other)

    def __rmul__(self, other: Any) -> Any:
        return refuse_scaling_unless_vector("Point", other)

    def __truediv__(self, other: Any) -> Any:
        return refuse_scaling_unless_vector("Point", other)

    def __rtruediv__(self, other: Any) -> Any:
        return refuse_scaling_unless_vector("Point", other)

    # Same reason: call the rules directly so their `TypeError` reaches the
    # user. An operand with no rule (a `Coordinate`) gets `NotImplemented`, so
    # its reflected operator can explain itself.
    def __add__(self, other: Any) -> Any:
        """``p + d`` translates ``p`` by a displacement ``d``.

        >>> import coordinax as cx
        >>> p = cx.Point.from_([1.0, 2.0, 3.0], "km")
        >>> p + p
        Traceback (most recent call last):
        ...
        TypeError: cannot add two Points: a point is affine, ...

        """
        return _binop(cxr.add, self, other)

    def __radd__(self, other: Any) -> Any:
        return _binop(cxr.add, other, self)

    def __sub__(self, other: Any) -> Any:
        """``p - q`` is a displacement, ``p - d`` a point.

        >>> import coordinax as cx
        >>> p = cx.Point.from_([4.0, 5.0, 6.0], "km")
        >>> q = cx.Point.from_([1.0, 2.0, 3.0], "km")
        >>> print(p - q)
        <Tangent: chart=Cart3D (x, y, z) [km]
            [3. 3. 3.]>

        """
        return _binop(cxr.subtract, self, other)

    def __rsub__(self, other: Any) -> Any:
        return _binop(cxr.subtract, other, self)

    def subtract(self, other: Any, /, *, tau: Any = None) -> Any:
        """``self - other``, first transforming ``other`` into ``self.frame``.

        ``other`` is a `Point` (the result is the displacement `Tangent`) or a
        displacement `Tangent` (the result is a `Point`). When the frames
        differ, ``other`` is moved into ``self.frame`` with
        ``frame_transition(other.frame, self.frame)`` at ``tau``, which a
        time-dependent transition requires. In one frame this is ``self -
        other`` and ``tau`` is ignored.

        >>> import coordinax as cx
        >>> import coordinax.frames as cxf
        >>> p = cx.Point.from_([10.0, 0.0, 0.0], "m", cxf.alex)
        >>> q = cx.Point.from_([0.0, 0.0, 0.0], "m", cxf.alice)
        >>> print(p.subtract(q))
        <Tangent: chart=Cart3D (x, y, z) [m]
            [ 10. -10.   0.]>

        """
        return cxr.subtract(self, _into_frame(other, self, tau))

    def add(self, d: Any, /, *, tau: Any = None) -> "Point":
        """``self + d``, first transforming the displacement ``d`` into ``self.frame``.

        The frame handling is that of `Point.subtract`.

        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.frames as cxf
        >>> import coordinax.representations as cxr
        >>> p = cx.Point.from_([0.0, 0.0, 0.0], "m", cxf.alex)
        >>> d = cx.Tangent(cx.cdict(u.Q([1.0, 0.0, 0.0], "m"), cx.cart3d), cx.cart3d,
        ...                cxr.coord_basis, cxr.dpl, frame=cxf.alice)
        >>> print(p.add(d).round(4))
        <Point: chart=Cart3D (x, y, z) [m]
            [0. 1. 0.]>

        """
        return cast("Point", cxr.add(self, _into_frame(d, self, tau)))

    def position_vector(
        self, basis: cxr.AbstractLinearBasis = cxr.coord_basis, /
    ) -> Any:
        """Return the displacement ``p - O`` from the origin, written at ``p``.

        ``O`` is the origin of the Cartesian chart in this frame, so the
        result depends on the frame and does not commute with a `Translate`.
        It is a `Tangent` with `Displacement` semantics in ``self.chart`` and
        ``basis``. The manifold must be an affine space.

        >>> import coordinax as cx
        >>> import coordinax.charts as cxc
        >>> import coordinax.representations as cxr
        >>> p = cx.Point.from_([3.0, 0.0, 4.0], "m").cconvert(cxc.cyl3d)
        >>> print(p.position_vector(cxr.phys_basis).round(4))
        <Tangent: chart=Cylindrical3D (rho, phi, z) [m]
            [3. 0. 4.]>

        """
        return cxr.change_basis(self, basis)


def _binop(f: Any, lhs: Any, rhs: Any, /) -> Any:
    """Apply an add/subtract rule, or `NotImplemented` when there is none."""
    if not (isinstance(lhs, AbstractVector) and isinstance(rhs, AbstractVector)):
        return NotImplemented
    try:
        return f(lhs, rhs)
    except plum.NotFoundLookupError:
        return NotImplemented


def _into_frame(x: Any, target: Point, tau: Any, /) -> Any:
    """Move ``x`` (a `Point` or a `Tangent`) into ``target.frame`` at ``tau``.

    A `Tangent` is moved with its base point ``target``, expressed in the
    tangent's own frame and chart.
    """
    if frames_equal(x.frame, target.frame) is True:
        return x
    op = cxf.frame_transition(x.frame, target.frame)
    if tau is None and cxfm.is_time_dependent(op):
        msg = (
            f"the transition from {x.frame!r} to {target.frame!r} is "
            "time-dependent: pass the time at which to combine them, `tau=...`."
        )
        raise TypeError(msg)
    if isinstance(x, Point):
        moved = cxfm.act(op, tau, x)
    else:
        back = cxf.frame_transition(target.frame, x.frame)
        at = cast("Point", cxfm.act(back, tau, target)).cconvert(x.chart)
        moved = cxfm.act(op, tau, x, at=at)
    return replace(moved, frame=target.frame)


# ===================================================================
# Constructors


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Point, /) -> Point:
    """Construct a point from another point.

    >>> import coordinax as cx
    >>> vec1 = cx.Point.from_([1, 2, 3], "m")
    >>> vec2 = cx.Point.from_(vec1)
    >>> print(vec2)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    """
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj  # fast path for same type
    return cls.from_(obj.data, obj.chart, obj.M)


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point], obj: Any, chart: cxc.AbstractChart, rep: cxr.Representation, /
) -> Point:
    """Construct a vector from an object, and chart and rep info.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Point.from_(xs, cx.cart3d, cx.point)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> xs = u.Q(jnp.array([[1, 2, 3], [4, 5, 6]]), "m")
    >>> vec = cx.Point.from_(xs, cx.cart3d, cx.point)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    if rep != cxr.point:
        raise ValueError(f"Point construction needs point rep, got {rep}.")
    data = cast("CDict", cxc.cdict(obj, chart))
    point = cls(data=data, chart=chart)
    check_dimensions(point.data, chart, cxr.loc.coord_dimensions(chart))
    return point


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Any, chart: cxc.AbstractChart, /) -> Point:
    """Construct a point from an object, and chart info.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Point.from_(xs, cx.cart3d)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> xs = {"x": u.Q([1, 2], "m"), "y": u.Q([3, 4], "m"), "z": u.Q([5, 6], "m")}
    >>> vec = cx.Point.from_(xs, cx.cart3d)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [[1 3 5]
         [2 4 6]]>

    >>> xs = u.Q(jnp.array([[1, 2, 3], [4, 5, 6]]), "m")
    >>> vec = cx.Point.from_(xs, cx.cart3d)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    return cls.from_(obj, chart, cxr.point)  # ty: ignore[invalid-return-type]


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Any, rep: cxr.Representation, /) -> Point:
    """Construct a point from an object, and rep info.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Point.from_(xs, cx.point)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> xs = {"x": u.Q([1, 2], "m"), "y": u.Q([3, 4], "m"), "z": u.Q([5, 6], "m")}
    >>> vec = cx.Point.from_(xs, cx.point)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [[1 3 5]
         [2 4 6]]>

    >>> xs = u.Q(jnp.array([[1, 2, 3], [4, 5, 6]]), "m")
    >>> vec = cx.Point.from_(xs, cx.point)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    data = cast("CDict", cxc.cdict(obj))
    return cls.from_(data, cxc.guess_chart(data), rep)  # ty: ignore[invalid-return-type]


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Any, /) -> Any:
    """Construct a point from an object.

    Note that this is a pretty limited constructor since it often lacks the
    necessary information to do a proper construction.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Point.from_(xs)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> vec = cx.Point.from_(u.Q([1, 2, 3], "m"))
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    """
    # Infer the chart from the object
    chart = cxc.guess_chart(obj)
    return cls.from_(obj, chart, cxr.point)


# -------------------------------------
# Array-like


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point], obj: ArrayLike | list[Any], unit: u.AbstractUnit | str, /
) -> Any:
    """Construct a cartesian vector from an array and unit.

    The ``ArrayLike[Any, (*#batch, N), "..."]`` is expected to have the
    components as the last dimension.

    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.Point.from_([1, 2, 3], "meter")
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.Point.from_(xs, "meter")
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    return cls.from_(u.Q(obj, u.unit(unit)))  # re-dispatch


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,
    /,
) -> Any:
    """Construct a vector from an array, unit, and chart.

    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.Point.from_([1, 2, 3], "m", cx.cart3d)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.Point.from_(xs, "m", cx.cart3d)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    return cls.from_(u.Q(obj, u.unit(unit)), chart)  # re-dispatch


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
) -> Any:
    """Construct a vector from an array, unit, chart, and rep.

    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.Point.from_([1, 2, 3], "m", cx.cart3d, cx.point)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.Point.from_(xs, "m", cx.cart3d, cx.point)
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    return cls.from_(u.Q(obj, u.unit(unit)), chart, rep)  # re-dispatch


# -------------------------------------
# Frame-aware constructors
#
# Every constructor above also accepts a trailing `AbstractReferenceFrame`.
# The frame is not part of the construction itself — it is attached to the
# result — so each overload below builds the point from the leading arguments
# and then rebinds the frame.


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Point, frame: cxf.AbstractReferenceFrame, /) -> Point:
    """Construct a point from another point, replacing its frame.

    Every constructor above also accepts a trailing frame, which is attached to
    the constructed point:

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.frames as cxf
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> d = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
    >>> for args in [(cx.Point.from_(d),), (d,), (d, cxc.cart3d),
    ...              (d, cxc.cart3d, cxr.point), ([1, 2, 3], "km")]:
    ...     print(cx.Point.from_(*args, cxf.alice).frame)
    Alice()
    Alice()
    Alice()
    Alice()
    Alice()

    An existing frame is replaced, not merged:

    >>> p_alice = cx.Point.from_(d, cxf.alice)
    >>> cx.Point.from_(p_alice, cxf.noframe).frame == cxf.noframe
    True

    """
    return replace(obj, frame=frame)


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Any, frame: cxf.AbstractReferenceFrame, /) -> Point:
    """Construct a point from any object, with a frame."""
    return replace(cls.from_(obj), frame=frame)


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: Any,
    chart: cxc.AbstractChart,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Point:
    """Construct a point from an object and chart, with a frame."""
    return replace(cls.from_(obj, chart), frame=frame)


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: Any,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Point:
    """Construct a point from an object, chart, and representation, with a frame."""
    return replace(cls.from_(obj, chart, rep), frame=frame)


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Point:
    """Construct a point from an array and unit, with a frame."""
    return replace(cls.from_(obj, unit), frame=frame)
