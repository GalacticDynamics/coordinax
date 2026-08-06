"""Point."""

__all__ = ("Point",)


from dataclasses import replace

from jaxtyping import ArrayLike
from typing import TYPE_CHECKING, Any, Generic, cast, final, override
from typing_extensions import TypeVar

import equinox as eqx
import quax_blocks

import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
from .base import AbstractVector, broadcast_and_index_data
from .custom_types import CKey, HasShape
from .mixins import AstropyRepresentationAPIMixin

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
    cls: type[Point],
    obj: Any,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
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
    return cls(data=data, chart=chart)


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
    data = cast("CDict", cxc.cdict(obj, chart))
    return cls(data, chart=chart)


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
    chart = cxc.guess_chart(data)
    return cls(data, chart=chart)


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
    # Infer the data from the chart and object
    data = cast("CDict", cxc.cdict(obj, chart))

    return cls(data, chart=chart)


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
