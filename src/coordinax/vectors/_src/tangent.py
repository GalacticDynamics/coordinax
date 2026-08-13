"""Tangent."""

__all__ = ("Tangent",)


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
from .point import _frame_converter

if TYPE_CHECKING:
    from coordinax._src.custom_types import CDict

ChartT = TypeVar(
    "ChartT",
    bound=cxc.AbstractChart[Any, Any, Any],
    default=cxc.AbstractChart[Any, Any, Any],
)
BasisT = TypeVar(
    "BasisT", bound=cxr.AbstractLinearBasis, default=cxr.AbstractLinearBasis
)
SemanticT = TypeVar(
    "SemanticT",
    bound=cxr.AbstractTangentSemanticKind,
    default=cxr.AbstractTangentSemanticKind,
)
V = TypeVar("V", bound=HasShape, default=u.Q)


@final
class Tangent(
    AstropyRepresentationAPIMixin,
    quax_blocks.NumpyInvertMixin[Any],
    quax_blocks.LaxLenMixin,
    AbstractVector[ChartT, cxr.TangentGeometry, BasisT, SemanticT, V],
    Generic[ChartT, BasisT, SemanticT, V],
):
    r"""A tangent-geometry vector with explicit basis and semantic kind.

    A `Tangent` stores four pieces of information:

    - **data**: a mapping from component name to scalar-like value (typically
      `unxt.Quantity`),
    - **chart**: a chart object describing the coordinate system and component
      schema,
    - **basis**: an `~coordinax.representations.AbstractLinearBasis` specifying
      the basis in which tangent components are expressed
      (e.g. `~coordinax.representations.CoordinateBasis` or
      `~coordinax.representations.PhysicalBasis`), and
    - **semantic**: an `~coordinax.representations.AbstractTangentSemanticKind`
      giving the physical interpretation of the tangent vector
      (e.g. `~coordinax.representations.Velocity`,
      `~coordinax.representations.Displacement`).

    The **representation** is computed from these, always with
    `~coordinax.representations.TangentGeometry` as the geometry kind:

    .. math::

        \mathrm{rep} = (
            \mathrm{TangentGeometry},\, \mathrm{basis},\, \mathrm{semantic}
        ).

    This is contrast to `~coordinax.vectors.Point`, which stores a fixed
    `~coordinax.representations.PointGeometry`-flavoured rep and a concrete
    location on the manifold.

    Parameters
    ----------
    data
        Mapping from chart component name to scalar value.
    chart
        A chart instance (e.g. `cxc.cart3d`) that defines the coordinate
        system.
    basis
        The linear basis in which the tangent components are expressed.
    semantic
        The semantic kind of the tangent vector (velocity, displacement, etc.).
    frame
        The reference frame. Defaults to ``cxf.noframe``.

    Examples
    --------
    Construct a **coordinate-basis velocity** in Cartesian 3D:

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> v = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_basis, cxr.vel,
    ... )
    >>> v.rep == cxr.coord_vel
    True

    """

    data: dict[CKey, V]
    """The data for each component."""

    chart: ChartT = eqx.field(static=True)
    """The chart of the vector, e.g. `cxc.cart3d`."""

    basis: BasisT = eqx.field(static=True)
    """The linear basis for tangent components."""

    semantic: SemanticT = eqx.field(static=True)
    """The semantic kind of the tangent vector."""

    frame: cxf.AbstractReferenceFrame = eqx.field(
        default=cxf.noframe, converter=_frame_converter
    )
    """The reference frame. Defaults to ``cxf.noframe``."""

    def __check_init__(self) -> None:
        self.M.check_chart(self.chart)
        self.chart.check_data(self.data, keys=True)

    @property
    def rep(self) -> cxr.Representation:
        """The representation, computed from basis and semantic."""
        return cxr.Representation(cxr.tangent_geom, self.basis, self.semantic)

    @override
    def __getitem__(self, key: Any) -> "V | Tangent":  # ty: ignore[invalid-method-override]
        if isinstance(key, str):
            return self.data[key]
        data = broadcast_and_index_data(self.data, self.shape, key)
        return replace(self, data=data)  # ty: ignore[invalid-return-type]


# ===================================================================
# Constructors


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Tangent], obj: Tangent, /) -> Tangent:
    """Construct a Tangent from another Tangent (identity / fast path).

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> v = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_basis, cxr.vel,
    ... )
    >>> v2 = cx.Tangent.from_(v)
    >>> v2 is v
    True

    """
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj  # fast path for same type
    return cls.from_(obj.data, obj.chart, obj.basis, obj.semantic)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: Any,
    chart: cxc.AbstractChart,
    basis: cxr.AbstractLinearBasis,
    semantic: cxr.AbstractTangentSemanticKind,
    /,
) -> Tangent:
    """Construct a Tangent from data, chart, basis, and semantic.

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> v = cx.Tangent.from_(d, cxc.cart3d, cxr.coord_basis, cxr.vel)
    >>> v.chart
    Cart3D(M=Rn(3))

    """
    data = cast("CDict", cxc.cdict(obj, chart))
    return Tangent(data=data, chart=chart, basis=basis, semantic=semantic)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent], obj: Any, chart: cxc.AbstractChart, rep: cxr.Representation, /
) -> Tangent:
    """Construct a Tangent from data, chart, and a tangent Representation.

    Extracts ``basis`` and ``semantic`` from the representation. Raises
    ``TypeError`` if the representation's geometry kind is not
    ``TangentGeometry``.

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> v = cx.Tangent.from_(d, cxc.cart3d, cxr.coord_vel)
    >>> v.basis == cxr.coord_basis
    True
    >>> v.semantic == cxr.vel
    True

    """
    if not isinstance(rep.geom_kind, cxr.TangentGeometry):
        raise TypeError(
            f"Tangent requires a TangentGeometry representation, got {rep.geom_kind!r}."
        )
    if not isinstance(rep.basis, cxr.AbstractLinearBasis):
        raise TypeError(f"Tangent requires an AbstractLinearBasis, got {rep.basis!r}.")
    if not isinstance(rep.semantic_kind, cxr.AbstractTangentSemanticKind):
        raise TypeError(
            f"Tangent requires an AbstractTangentSemanticKind,"
            f" got {rep.semantic_kind!r}."
        )
    return cls.from_(obj, chart, rep.basis, rep.semantic_kind)  # ty: ignore[invalid-return-type]


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Tangent], obj: Any, chart: cxc.AbstractChart, /) -> Tangent:
    """Construct a Tangent from data and chart (rep inferred from data).

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> v = cx.Tangent.from_(d, cxc.cart3d)
    >>> isinstance(v, cx.Tangent)
    True

    """
    data = cast("CDict", cxc.cdict(obj, chart))
    rep = cxr.guess_rep(data, chart)
    return cls.from_(obj, chart, rep)  # ty: ignore[invalid-return-type]


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Tangent], obj: Any, /) -> Tangent:
    """Construct a Tangent from data alone (chart and rep inferred).

    >>> import coordinax as cx
    >>> import unxt as u

    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> v = cx.Tangent.from_(d)
    >>> isinstance(v, cx.Tangent)
    True

    """
    chart = cxc.guess_chart(obj)
    data = cast("CDict", cxc.cdict(obj, chart))
    rep = cxr.guess_rep(data, chart)
    return cls.from_(data, chart, rep)  # ty: ignore[invalid-return-type]


# -----------------------------------------
# Array-like constructors


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent], obj: ArrayLike | list[Any], unit: u.AbstractUnit | str, /
) -> Tangent:
    """Construct a Tangent from an array and unit (chart inferred).

    >>> import coordinax as cx

    >>> v = cx.Tangent.from_([1.0, 2.0, 3.0], "m/s")
    >>> isinstance(v, cx.Tangent)
    True

    """
    return cls.from_(u.Q(obj, u.unit(unit)))  # ty: ignore[invalid-return-type]


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,
    /,
) -> Tangent:
    """Construct a Tangent from an array, unit, and chart.

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc

    >>> v = cx.Tangent.from_([1.0, 2.0, 3.0], "m/s", cxc.cart3d)
    >>> isinstance(v, cx.Tangent)
    True

    """
    return cls.from_(u.Q(obj, u.unit(unit)), chart)  # ty: ignore[invalid-return-type]


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
) -> Tangent:
    """Construct a Tangent from an array, unit, chart, and Representation.

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = cx.Tangent.from_([1.0, 2.0, 3.0], "m/s", cxc.cart3d, cxr.coord_vel)
    >>> v.basis == cxr.coord_basis
    True

    """
    return cls.from_(u.Q(obj, u.unit(unit)), chart, rep)  # ty: ignore[invalid-return-type]


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Tangent:
    """Construct a Tangent from an array, unit, chart, Representation, and frame."""
    return replace(cls.from_(u.Q(obj, u.unit(unit)), chart, rep), frame=frame)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,
    basis: cxr.AbstractLinearBasis,
    semantic: cxr.AbstractTangentSemanticKind,
    /,
) -> Tangent:
    """Construct a Tangent from array, unit, chart, basis, and semantic.

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = cx.Tangent.from_(
    ...     [1.0, 2.0, 3.0], "m/s", cxc.cart3d, cxr.coord_basis, cxr.vel
    ... )
    >>> v.basis == cxr.coord_basis
    True

    """
    return cls.from_(u.Q(obj, u.unit(unit)), chart, basis, semantic)  # ty: ignore[invalid-return-type]


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: u.AbstractQuantity,
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,
    basis: cxr.AbstractLinearBasis,
    semantic: cxr.AbstractTangentSemanticKind,
    /,
) -> Tangent:
    """Construct a Tangent from a Quantity, unit, chart, basis, and semantic.

    The Quantity is converted to the given unit before construction.

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> v = cx.Tangent.from_(
    ...     u.Q([1.0, 2.0, 3.0], "m/s"), "m/s", cxc.cart3d, cxr.coord_basis, cxr.vel
    ... )
    >>> v.basis == cxr.coord_basis
    True

    """
    return cls.from_(  # ty: ignore[invalid-return-type]
        u.uconvert(u.unit(unit), obj), chart, basis, semantic
    )


# -----------------------------------------
# Frame-aware constructors
#
# The frame is not part of the construction itself — it is attached to the
# result — so each overload below builds the tangent from the leading
# arguments and then rebinds the frame.


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent], obj: Tangent, frame: cxf.AbstractReferenceFrame, /
) -> Tangent:
    """Construct a Tangent from another Tangent, replacing its frame.

    Every constructor above also accepts a trailing frame, which is attached to
    the constructed tangent:

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.frames as cxf
    >>> import unxt as u

    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> for args in [(cx.Tangent.from_(d),), (d,), (d, cxc.cart3d),
    ...              (d, cxc.cart3d, cxr.coord_basis, cxr.vel),
    ...              ([1.0, 2.0, 3.0], "m/s"),
    ...              ([1.0, 2.0, 3.0], "m/s", cxc.cart3d, cxr.coord_vel)]:
    ...     print(cx.Tangent.from_(*args, cxf.alice).frame)
    Alice()
    Alice()
    Alice()
    Alice()
    Alice()
    Alice()

    """
    return replace(obj, frame=frame)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent], obj: Any, frame: cxf.AbstractReferenceFrame, /
) -> Tangent:
    """Construct a Tangent from data with a frame (chart and rep inferred)."""
    return replace(cls.from_(obj), frame=frame)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: Any,
    chart: cxc.AbstractChart,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Tangent:
    """Construct a Tangent from data and chart, with a frame."""
    return replace(cls.from_(obj, chart), frame=frame)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: Any,
    chart: cxc.AbstractChart,
    basis: cxr.AbstractLinearBasis,
    semantic: cxr.AbstractTangentSemanticKind,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Tangent:
    """Construct a Tangent from data, chart, basis, and semantic, with a frame."""
    return replace(cls.from_(obj, chart, basis, semantic), frame=frame)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Tangent:
    """Construct a Tangent from an array and unit, with a frame."""
    return replace(cls.from_(obj, unit), frame=frame)
