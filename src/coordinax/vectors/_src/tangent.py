"""Tangent."""

__all__ = ("Tangent",)


from dataclasses import replace

from jaxtyping import ArrayLike
from typing import TYPE_CHECKING, Any, Generic, cast, final
from typing_extensions import TypeVar, override

import equinox as eqx
import quax_blocks
import wadler_lindig as wl
from jax.core import ShapedArray

import dataclassish
import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
from .base import AbstractVector
from .custom_types import CKey, HasShape
from .mixins import AstropyRepresentationAPIMixin
from .point import _frame_converter, _vector_comps_unit_docs, _vector_values_str
from coordinax.internal import pos_named_objs

if TYPE_CHECKING:
    from coordinax.internal import CDict

ChartT = TypeVar(
    "ChartT", bound=cxc.AbstractChart[Any, Any], default=cxc.AbstractChart[Any, Any]
)
BasisT = TypeVar(
    "BasisT",
    bound=cxr.AbstractLinearBasis,
    default=cxr.AbstractLinearBasis,
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

    >>> import coordinax.main as cx
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
        return replace(self, data={k: v[key] for k, v in self.data.items()})  # ty: ignore[invalid-return-type,not-subscriptable]

    # ===============================================================
    # Quax API

    def aval(self) -> ShapedArray:
        """Return the vector as a JAX abstract array."""
        fvs = self.data.values()
        shape = (*jnp.broadcast_shapes(*map(jnp.shape, fvs)), len(fvs))
        dtype = jnp.result_type(*map(jnp.dtype, fvs))
        return ShapedArray(shape, dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the batch shape of the vector."""
        shapes = [v.shape for v in self.data.values()]
        return jnp.broadcast_shapes(*shapes)

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, vector_form: bool = False, **kw: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig docstring for the vector.

        Parameters
        ----------
        vector_form
            If True, return the compact vector-form representation.
        **kw
            Additional keyword arguments forwarded to the formatter.

        """
        if vector_form:
            return _vectorform_pdoc(self, **kw)

        kw.setdefault("use_short_name", True)
        kw.setdefault("named_unit", False)
        kw.setdefault("include_params", False)
        kw.setdefault("canonical", True)

        docs = pos_named_objs(
            dataclassish.field_items(self), ["data"], self.__dataclass_fields__, **kw
        )
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kw.get("indent", 4),
        )


def _vectorform_pdoc(
    vector: Tangent[Any, Any, Any, Any],
    *,
    class_name: str | None = None,
    **kwargs: Any,
) -> wl.AbstractDoc:
    """Return the compact vector-form docstring for a Tangent."""
    kwargs.setdefault("canonical", True)
    cls_name = class_name if class_name is not None else vector.__class__.__name__
    chart_name = type(vector.chart).__name__
    # Reuse Point's helper (works on any object with .chart and .data)
    comps_doc, unit_doc = _vector_comps_unit_docs(vector)
    values_str = _vector_values_str(vector, **kwargs)

    header = f"<{cls_name}: chart={chart_name} {comps_doc}"
    if unit_doc:
        header = f"{header} {unit_doc}"

    return wl.TextDoc(header + values_str + ">")


# ===================================================================
# Constructors


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Tangent], obj: Tangent, /) -> Tangent:
    """Construct a Tangent from another Tangent (identity / fast path).

    Examples
    --------
    >>> import coordinax.main as cx
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

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> v = cx.Tangent.from_(d, cxc.cart3d, cxr.coord_basis, cxr.vel)
    >>> v.chart
    Cart3D(M=Rn(3))

    """
    data = cast("CDict", cxc.cdict(obj, chart))
    return Tangent(  # ty: ignore[missing-argument]
        data=data, chart=chart, basis=basis, semantic=semantic
    )


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: Any,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
) -> Tangent:
    """Construct a Tangent from data, chart, and a tangent Representation.

    Extracts ``basis`` and ``semantic`` from the representation. Raises
    ``TypeError`` if the representation's geometry kind is not
    ``TangentGeometry``.

    Examples
    --------
    >>> import coordinax.main as cx
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

    Examples
    --------
    >>> import coordinax.main as cx
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

    Examples
    --------
    >>> import coordinax.main as cx
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
    cls: type[Tangent],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    /,
) -> Tangent:
    """Construct a Tangent from an array and unit (chart inferred).

    Examples
    --------
    >>> import coordinax.main as cx

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

    Examples
    --------
    >>> import coordinax.main as cx
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
    basis: cxr.AbstractLinearBasis,
    semantic: cxr.AbstractTangentSemanticKind,
    /,
) -> Tangent:
    """Construct a Tangent from array, unit, chart, basis, and semantic.

    Examples
    --------
    >>> import coordinax.main as cx
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

    Examples
    --------
    >>> import coordinax.main as cx
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


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: Tangent,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Tangent:
    """Construct a Tangent from another Tangent, replacing its frame.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.frames as cxf
    >>> import unxt as u

    >>> v = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_basis, cxr.vel,
    ... )
    >>> v2 = cx.Tangent.from_(v, cxf.alice)
    >>> v2.frame
    Alice()

    """
    return replace(obj, frame=frame)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: Any,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Tangent:
    """Construct a Tangent from data with a frame (chart and rep inferred).

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf
    >>> import unxt as u

    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> v = cx.Tangent.from_(d, cxf.alice)
    >>> v.frame
    Alice()

    """
    v = cls.from_(obj)
    return replace(v, frame=frame)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: Any,
    chart: cxc.AbstractChart,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Tangent:
    """Construct a Tangent from data, chart, and frame (basis/semantic inferred).

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.frames as cxf
    >>> import unxt as u

    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> v = cx.Tangent.from_(d, cxc.cart3d, cxf.alice)
    >>> v.frame
    Alice()

    """
    v = cls.from_(obj, chart)
    return replace(v, frame=frame)


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
    """Construct a Tangent from data, chart, basis, semantic, and frame.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.frames as cxf
    >>> import unxt as u

    >>> d = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> v = cx.Tangent.from_(d, cxc.cart3d, cxr.coord_basis, cxr.vel, cxf.alice)
    >>> v.frame
    Alice()

    """
    v = cls.from_(obj, chart, basis, semantic)
    return replace(v, frame=frame)


@Tangent.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Tangent],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Tangent:
    """Construct a Tangent from an array, unit, and frame.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf

    >>> v = cx.Tangent.from_([1.0, 2.0, 3.0], "m/s", cxf.alice)
    >>> v.frame
    Alice()

    """
    v = cls.from_(obj, unit)
    return replace(v, frame=frame)
