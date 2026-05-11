"""Point."""

__all__ = ("Point",)


from dataclasses import replace

from jaxtyping import Array, ArrayLike
from typing import TYPE_CHECKING, Any, Generic, cast, final
from typing_extensions import TypeVar, override

import equinox as eqx
import jax
import numpy as np
import quax_blocks
import wadler_lindig as wl

import dataclassish
import quaxed.numpy as jnp
import unxt as u
import unxt.quantity as uq

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
from .base import AbstractVector
from .custom_types import CKey, HasShape
from .mixins import AstropyRepresentationAPIMixin
from coordinax.internal import pos_named_objs

if TYPE_CHECKING:
    from .custom_types import CDict

ChartT = TypeVar(
    "ChartT", bound=cxc.AbstractChart[Any, Any], default=cxc.AbstractChart[Any, Any]
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

    >>> import coordinax.main as cx
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

    def _check_init(self) -> None:
        # Pass a check to self.chart.check_data
        self.M.has_chart(self.chart)
        self.chart.check_data(self.data, keys=True)

    @property
    def rep(self) -> cxr.Representation[cxr.PointGeometry, cxr.NoBasis, cxr.Location]:
        """The representation of the vector."""
        return cxr.point

    @override
    def __getitem__(self, key: Any) -> "V | Point":  # ty: ignore[invalid-method-override]
        if isinstance(key, str):
            return self.data[key]
        return replace(self, data={k: v[key] for k, v in self.data.items()})  # ty: ignore[invalid-return-type,not-subscriptable]

    # ===============================================================
    # Quax API

    # TODO: generalize to work with FourVector, and Space
    def aval(self) -> jax.core.ShapedArray:  # ty: ignore[possibly-missing-submodule]
        """Return the vector as a JAX array."""
        fvs = self.data.values()
        shape = (*jnp.broadcast_shapes(*map(jnp.shape, fvs)), len(fvs))
        dtype = jnp.result_type(*map(jnp.dtype, fvs))
        return jax.core.ShapedArray(shape, dtype)  # ty: ignore[possibly-missing-submodule]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        shapes = [v.shape for v in self.data.values()]
        return jnp.broadcast_shapes(*shapes)

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, vector_form: bool = False, **kw: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig docstring for the vector.

        Parameters
        ----------
        vector_form
            If True, return the vector form of the docstring.
        short_arrays
            If True, use short arrays for the docstring.
        **kw
            Additional keyword arguments to pass to the Wadler-Lindig docstring
            formatter.

        """
        if vector_form:
            return vectorform_pdoc(self, **kw)

        # Prefer to use short names (e.g. Quantity -> Q) and compact unit forms
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


# ===================================================================
# Constructors


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Point, /) -> Point:
    """Construct a point from another point.

    Examples
    --------
    >>> import coordinax.main as cx
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

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

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
    return cls(data=data, chart=chart)  # ty: ignore[missing-argument]


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Any, chart: cxc.AbstractChart, /) -> Point:
    """Construct a point from an object, and chart info.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

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
    return cls(data, chart=chart)  # ty: ignore[missing-argument]


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Any, rep: cxr.Representation, /) -> Point:
    """Construct a point from an object, and rep info.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

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
    return cls(data, chart=chart)  # ty: ignore[missing-argument]


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Point], obj: Any, /) -> Any:
    """Construct a point from an object.

    Note that this is a pretty limited constructor since it often lacks the
    necessary information to do a proper construction.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

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

    return cls(data, chart=chart)  # ty: ignore[missing-argument]


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
    >>> import coordinax.main as cx

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
    >>> import coordinax.main as cx

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

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.main as cx

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


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: Point,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Point:
    """Construct a point from another point, replacing its frame.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf

    >>> p = cx.Point.from_([1, 0, 0], "km")
    >>> p2 = cx.Point.from_(p, cxf.alice)
    >>> p2.frame
    Alice()

    >>> # Replace an existing frame
    >>> p3 = cx.Point.from_(p2, cxf.noframe)
    >>> p3.frame == cxf.noframe
    True

    """
    return replace(obj, frame=frame)


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: Any,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Point:
    """Construct a point from any object with a frame.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf
    >>> import unxt as u

    >>> p = cx.Point.from_(
    ...     {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")},
    ...     cxf.alice,
    ... )
    >>> p.frame
    Alice()

    """
    p = cls.from_(obj)
    return replace(p, frame=frame)


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: Any,
    chart: cxc.AbstractChart,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Point:
    """Construct a point from an object, chart, and frame.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> d = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
    >>> p = cx.Point.from_(d, cxc.cart3d, cxf.alice)
    >>> p.chart
    Cart3D(M=Rn(3))
    >>> p.frame
    Alice()

    """
    p = cls.from_(obj, chart)
    return replace(p, frame=frame)


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: Any,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Point:
    """Construct a point from an object, chart, representation, and frame.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> d = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
    >>> p = cx.Point.from_(d, cxc.cart3d, cxr.point, cxf.alice)
    >>> p.chart
    Cart3D(M=Rn(3))
    >>> p.rep
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())
    >>> p.frame
    Alice()

    """
    p = cls.from_(obj, chart, rep)
    return replace(p, frame=frame)


@Point.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Point],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    frame: cxf.AbstractReferenceFrame,
    /,
) -> Point:
    """Construct a point from an array, unit, and frame.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf

    >>> p = cx.Point.from_([1, 0, 0], "km", cxf.alice)
    >>> p.frame
    Alice()

    """
    p = cls.from_(obj, unit)
    return replace(p, frame=frame)


# ===============================================


def _vector_comps_unit_docs(vector: AbstractVector) -> tuple[str, str]:
    """Return ``(comps_doc, unit_doc)`` strings for a vector header.

    ``comps_doc`` is the parenthesised component list, e.g. ``(x, y, z)`` or
    ``(x[m], y[m/s], z[m/s])`` when units differ per component.
    ``unit_doc`` is the bracketed shared unit string, e.g. ``[m]``, or an
    empty string when units are absent or differ per component.
    """
    comps = vector.chart.components
    unit_vals = [
        cast("u.AbstractUnit", u.unit_of(v))
        if uq.is_any_quantity(v := vector.data[comp])
        else None
        for comp in comps
    ]

    unit_doc = ""
    if unit_vals and all(u_ is not None for u_ in unit_vals):
        unit0 = unit_vals[0]
        if all(u_ == unit0 for u_ in unit_vals):
            unit_doc = f"[{unit0}]"
            comps_doc = f"({', '.join(comps)})"
        else:
            comps_doc = (
                "("
                + ", ".join(
                    f"{c}[{u_}]" for c, u_ in zip(comps, unit_vals, strict=True)
                )
                + ")"
            )
    elif any(u_ is not None for u_ in unit_vals):
        comps_doc = (
            "("
            + ", ".join(
                f"{c}[{u_}]" if u_ is not None else c
                for c, u_ in zip(comps, unit_vals, strict=True)
            )
            + ")"
        )
    else:
        comps_doc = f"({', '.join(comps)})"

    return comps_doc, unit_doc


def _vector_values_str(vector: AbstractVector, **kwargs: Any) -> str:
    r"""Return the formatted array string ``'\\n    [values]'`` (no closing ``>``)."""
    comps = vector.chart.components
    vals: list[Array] = [
        jnp.asarray(u.ustrip(u.unit_of(v), v) if uq.is_any_quantity(v) else v)
        for comp in comps
        for v in (vector.data[comp],)
    ]

    # If there are no component leaves to display, return an empty string so
    # the caller can append the closing ``>`` directly after the header.
    if not vals:
        return ""

    stacked = jnp.stack(jnp.broadcast_arrays(*vals), axis=-1)
    val_str = np.array2string(
        np.asarray(stacked),
        precision=kwargs.get("precision", 3),
        threshold=kwargs.get("threshold", 1000),
    )
    return f"\n    {val_str.replace(chr(10), chr(10) + '    ')}"


def vectorform_pdoc(
    vector: Point[Any, Any],
    *,
    class_name: str | None = None,
    **kwargs: Any,
) -> wl.AbstractDoc:
    """Return the Wadler-Lindig docstring for the vector.

    Parameters
    ----------
    vector
        The vector to generate the docstring for.
    class_name
        Override the class name used in the header.  Defaults to
        ``type(vector).__name__``.
    **kwargs
        Additional keyword arguments passed to the Wadler-Lindig formatter
        (e.g. ``precision``, ``threshold``, ``canonical``).

    """
    kwargs.setdefault("canonical", True)
    cls_name = class_name if class_name is not None else vector.__class__.__name__
    chart_name = type(vector.chart).__name__
    comps_doc, unit_doc = _vector_comps_unit_docs(vector)
    values_str = _vector_values_str(vector, **kwargs)

    header = f"<{cls_name}: chart={chart_name} {comps_doc}"
    if unit_doc:
        header = f"{header} {unit_doc}"

    return wl.TextDoc(header + values_str + ">")
