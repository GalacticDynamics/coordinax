"""Vector."""

__all__ = ("Vector",)


from jaxtyping import Array, ArrayLike
from typing import Any, Generic, final
from typing_extensions import TypeVar, override

import equinox as eqx
import jax
import numpy as np
import plum
import quax_blocks
import wadler_lindig as wl  # type: ignore[import-untyped]
from zeroth import zeroth

import quaxed.numpy as jnp
import unxt as u
import unxt.quantity as uq

import coordinax.charts as cxc
import coordinax.representations as cxr
from .base import AbstractVector
from .custom_types import CKey, HasShape
from .mixins import AstropyRepresentationAPIMixin
from coordinax.internal.custom_types import CDict

ChartT = TypeVar(
    "ChartT", bound=cxc.AbstractChart[Any, Any], default=cxc.AbstractChart[Any, Any]
)
GeomT = TypeVar("GeomT", bound=cxr.AbstractGeometry, default=cxr.AbstractGeometry)
BasisT = TypeVar("BasisT", bound=cxr.AbstractBasis, default=cxr.AbstractBasis)
SemanticT = TypeVar(
    "SemanticT", bound=cxr.AbstractSemanticKind, default=cxr.AbstractSemanticKind
)
V = TypeVar("V", bound=HasShape, default=u.Q)


@final
class Vector(
    # IPythonReprMixin,
    AstropyRepresentationAPIMixin,
    quax_blocks.NumpyInvertMixin[Any],
    quax_blocks.LaxLenMixin,
    AbstractVector,
    Generic[ChartT, GeomT, BasisT, SemanticT, V],
):
    r"""A coordinate-carrying geometric vector.

    A `Vector` stores three pieces of information:

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
    rep
        A `coordinax.representations.Representation` instance (e.g. `cxr.point`)
        that selects the correct transformation semantics.

    Examples
    --------
    Construct a **point** in Cartesian 3D and convert to spherical:

    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> cart = cx.Vector({"x": u.Q(1, "m"), "y": u.Q(1, "m"), "z": u.Q(1, "m")},
    ...                  chart=cxc.cart3d, rep=cxr.point)
    >>> sph = cart.vconvert(cxc.sph3d)
    >>> sph["r"]
    Q(1.73205081, 'm')

    Notes
    -----
    Notes on units and array shape:

    - A `Vector` does **not** require that all components share one unit. This
      is essential for charts like spherical coordinates where point components
      naturally mix dimensions.
    - Batching is represented by broadcasting the component leaves; the
      conceptual shape of the `Vector` is `broadcast_shapes(*(v.shape for v in
      data.values()))`.

    Core operations:

    - Indexing: ``vec["x"]`` returns a component leaf.
    - Conversion: ``vec.vconvert(target_chart, at=...)`` converts the vector to
      `target_chart`. For ``Point`` this is a coordinate transform.

    """

    data: dict[CKey, V]
    """The data for each component."""

    chart: ChartT = eqx.field(static=True)
    """The chart of the vector, e.g. `cxc.cart3d`."""

    rep: cxr.Representation[GeomT, BasisT, SemanticT] = eqx.field(static=True)
    """The `coordinax.representations.Representation`, e.g. `cxr.point`."""

    def _check_init(self) -> None:
        # Pass a check to self.chart.check_data
        self.chart.check_data(self.data)

    @override
    def __getitem__(self, key: str) -> V:  # type: ignore[override]
        return self.data[key]

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

        docs = [
            wl.pdoc(self.data, **kw),
            *wl.named_objs([("chart", self.chart), ("rep", self.rep)], **kw),
        ]
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kw.get("indent", 4),
        )

    # ===============================================================
    # Vector API

    @plum.dispatch
    def vconvert(
        self,
        target: cxc.AbstractChart,  # type: ignore[type-arg]
        /,
        *args: Any,
        **kwargs: Any,
    ) -> "AbstractVector":
        """Represent the vector as another type.

        This just forwards to `coordinax.vconvert`.

        Parameters
        ----------
        target : type[`coordinax.AbstractVector`]
            The type to represent the vector as.
        *args, **kwargs
            Extra arguments. These are passed to `coordinax.vconvert` and
            might be used, depending on the dispatched method.

        Examples
        --------
        >>> import coordinax.main as cx

        >>> vec = cx.Vector.from_([1, 2, 3], "m")
        >>> print(vec)
        <Vector: chart=Cart3D, rep=point (x, y, z) [m]
            [1 2 3]>

        >>> print(vec.vconvert(cx.sph3d))
        <Vector: chart=Spherical3D, rep=point (r[m], theta[rad], phi[rad])
            [3.742 0.641 1.107]>

        """
        return cxr.vconvert(target, self, *args, **kwargs)

    # ===============================================================
    # Quax API

    # TODO: generalize to work with FourVector, and Space
    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        fvs = self.data.values()
        shape = (*jnp.broadcast_shapes(*map(jnp.shape, fvs)), len(fvs))
        dtype = jnp.result_type(*map(jnp.dtype, fvs))  # type: ignore[arg-type]
        return jax.core.ShapedArray(shape, dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        shapes = [v.shape for v in self.data.values()]
        return jnp.broadcast_shapes(*shapes)

    # ===============================================================
    # Misc

    def norm(self, *args: "Vector") -> u.AbstractQuantity:
        msg = "TODO"
        raise NotImplementedError(msg)
        # return self.chart.norm(self.data, *args)


# ===================================================================
# Constructors


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Vector, /) -> Vector:
    """Construct a vector from another vector.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> vec1 = cx.Vector.from_([1, 2, 3], "m")
    >>> vec2 = cx.Vector.from_(vec1)
    >>> print(vec2)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 2 3]>

    """
    if type(obj) is cls:
        return obj  # fast path for same type
    return cls.from_(obj.data, obj.chart, obj.rep)  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: CDict,  # type: ignore[type-arg]
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    rep: cxr.Representation,
    /,
) -> Vector:
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs, cxc.cart3d, cxr.point)
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 2 3]>

    """
    return cls(data=obj, chart=chart, rep=rep)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: CDict, chart: cxc.AbstractChart, /) -> Vector:  # type: ignore[type-arg]
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs, cxc.cart3d)
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 2 3]>

    >>> xs = {"x": u.Q([1, 2], "m"), "y": u.Q([3, 4], "m"), "z": u.Q([5, 6], "m")}
    >>> vec = cx.Vector.from_(xs, cxc.cart3d)
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [[1 3 5]
         [2 4 6]]>

    """
    # Infer the role from the physical dimension
    dim = u.dimension_of(obj[zeroth(obj)])
    rep = cxr.guess_rep(dim)
    # Re-dispatch to the full constructor
    return cls(obj, chart, rep)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: CDict, /) -> Any:  # type: ignore[type-arg]
    """Construct a vector from just a mapping.

    Note that this is a pretty limited constructor and can only match
    `coordinax.r.AbstractFixedComponentsChart` representations, since those have
    fixed component names.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs)
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 2 3]>

    """
    # Infer the role from the physical dimension
    dim = u.dimension_of(obj[zeroth(obj)])
    role = cxr.guess_rep(dim)

    # Infer the representation from the keys
    chart = cxc.guess_chart(obj)

    # Re-dispatch to the full constructor
    return cls(obj, chart, role)


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: u.AbstractQuantity,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    rep: cxr.Representation,
    /,
) -> Any:
    """Construct a vector from a quantity, chart, and role.

    Validates that the physical dimension of obj is compatible with the role.

    Parameters
    ----------
    cls
        The Vector class
    obj
        The quantity to construct from
    chart
        The chart (coordinate representation)
    rep
        The representation (transformation semantics)

    Raises
    ------
    ValueError
        If the dimension of obj is incompatible with the representation.

    """
    # Map the components
    obj = jnp.atleast_1d(obj)
    data = cxc.cdict(chart, obj)

    # Construct the vector
    return cls(data, chart, rep)


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: u.AbstractQuantity,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    /,
) -> Any:
    """Construct a vector from a quantity and chart, inferring representation.

    The representation is inferred from the physical dimension:
    - length → Point (affine point)
    - other → error (representation cannot be inferred)

    """
    # Map the components
    obj = jnp.atleast_1d(obj)
    data = cxc.cdict(chart, obj)

    # Infer role from physical dimension
    dim = u.dimension_of(obj)
    try:
        role = cxr.guess_rep(dim)
    except KeyError as e:
        msg = (
            f"Cannot infer Vector representation from quantity with dimension {dim}. "
            "Specify the representation explicitly, e.g. Vector.from_(q, chart, rep) "
            "or Vector.from_(q, rep)."
        )
        raise ValueError(msg) from e

    # Construct the vector
    return cls(data, chart, role)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: u.AbstractQuantity, /) -> Any:
    """Construct a Cartesian vector from a quantity, inferring representation.

    The representation is inferred from the physical dimension:
    - length → Point (affine point)
    - other → error (representation cannot be inferred)

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    Length quantity → Point representation:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 2 3]>

    """
    obj = jnp.atleast_1d(obj)
    chart = cxc.guess_chart(obj)
    return cls.from_(obj, chart)


@Vector.from_.dispatch
def from_(
    cls: type[Vector], obj: u.AbstractQuantity, rep: cxr.Representation, /
) -> Any:
    """Construct a Cartesian vector from a quantity and an explicit representation.

    The chart is inferred from the quantity's shape. The representation must be
    compatible with the quantity's physical dimension.

    Parameters
    ----------
    cls
        The Vector class
    obj
        The quantity to construct from
    rep
        The representation (transformation semantics)

    Raises
    ------
    ValueError
        If the dimension of obj is incompatible with the representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    Construct a Pos from a length quantity:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"), cxr.point)
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 2 3]>

    """
    obj = jnp.atleast_1d(obj)
    chart = cxc.guess_chart(obj)
    return cls.from_(obj, chart, rep)


@Vector.from_.dispatch
def from_(
    cls: type[Vector], obj: ArrayLike | list[Any], unit: u.AbstractUnit | str, /
) -> Any:
    """Construct a cartesian vector from an array and unit.

    The ``ArrayLike[Any, (*#batch, N), "..."]`` is expected to have the
    components as the last dimension.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.main as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "meter")
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.Vector.from_(xs, "meter")
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    return cls.from_(u.Q(obj, u.unit(unit)))  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector], obj: ArrayLike | list[Any], unit: u.AbstractUnit | str, /
) -> Any:
    """Construct a cartesian vector from an array and unit.

    The ``ArrayLike[Any, (*#batch, N), "..."]`` is expected to have the
    components as the last dimension.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.main as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "meter")
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.Vector.from_(xs, "meter")
    >>> print(vec)
    <Vector: chart=Cart3D, rep=point (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    return cls.from_(u.Q(obj, u.unit(unit)))  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    /,
) -> Any:
    """Construct a cartesian vector from an array and unit."""
    return cls.from_(u.Q(obj, u.unit(unit)), chart)  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    rep: cxr.Representation,
    /,
) -> Any:
    """Construct a cartesian vector from an array and unit."""
    return cls.from_(u.Q(obj, u.unit(unit)), chart, rep)  # re-dispatch


# ===============================================


def vectorform_pdoc(vector: Vector[Any, Any, Any], **kwargs: Any) -> wl.AbstractDoc:
    """Return the Wadler-Lindig docstring for the vector.

    Parameters
    ----------
    vector
        The vector to generate the docstring for.
    short_arrays
        If True, use short arrays for the docstring.
    **kwargs
        Additional keyword arguments to pass to the Wadler-Lindig docstring
        formatter.

    """
    chart_name = type(vector.chart).__name__
    # if isinstance(vector.chart, cxe.EmbeddedChart):
    #     chart_name = type(vector.chart.intrinsic).__name__
    #     ambient_name = type(vector.chart.ambient).__name__
    #     chart_name = f"{chart_name}({chart_name} -> {ambient_name})"
    kwargs.setdefault("canonical", True)
    rep_name = wl.pformat(vector.rep, **kwargs)

    comps = vector.chart.components
    unit_vals: list[u.AbstractUnit | None] = []
    for comp in comps:
        v = vector.data[comp]
        unit_vals.append(u.unit_of(v) if uq.is_any_quantity(v) else None)

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

    vals: list[Array] = [
        (u.ustrip(u.unit_of(v), v) if uq.is_any_quantity(v) else jnp.asarray(v))
        for comp in comps
        for v in (vector.data[comp],)
    ]

    # Branch on whether we have any component leaves to display. If present, we
    # broadcast + stack them and pretty-print the resulting array; otherwise we
    # emit only the closing `>` to keep the header valid for empty vectors.
    if vals:
        stacked = jnp.stack(jnp.broadcast_arrays(*vals), axis=-1)
        val_str = np.array2string(
            np.asarray(stacked),
            precision=kwargs.get("precision", 3),
            threshold=kwargs.get("threshold", 1000),
        )
        val_str = val_str.replace("\n", "\n    ")
        values_doc = f"\n    {val_str}>"
    else:
        values_doc = ">"

    # Build header
    header_parts = [
        f"<{vector.__class__.__name__}: chart={chart_name}",
        f"rep={rep_name}",
    ]

    header = ", ".join(header_parts) + f" {comps_doc}"
    if unit_doc:
        header = f"{header} {unit_doc}"

    return wl.TextDoc(header + values_doc)
