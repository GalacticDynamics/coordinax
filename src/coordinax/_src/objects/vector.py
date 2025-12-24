"""Vector."""

__all__ = ("Vector",)

from enum import Enum

from collections.abc import Callable, Mapping
from jaxtyping import Array, ArrayLike, Bool
from typing import Any, Generic, Literal, NoReturn, cast, final
from typing_extensions import TypeVar, override

import equinox as eqx
import jax
import jax.tree as jtu
import numpy as np
import plum
import quax
import quax_blocks
import wadler_lindig as wl
from zeroth import zeroth

import dataclassish
import quaxed.numpy as jnp
import unxt as u
import unxt.quantity as uq

import coordinax._src.operators as cxo
import coordinax.charts as cxc
import coordinax.embeddings as cxe
import coordinax.roles as cxr
import coordinax_api as cxapi
from .base import AbstractVectorLike
from .custom_types import HasShape
from .mixins import AstropyRepresentationAPIMixin
from coordinax._src import api
from coordinax._src.charts.utils import guess_chart
from coordinax._src.custom_types import Shape
from coordinax._src.roles import DIM_TO_ROLE_MAP

ChartT = TypeVar(
    "ChartT", bound=cxc.AbstractChart[Any, Any], default=cxc.AbstractChart[Any, Any]
)
RoleT = TypeVar("RoleT", bound=cxr.AbstractRole, default=cxr.AbstractRole)
V = TypeVar("V", bound=HasShape, default=u.Q)

LENGTH = u.dimension("length")
SPEED = u.dimension("speed")
ACCELERATION = u.dimension("acceleration")


@final
class Vector(
    # IPythonReprMixin,
    AstropyRepresentationAPIMixin,
    quax_blocks.NumpyInvertMixin[Any],
    quax_blocks.LaxLenMixin,
    AbstractVectorLike,
    Generic[ChartT, RoleT, V],
):
    r"""A coordinate-carrying geometric vector.

    A `Vector` stores three pieces of information:

    - **data**: a mapping from component name to scalar-like value (typically
      `unxt.Quantity`),
    - **chart**: a chart object describing the coordinate system and component
      schema, and
    - **role**: a role flag describing the *geometric meaning* of the components
      and therefore the correct transformation law.

    The design goal is to make the **public API simple** (construct, convert,
    index) while keeping the **mathematics correct** and the numerical kernels
    JAX-friendly (operate on scalar leaves; rely on `jit`/`vmap`).

    Parameters
    ----------
    data
        Mapping from chart component name to scalar value. Each leaf may be a
        `unxt.Quantity` (recommended) or an array-like. Components are expected
        to be *scalar leaves*; batching happens via broadcasting of these leaves.
    chart
        A chart instance (e.g. `cxc.cart3d`, `cxc.sph3d`) that defines component
        names and per-component physical dimensions.
    role
        A role flag instance (e.g. `cxr.point`, `cxr.pos`, `cxr.vel`,
        `cxr.acc`) that selects the correct transformation semantics.

    Mathematical background
    -----------------------
    Let $M$ be a manifold and let $(U,\varphi)$ be a chart with
    coordinate map $\varphi: U \to \mathbb{R}^n$. Coordinax distinguishes:

    **Point** (role = ``Point`` / instance ``cxr.point``)
        A point $p \in M$ represented by its chart coordinates
        $q = \varphi(p)$. A point transforms by coordinate change:
        $q' = (\varphi' \circ \varphi^{-1})(q)$.

        In Euclidean charts, point coordinates may have *heterogeneous physical
        dimensions* (e.g. spherical $(r,\theta,\phi)$ mixes length and
        angle). This is expected.

    **Physical tangent vectors** (roles = ``Pos``, ``Vel``, ``Acc``)
        A physical tangent vector $v \in T_p M$ represented by *physical
        components* in an orthonormal basis attached to the chart at the base
        point $p$. These roles require a base point ``at=`` to transform.

        If $B_R(p)$ is the matrix whose columns are the orthonormal basis
        vectors of chart $R$ expressed in ambient Cartesian components at
        $p$, then physical components transform by a basis change:
        $v_{\mathrm{cart}} = B_R(p)\, v_R$ and
        $v_S = B_S(p)^\mathsf{T} v_{\mathrm{cart}}$, hence

        $$
        $$
            v_S = \bigl(B_S(p)^\mathsf{T} B_R(p)\bigr)\, v_R.

        This is the rule used for **Pos** (physical displacement),
        **Vel** (physical velocity), and **Acc** (physical acceleration). The
        difference is their units: length, length/time, and length/time^2.

    Notes on units and array shape
    ------------------------------
    - A `Vector` does **not** require that all components share one unit. This
      is essential for charts like spherical coordinates where point components
      naturally mix dimensions.
    - For physical tangent roles (``pos/vel/acc``), components are expected to
      be mutually compatible with a single physical dimension (length, speed,
      acceleration), even if expressed in different *units* (e.g. m vs km).
    - Batching is represented by broadcasting the component leaves; the
      conceptual shape of the `Vector` is `broadcast_shapes(*(v.shape for v in
      data.values()))`.

    Core operations
    ---------------
    - Indexing: ``vec["x"]`` returns a component leaf.
    - Conversion: ``vec.vconvert(target_chart, at=...)`` converts the vector to
      `target_chart`. For ``Point`` this is a coordinate transform. For physical
      tangent roles it is a tangent basis transform evaluated at ``at``.
    - Addition: use ``vec.add(other, at=...)`` for role-aware addition. In
      Euclidean charts, ``Point + Pos -> Point`` and ``Pos + Pos -> Pos`` are
      supported; ``Point + Point`` is not.

    Examples
    --------
    Construct a **point** in Cartesian 3D and convert to spherical:

    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import unxt as u
    >>> cart = cx.Vector({"x": u.Q(1, "m"), "y": u.Q(1, "m"), "z": u.Q(1, "m")},
    ...                  chart=cxc.cart3d, role=cxr.point)
    >>> sph = cart.vconvert(cxc.sph3d)
    >>> sph["r"]
    Quantity(..., unit='m')

    Construct a **physical displacement** (Pos) and convert it at a base point:

    >>> d = cx.Vector({"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")},
    ...               chart=cxc.cart3d, role=cxr.pos)
    >>> p = cart  # base point (must have Point role)
    >>> d_sph = d.vconvert(cxc.sph3d, at=p)
    >>> d_sph.role
    <...Pos object at ...>

    Add a displacement to a point (Euclidean case):

    >>> p2 = cart.add(d)
    >>> p2.role
    <...Point object at ...>

    Convenience construction via multiple-dispatch:

    >>> v = cx.Vector.from_([1, 2, 3], "m/s")  # infers cart3d and Vel role
    >>> v.role
    <...Vel object at ...>

    """

    data: Mapping[str, V]
    """The data for each """

    chart: ChartT
    """The chart of the vector, e.g. `cxc.cart3d`."""

    role: RoleT
    """The role, e.g. `cxr.point`, `cxr.pos`, `cxr.vel`, `cxr.acc`."""

    def _check_init(self) -> None:
        # Pass a check to self.chart.check_data
        self.chart.check_data(self.data)

    @override
    def __getitem__(self, key: str) -> V:  # type: ignore[override]
        return self.data[key]

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(  # noqa: C901
        self, *, vector_form: bool = False, show_metric: bool = False, **kwargs: Any
    ) -> wl.AbstractDoc:
        """Return the Wadler-Lindig docstring for the vector.

        Parameters
        ----------
        vector_form
            If True, return the vector form of the docstring.
        show_metric
            If True, include metric information in the vector form.
            Default is False.
        short_arrays
            If True, use short arrays for the docstring.
        **kwargs
            Additional keyword arguments to pass to the Wadler-Lindig docstring
            formatter.

        """
        # Prefer to use short names (e.g. Quantity -> Q) and compact unit forms
        kwargs.setdefault("use_short_name", True)
        kwargs.setdefault("named_unit", False)

        if not vector_form:
            docs = wl.named_objs(
                [("data", self.data), ("chart", self.chart), ("role", self.role)],
                **kwargs,
            )
            return wl.bracketed(
                begin=wl.TextDoc(f"{self.__class__.__name__}("),
                docs=docs,
                sep=wl.comma,
                end=wl.TextDoc(")"),
                indent=kwargs.get("indent", 4),
            )

        rep_name = type(self.chart).__name__
        if isinstance(self.chart, cxc.EmbeddedManifold):
            chart_name = type(self.chart.intrinsic_chart).__name__
            ambient_name = type(self.chart.ambient_chart).__name__
            rep_name = f"{rep_name}({chart_name} -> {ambient_name})"
        role_name = type(self.role).__name__

        comps = self.chart.components
        unit_vals: list[u.AbstractUnit | None] = []
        for comp in comps:
            v = self.data[comp]
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

        vals: list[Array] = []
        for comp in comps:
            v = self.data[comp]
            if uq.is_any_quantity(v):
                unit = u.unit_of(v)
                vals.append(u.ustrip(unit, v))
            else:
                vals.append(jnp.asarray(v))

        if vals:
            fvals = jnp.broadcast_arrays(*vals)
            stacked = jnp.stack(fvals, axis=-1)
            precision = kwargs.get("precision", 3)
            threshold = kwargs.get("threshold", 1000)
            val_str = np.array2string(
                np.asarray(stacked),
                precision=precision,
                threshold=threshold,
            )
            val_str = val_str.replace("\n", "\n    ")
            values_doc = f"\n    {val_str}>"
        else:
            values_doc = ">"

        # Build header
        header_parts = [
            f"<{self.__class__.__name__}: chart={rep_name}",
            f"role={role_name}",
        ]
        if show_metric:
            metric = api.metric_of(self.chart)
            metric_name = repr(metric)
            header_parts.append(f"metric={metric_name}")

        header = ", ".join(header_parts) + f" {comps_doc}"
        if unit_doc:
            header = f"{header} {unit_doc}"
        return wl.TextDoc(header + values_doc)

    # ===============================================================
    # Vector API

    @plum.dispatch
    def vconvert(
        self,
        target: cxc.AbstractChart,  # type: ignore[type-arg]
        /,
        *args: Any,
        **kwargs: Any,
    ) -> "AbstractVectorLike":
        """Represent the vector as another type.

        This just forwards to `coordinax.vconvert`.

        Parameters
        ----------
        target : type[`coordinax.AbstractVectorLike`]
            The type to represent the vector as.
        *args, **kwargs
            Extra arguments. These are passed to `coordinax.vconvert` and
            might be used, depending on the dispatched method.

        """
        # Support 'at=' keyword for tangent-like roles by forwarding it
        # positionally to the underlying multiple-dispatch function.
        at = kwargs.pop("at", None)
        if at is not None:
            return cxapi.vconvert(self.role, target, self, at, *args, **kwargs)
        return cxapi.vconvert(self.role, target, self, *args, **kwargs)

    # ===============================================================
    # Quax API

    # TODO: generalize to work with FourVector, and Space
    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        fvs = self.data.values()
        shape = (*jnp.broadcast_shapes(*map(jnp.shape, fvs)), len(fvs))
        dtype = jnp.result_type(*map(jnp.dtype, fvs))  # type: ignore[arg-type]
        return jax.core.ShapedArray(shape, dtype)

    def materialise(self) -> NoReturn:
        """Materialise the vector for `quax`.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.Vector.from_([1, 2, 3], "m")

        >>> try: vec.materialise()
        ... except RuntimeError as e: print(e)
        Refusing to materialise `Vector`.

        """
        msg = f"Refusing to materialise `{type(self).__name__}`."
        raise RuntimeError(msg)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        shapes = [v.shape for v in self.data.values()]
        return jnp.broadcast_shapes(*shapes)

    # ===============================================================
    # Vector arithmetic

    def add(self, other: "Vector", /, *, at: "Vector | None" = None) -> "Vector":
        r"""Add another vector with role-aware semantics.

        Mathematical definition
        -----------------------
        Vector addition follows affine geometry rules:

        - **Displacement + Displacement → Displacement**:
          Adding two tangent vectors yields another tangent vector.

        - **Position + Displacement → Position**:
          Translating a point by a displacement yields a new point.

        Parameters
        ----------
        other
            Vector to add.
        at
            Base point for non-Euclidean manifolds. Required when the
            representation is non-Euclidean (e.g., on a sphere). The addition
            is performed in the tangent space at this point and then mapped
            back to the manifold.

        Returns
        -------
        Vector
            The sum with appropriate role:

            - ``Displacement.add(Displacement)`` → ``Displacement``
            - ``Pos.add(Displacement)`` → ``Pos``

        Raises
        ------
        TypeError
            If the role combination is not supported (e.g., ``Pos + Pos``).
        ValueError
            If ``at`` is required but not provided.

        Examples
        --------
        >>> import coordinax as cx
        >>> import unxt as u

        Adding two Pos (position-difference) vectors:

        >>> d1 = cx.Vector(
        ...     {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.pos,
        ... )
        >>> d2 = cx.Vector(
        ...     {"x": u.Q(0.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.pos,
        ... )
        >>> result = d1.add(d2)
        >>> result.role
        <...Pos object at ...>

        Adding a Pos to a Point:

        >>> point = cx.Vector(
        ...     {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.point,
        ... )
        >>> result = point.add(d1)
        >>> result.role
        <...Point object at ...>

        """
        return add(self.role, other.role, self, other, at=at)

    def sub(self, other: "Vector", /, *, at: "Vector | None" = None) -> "Vector":
        r"""Subtract another vector with role-aware semantics.

        Mathematical definition
        -----------------------
        Vector subtraction follows affine geometry rules:

        - **Point - Point → Pos**:
          Subtracting two affine points gives the displacement (position-difference)
          vector between them.

        - **Pos - Pos → Pos**:
          Subtracting two displacement vectors yields another displacement.

        - **Point - Pos → Point**:
          Moving a point backwards by a displacement yields a new point.

        For physical tangent roles (Pos/Vel/Acc), cross-chart subtraction is
        only defined with an explicit base point; pass `at=` or operate on an
        anchored container (e.g. {class}`coordinax.FiberPoint`).


        Parameters
        ----------
        other
            Vector to subtract.
        at
            Base point for non-Euclidean manifolds. Required when the
            representation is non-Euclidean and roles are tangent vectors.

        Returns
        -------
        Vector
            The difference with appropriate role:

            - ``Point.sub(Point)`` → ``Pos`` (affine difference)
            - ``Pos.sub(Pos)`` → ``Pos`` (vector subtraction)
            - ``Point.sub(Pos)`` → ``Point" (backwards translation)

        Raises
        ------
        TypeError
            If the role combination is not supported (e.g., ``Pos - Point``).
        ValueError
            If ``at`` is required but not provided.

        Examples
        --------
        >>> import coordinax as cx
        >>> import unxt as u

        Subtracting two Points to get a Pos (displacement):

        >>> p1 = cx.Vector(
        ...     {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.point,
        ... )
        >>> p2 = cx.Vector(
        ...     {"x": u.Q(0.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.point,
        ... )
        >>> result = p1.sub(p2)
        >>> result.role
        <...Pos object at ...>

        Subtracting a Pos from a Point:

        >>> disp = cx.Vector(
        ...     {"x": u.Q(0.5, "m"), "y": u.Q(0.5, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.pos,
        ... )
        >>> result = p1.sub(disp)
        >>> result.role
        <...Point object at ...>

        """
        return sub(self.role, other.role, self, other, at=at)

    # ===============================================================
    # Misc

    def norm(self, *args: "Vector") -> u.AbstractQuantity:
        msg = "TODO"
        raise NotImplementedError(msg)
        # return self.chart.norm(self.data, *args)


# ===================================================================
# Validation helpers


def _validate_dimension_for_role(
    role: cxr.AbstractRole, dimension: u.AbstractDimension
) -> None:
    """Validate that a physical dimension is compatible with a role.

    Parameters
    ----------
    role
        The role flag (Point, Pos, Vel, Acc, etc.)
    dimension
        The physical dimension to validate

    Raises
    ------
    ValueError
        If the dimension is incompatible with the role.

    """
    if isinstance(role, cxr.Point):
        # Point allows any dimension
        return
    if isinstance(role, cxr.Pos):
        if dimension != LENGTH:
            msg = f"Pos role requires dimension=length, got dimension={dimension}"
            raise ValueError(msg)
        return
    if isinstance(role, cxr.Vel):
        if dimension != SPEED:  # length/time
            msg = (
                f"Vel role requires dimension=speed (length/time), "
                f"got dimension={dimension}"
            )
            raise ValueError(msg)
        return
    if isinstance(role, cxr.Acc):
        if dimension != ACCELERATION:  # length/time^2
            msg = (
                f"Acc role requires dimension=acceleration (length/time^2), "
                f"got dimension={dimension}"
            )
            raise ValueError(msg)
        return
    # For other roles, no validation (could be extended in future)


# ===================================================================
# Constructors


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Vector, /) -> Vector:
    """Construct a vector from another vector.

    Examples
    --------
    >>> import coordinax as cx
    >>> vec1 = cx.Vector.from_([1, 2, 3], "m")
    >>> vec2 = cx.Vector.from_(vec1)
    >>> print(vec2)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    """
    return cls.from_(obj.data, obj.chart, obj.role)  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: Mapping,  # type: ignore[type-arg]
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    role: cxr.AbstractRole,
    /,
) -> Vector:
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs, cxc.cart3d, cxr.pos)
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    >>> xs = {"x": u.Q([1, 2], "m"), "y": u.Q([3, 4], "m"), "z": u.Q([5, 6], "m")}
    >>> vec = cx.Vector.from_(xs, cxc.cart3d, cxr.pos)
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [[1 3 5]
         [2 4 6]]>

    """
    return cls(data=obj, chart=chart, role=role)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Mapping, chart: cxc.AbstractChart, /) -> Vector:  # type: ignore[type-arg]
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs, cxc.cart3d)
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    >>> xs = {"x": u.Q([1, 2], "m"), "y": u.Q([3, 4], "m"), "z": u.Q([5, 6], "m")}
    >>> vec = cx.Vector.from_(xs, cxc.cart3d)
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [[1 3 5]
         [2 4 6]]>

    """
    # Infer the role from the physical dimension
    dim = u.dimension_of(obj[zeroth(obj)])
    role = DIM_TO_ROLE_MAP[dim]()
    # Re-dispatch to the full constructor
    return cls(obj, chart, role)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Mapping, /) -> Vector:  # type: ignore[type-arg]
    """Construct a vector from just a mapping.

    Note that this is a pretty limited constructor and can only match
    `coordinax.r.AbstractFixedComponentsChart` representations, since those have
    fixed component names.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs)
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    """
    # Infer the role from the physical dimension
    dim = u.dimension_of(obj[zeroth(obj)])
    role = DIM_TO_ROLE_MAP[dim]()

    # Infer the representation from the keys
    chart = guess_chart(obj)

    # Re-dispatch to the full constructor
    return cls(obj, chart, role)


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: u.AbstractQuantity,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    role: cxr.AbstractRole,
    /,
) -> Vector:
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
    role
        The role flag (Point, Pos, Vel, Acc)

    Returns
    -------
    Vector
        The constructed vector with the specified role

    Raises
    ------
    ValueError
        If the dimension of obj is incompatible with the role.

    """
    # Validate dimension compatibility
    obj_dim = u.dimension_of(obj)
    _validate_dimension_for_role(role, obj_dim)

    # Map the components
    obj = jnp.atleast_1d(obj)
    data = api.cdict(obj, chart)

    # Construct the vector
    return cls(data, chart, role)


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: u.AbstractQuantity,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    /,
) -> Vector:
    """Construct a vector from a quantity and chart, inferring role from dimension.

    The role is inferred from the physical dimension:
    - length → Point (affine point)
    - length/time → Vel (velocity)
    - length/time² → Acc (acceleration)
    - other → error (role cannot be inferred)
    """
    # Map the components
    obj = jnp.atleast_1d(obj)
    data = api.cdict(obj, chart)

    # Infer role from physical dimension
    dim = u.dimension_of(obj)
    try:
        role = DIM_TO_ROLE_MAP[dim]()
    except KeyError as e:
        msg = (
            f"Cannot infer Vector role from quantity with dimension {dim}. "
            "Specify the role explicitly, e.g. Vector.from_(q, chart, role) "
            "or Vector.from_(q, role)."
        )
        raise ValueError(msg) from e

    # Construct the vector
    return cls(data, chart, role)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: u.AbstractQuantity, /) -> Vector:
    """Construct a Cartesian vector from a quantity, inferring role from dimension.

    The role is inferred from the physical dimension:
    - length → Point (affine point)
    - length/time → Vel (velocity)
    - length/time² → Acc (acceleration)
    - other → error (role cannot be inferred)

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Length quantity → Point role:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> print(vec.role)
    <...Point object at ...>

    Velocity quantity → Velocity role:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m/s"))
    >>> print(vec.role)
    <...Vel object at ...>

    To override role, use Vector.from_(q, chart, role):

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m/s"), cxc.cart3d, cxr.point)
    >>> print(vec.role)
    <...Vel object at ...>

    """
    obj = jnp.atleast_1d(obj)
    chart = guess_chart(obj)
    return cls.from_(obj, chart)


@Vector.from_.dispatch
def from_(
    cls: type[Vector], obj: u.AbstractQuantity, role: cxr.AbstractRole, /
) -> Vector:
    """Construct a Cartesian vector from a quantity and an explicit role.

    The chart is inferred from the quantity's shape. The role must be compatible
    with the quantity's physical dimension.

    Parameters
    ----------
    cls
        The Vector class
    obj
        The quantity to construct from
    role
        The role flag (Point, Pos, Vel, Acc)

    Returns
    -------
    Vector
        The constructed vector with the specified role and inferred chart

    Raises
    ------
    ValueError
        If the dimension of obj is incompatible with the role.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Construct a Pos from a length quantity:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"), cxr.pos)
    >>> print(vec.role)
    <...Pos object at ...>

    Construct a Vel from a velocity quantity:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m/s"), cxr.vel)
    >>> print(vec.role)
    <...Vel object at ...>

    Attempting to construct a Pos from a velocity quantity raises an error:

    >>> cx.Vector.from_(u.Q([1, 2, 3], "m/s"), cxr.pos)
    Traceback (most recent call last):
        ...
    ValueError: Pos role requires dimension=length, got dimension=speed

    """
    obj = jnp.atleast_1d(obj)
    chart = guess_chart(obj)
    return cls.from_(obj, chart, role)


@Vector.from_.dispatch
def from_(
    cls: type[Vector], obj: ArrayLike | list[Any], unit: u.AbstractUnit | str, /
) -> Vector:
    """Construct a cartesian vector from an array and unit.

    The ``ArrayLike[Any, (*#batch, N), "..."]`` is expected to have the
    components as the last dimension.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "meter")
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.Vector.from_(xs, "meter")
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    return cls.from_(u.Q.from_(obj, unit))  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector], obj: ArrayLike | list[Any], unit: u.AbstractUnit | str, /
) -> Vector:
    """Construct a cartesian vector from an array and unit.

    The ``ArrayLike[Any, (*#batch, N), "..."]`` is expected to have the
    components as the last dimension.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "meter")
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.Vector.from_(xs, "meter")
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    return cls.from_(u.Q.from_(obj, unit))  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    /,
) -> Vector:
    """Construct a cartesian vector from an array and unit."""
    return cls.from_(u.Q.from_(obj, unit), chart)  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    role: cxr.AbstractRole,
    /,
) -> Vector:
    """Construct a cartesian vector from an array and unit."""
    return cls.from_(u.Q.from_(obj, unit), chart, role)  # re-dispatch


##############################################################################
# Vector conversion


@plum.dispatch
def vconvert(
    role: cxr.Point,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_vec: Vector,
    /,
) -> Vector:
    """Convert a vector from one representation to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 1, 1], "m")
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cx.vconvert(cxc.sph3d, vec)
    >>> print(sph_vec)
    <Vector: chart=Spherical3D, role=Point (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    from_vec = eqx.error_if(
        from_vec,
        not isinstance(from_vec.role, cxr.Point),
        "from_vec is not a point vector and requires a point vector "
        "for the change-of-basis.",
    )
    # Call the `vconvert` function on the data from the vector's kind
    p = cxapi.vconvert(from_vec.role, to_chart, from_vec.chart, from_vec.data)
    # Return a new vector
    return Vector(data=p, chart=to_chart, role=role)


@plum.dispatch
def vconvert(
    role: cxr.Pos,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_vec: Vector,
    from_pos: Vector,
    /,
) -> Vector:
    r"""Convert a position-difference (physical displacement) vector.

    From one representation to another.

    Mathematical Definition
    -----------------------
    A **position difference** (Pos role) is a tangent vector v \in T_p M.
    It transforms via the pushforward (Jacobian) at the base point p:

    $$
       v_S = J_{R \to S}(p) \, v_R
    $$
    This is the same transformation rule as velocity and acceleration, but
    Pos has units of length (not length/time).

    Parameters
    ----------
    role : Pos
        The position-difference role flag.
    to_chart : AbstractChart
        Target representation.
    from_vec : Vector
        Position-difference vector to transform.
    from_pos : Vector
        Base point (position) at which to evaluate the transformation.
        Must have `Point` role.

    Returns
    -------
    Vector
        Transformed position-difference in the target representation.

    Notes
    -----
    - Pos transforms via **physical_tangent_transform** (tangent space transformation).
    - Point transforms via **point_transform** (coordinate transformation).
    - This distinction is fundamental in differential geometry.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> # A position-difference and a base point
    >>> disp = cx.Vector(
    ...     {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
    ...     cxc.cart3d,
    ...     cxr.pos,
    ... )
    >>> point = cx.Vector.from_([1, 1, 1], "m")

    >>> # Transform to spherical - requires base point
    >>> sph_disp = cx.vconvert(cxc.sph3d, disp, point)
    >>> sph_disp.role
    <...Pos object at ...>

    """
    from_vec = eqx.error_if(
        from_vec,
        not isinstance(from_vec.role, cxr.Pos),
        "from_vec is not a position-difference vector.",
    )
    from_pos = eqx.error_if(
        from_pos,
        not isinstance(from_pos.role, cxr.Point),
        "'from_pos' must be a point vector",
    )

    # Convert the base point to the displacement's representation
    from_pos = from_pos.vconvert(from_vec.chart)

    # Pos transforms via physical_tangent_transform (tangent space / pushforward)
    # This is the SAME rule as Vel/Acc, just different units
    p = cxapi.vconvert(
        from_vec.role, to_chart, from_vec.chart, from_vec.data, from_pos.data
    )
    # Return a new vector with Pos role preserved
    return Vector(data=p, chart=to_chart, role=role)


@plum.dispatch
def vconvert(to_chart: cxc.AbstractChart, from_vec: Vector, /) -> Vector:  # type: ignore[type-arg]
    """Convert a vector from one representation to another."""
    # Redispatch to the role-specific version
    return cxapi.vconvert(from_vec.role, to_chart, from_vec)


@plum.dispatch
def vconvert(
    role: cxr.Vel | cxr.Acc,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_dif: Vector,
    from_pos: Vector,
    /,
) -> Vector:
    """Convert a vector from one differential to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> qvec = cx.Vector.from_([1, 1, 1], "m")
    >>> vvec = cx.Vector.from_([10, 10, 10], "m/s")

    >>> sph_vvec = cx.vconvert(cxc.sph3d, vvec, qvec)

    """
    # Checks
    from_dif = eqx.error_if(
        from_dif,
        isinstance(from_dif.role, cxr.Point),
        "'from_dif' must be a differential vector",
    )
    from_pos = eqx.error_if(
        from_pos,
        not isinstance(from_pos.role, cxr.Point),
        "'from_pos' must be a point vector",
    )

    # Convert the position to the differential's representation
    from_pos = from_pos.vconvert(from_dif.chart)

    # Call the `vconvert` function on the data from the vector's kind
    p = cxapi.vconvert(
        from_dif.role, to_chart, from_dif.chart, from_dif.data, from_pos.data
    )
    # Return a new vector
    return Vector(data=p, chart=to_chart, role=from_dif.role)


@plum.dispatch
def vconvert(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_dif: Vector,
    from_pos: Vector,
    /,
) -> Vector:
    """Convert a vector from one representation to another."""
    # Redispatch to the role-specific version
    return cxapi.vconvert(from_dif.role, to_chart, from_dif, from_pos)


##############################################################################
# Dataclassish


@plum.dispatch
def replace(obj: Vector, /, **kwargs: Any) -> Vector:
    """Replace fields of a vector.

    Examples
    --------
    >>> import dataclassish
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "m")
    >>> vec
    Vector(...)

    >>> print(dataclassish.replace(vec, z=u.Q(4, "km")))
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 4000]>

    """
    chart = kwargs.pop("chart", obj.chart)
    role = kwargs.pop("role", obj.role)
    return Vector(
        data=dataclassish.chartlace(obj.data, **kwargs), chart=chart, role=role
    )


# ===================================================================
# Primitives


@quax.register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_p_absvec(
    operand: Vector, /, *, shape: Shape, **kwargs: Any
) -> Vector:
    """Broadcast in a dimension.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> q = cx.Vector.from_([1, 2, 3], "m")
    >>> q.x
    Quantity(Array(1, dtype=int32), unit='m')

    >>> jnp.broadcast_to(q, (1, 3)).x
    Quantity(Array([1], dtype=int32), unit='m')

    >>> p = cx.Vector.from_([1, 2, 3], "m/s")
    >>> p.x
    Quantity(Array(1, dtype=int32), unit='m / s')

    >>> jnp.broadcast_to(p, (1, 3)).x
    Quantity(Array([1], dtype=int32), unit='m / s')

    >>> a = cx.Vector.from_([1, 2, 3], "m/s2")
    >>> print(a)
    <Vector: chart=Cart3D, role=Acc (x, y, z) [m / s2]
        [1 2 3]>

    >>> print(jnp.broadcast_to(a, (1, 3)))
    <Vector: chart=Cart3D, role=Acc (x, y, z) [m / s2]
        [[1 2 3]]>

    """
    c_shape = shape[:-1]
    return Vector(
        jtu.map(lambda v: jnp.broadcast_to(v, c_shape), operand.data),
        chart=operand.chart,
        role=operand.role,
    )


@quax.register(jax.lax.convert_element_type_p)
def convert_element_type_p_absvec(operand: Vector, /, **kwargs: Any) -> Vector:
    """Convert the element type of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "m")
    >>> vec.q.dtype
    dtype('int32')

    >>> print(qlax.convert_element_type(vec, float))
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1. 2. 3.]>

    """
    convert_p = quax.quaxify(jax.lax.convert_element_type_p.bind)
    data = jtu.map(lambda v: convert_p(v, **kwargs), operand.data)
    return Vector(data, chart=operand.chart, role=operand.role)


@quax.register(jax.lax.eq_p)
def eq_p_absvecs(lhs: Vector, rhs: Vector, /) -> Bool[Array, "..."]:
    """Element-wise equality of two vectors.

    See `Vector.__eq__` for examples.

    """
    # Map the equality over the leaves, which are Quantities.
    comp_tree = jtu.map(
        jnp.equal,
        jtu.leaves(lhs.data, is_leaf=uq.is_any_quantity),
        jtu.leaves(rhs.data, is_leaf=uq.is_any_quantity),
        is_leaf=uq.is_any_quantity,
    )

    # Reduce the equality over the leaves.
    return jax.tree.reduce(jnp.logical_and, comp_tree)


# ===============================================
# Add


# -------------------------------------------------------------------
# Internal helpers for role-aware arithmetic


def _require_at_point(
    opname: str,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    at: Vector | None,
) -> Vector:
    """Validate and return a base point for non-Euclidean operations."""
    if at is None:
        msg = (
            f"{opname} on non-Euclidean representation {type(chart).__name__} "
            "requires an `at` base-point parameter. "
            f"Use Vector.{opname.lower()}(other, at=base_point)."
        )
        raise ValueError(msg)
    at = eqx.error_if(
        at, not isinstance(at.role, cxr.Point), "`at` must be a Point vector."
    )
    return at  # noqa: RET504


def _point_in_chart(vec: Vector, chart: cxc.AbstractChart) -> Vector:  # type: ignore[type-arg]
    """Express a Point vector in the requested chart."""
    if vec.chart == chart:
        return vec
    return cast("Vector", vec.vconvert(chart))


def _at_in_chart(at: Vector, chart: cxc.AbstractChart) -> Vector:  # type: ignore[type-arg]
    """Express a base Point in the requested chart."""
    return _point_in_chart(at, chart)


def _tangent_in_chart(
    vec: Vector,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    at_in_chart: Vector,
) -> Vector:
    """Express a physical tangent vector in the requested chart at a base point."""
    if vec.chart == chart:
        return vec
    return cast("Vector", vec.vconvert(chart, at_in_chart))


def _leaf_binop(
    op: Callable,  # type: ignore[type-arg]
    a: Mapping[str, Any],
    b: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a binary op to matching leaves in two component dicts."""
    return cast(
        "dict[str, Any]",
        jtu.map(op, a, b, is_leaf=uq.is_any_quantity),
    )


def _embedded_point_tangent_update(
    op: Callable,  # type: ignore[type-arg]
    *,
    point: Vector,
    tangent: Vector,
    at: Vector,
) -> dict[str, Any]:
    """Update embedded-manifold point by tangent vector w/ ambient arithmetic."""
    chart = point.chart
    at_in_chart = _at_in_chart(at, chart)
    tangent_in_chart = _tangent_in_chart(tangent, chart, at_in_chart)

    p_amb = cxe.embed_point(chart, point.data)
    v_amb = cxe.embed_tangent(chart, tangent_in_chart.data, at=at_in_chart.data)
    p_out_amb = jtu.map(op, p_amb, v_amb, is_leaf=uq.is_any_quantity)
    return cxe.project_point(chart, p_out_amb)


def _embedded_tangent_binop(
    op: Callable,  # type: ignore[type-arg]
    *,
    lhs: Vector,
    rhs: Vector,
    at: Vector,
) -> dict[str, Any]:
    """Combine two embedded-manifold tangents via ambient op then project."""
    chart = lhs.chart
    at_in_chart = _at_in_chart(at, chart)
    rhs_in_chart = _tangent_in_chart(rhs, chart, at_in_chart)

    v_lhs_amb = cxe.embed_tangent(chart, lhs.data, at=at_in_chart.data)
    v_rhs_amb = cxe.embed_tangent(chart, rhs_in_chart.data, at=at_in_chart.data)
    v_out_amb = jtu.map(op, v_lhs_amb, v_rhs_amb, is_leaf=uq.is_any_quantity)
    return cxe.project_tangent(chart, v_out_amb, at=at_in_chart.data)


def _embedded_point_point_to_pos(
    op: Callable,  # type: ignore[type-arg]
    *,
    lhs: Vector,
    rhs: Vector,
    at: Vector,
) -> dict[str, Any]:
    """Compute Point∘Point -> Pos.

    On an embedded manifold via ambient chord then tangent projection.

    """
    chart = lhs.chart
    at_in_chart = _at_in_chart(at, chart)
    rhs_in_chart = _point_in_chart(rhs, chart)

    p_lhs_amb = cxe.embed_point(chart, lhs.data)
    p_rhs_amb = cxe.embed_point(chart, rhs_in_chart.data)
    v_amb = jtu.map(op, p_lhs_amb, p_rhs_amb, is_leaf=uq.is_any_quantity)
    return cxe.project_tangent(chart, v_amb, at=at_in_chart.data)


@quax.register(jax.lax.add_p)
def add_p_absvecs(lhs: Vector, rhs: Vector, /) -> Vector:
    r"""Element-wise addition of two vectors with role semantics.

    This primitive implements vector addition following affine geometry:

    - ``Displacement + Displacement`` → ``Displacement``
    - ``Position + Displacement`` → ``Position``
    - ``Position + Position`` → **TypeError**
    - ``Displacement + Position`` → **TypeError**

    For non-Euclidean representations, use `Vector.add(other, at=base_point)`.

    """
    return add(lhs.role, rhs.role, lhs, rhs, at=None)


@quax.register(jax.lax.add_p)
def add_p_vec_qty(lhs: Vector, rhs: u.AbstractQuantity, /) -> Vector:
    r"""Element-wise addition of Vector + Quantity.

    Desugars to Vector + Vector by interpreting the Quantity:

    - If dimension = length: interpret as Pos (physical displacement)
    - Otherwise: interpret as Point (affine point, default)

    Then delegates to Vector + Vector semantics.

    """
    if u.dimension_of(rhs) == LENGTH:
        rhs_vec = Vector.from_(rhs, cxr.pos)
    else:
        rhs_vec = Vector.from_(rhs)
    return add(lhs.role, rhs_vec.role, lhs, rhs_vec, at=None)


@quax.register(jax.lax.add_p)
def add_p_qty_vec(lhs: u.AbstractQuantity, rhs: Vector, /) -> Vector:
    r"""Element-wise addition of Quantity + Vector.

    Desugars to Vector + Vector by interpreting the Quantity:

    - If dimension = length: interpret as Pos (physical displacement)
    - Otherwise: interpret as Point (affine point, default)

    Then delegates to Vector + Vector semantics.

    """
    if u.dimension_of(lhs) == LENGTH:
        lhs_vec = Vector.from_(lhs, cxr.pos)
    else:
        lhs_vec = Vector.from_(lhs)  # defaults to Point
    return add(lhs_vec.role, rhs.role, lhs_vec, rhs, at=None)


@plum.dispatch.abstract
def add(
    role_lhs: cxr.AbstractRole,
    role_rhs: cxr.AbstractRole,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Add two vectors."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
def add(
    role_lhs: cxr.Point,
    role_rhs: cxr.Pos,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Point + Pos -> Point."""
    # Embedded / non-Euclidean requires a base point.
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Addition", lhs.chart, at)

        if not isinstance(lhs.chart, cxe.EmbeddedManifold):
            msg = (
                f"Addition on intrinsic non-Euclidean "
                f"manifold {type(lhs.chart).__name__} "
                "is not yet implemented. Provide an embedding (EmbeddedManifold) "
                "or implement intrinsic exponential-map semantics."
            )
            raise NotImplementedError(msg)

        out = _embedded_point_tangent_update(jnp.add, point=lhs, tangent=rhs, at=at_vec)
        return Vector(out, lhs.chart, cxr.point)

    # Euclidean: same chart fast path.
    if lhs.chart == rhs.chart:
        return Vector(_leaf_binop(jnp.add, lhs.data, rhs.data), lhs.chart, cxr.point)

    # Euclidean: convert rhs tangent to lhs chart at the base point lhs.
    rhs_in_lhs = _tangent_in_chart(rhs, lhs.chart, lhs)
    return Vector(_leaf_binop(jnp.add, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.point)


@plum.dispatch
def add(
    role_lhs: cxr.Pos,
    role_rhs: cxr.Point,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> NoReturn:
    """Pos + Point is not allowed."""
    msg = "Cannot add Pos + Point. Use Point + Pos instead, or convert both to Pos."
    raise TypeError(msg)


@plum.dispatch
def add(
    role_lhs: cxr.Pos,
    role_rhs: cxr.Pos,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Addition", lhs.chart, at)

        if isinstance(lhs.chart, cxe.EmbeddedManifold):
            out = _embedded_tangent_binop(jnp.add, lhs=lhs, rhs=rhs, at=at_vec)
            return Vector(out, lhs.chart, cxr.pos)

        msg = (
            f"Addition on intrinsic non-Euclidean manifold {type(lhs.chart).__name__} "
            "is not yet implemented. Provide an embedding (EmbeddedManifold) "
            "or implement intrinsic parallel transport / exp-map semantics."
        )
        raise NotImplementedError(msg)

    # Euclidean: same chart fast path.
    if lhs.chart == rhs.chart:
        return Vector(_leaf_binop(jnp.add, lhs.data, rhs.data), lhs.chart, cxr.pos)

    # Euclidean cross-chart requires a base point because the physical basis may
    # depend on position.
    at_vec = _require_at_point("Addition", lhs.chart, at)
    at_in_lhs = _at_in_chart(at_vec, lhs.chart)
    rhs_in_lhs = _tangent_in_chart(rhs, lhs.chart, at_in_lhs)
    return Vector(_leaf_binop(jnp.add, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.pos)


@plum.dispatch
def add(
    role_lhs: cxr.Vel,
    role_rhs: cxr.Vel,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Add Vel + Vel -> Vel.

    If both velocities are in the same representation, operates directly.
    Otherwise, converts to Cartesian for the operation. If `at` is provided,
    converts the result back to lhs's representation.
    """
    # Same representation - direct component-wise addition
    if lhs.chart == rhs.chart:
        data = jtu.map(jnp.add, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Different representations - convert to Cartesian
    lhs_cart = (
        lhs.vconvert(lhs.chart.cartesian) if hasattr(lhs.chart, "cartesian") else lhs
    )
    rhs_cart = (
        rhs.vconvert(rhs.chart.cartesian) if hasattr(rhs.chart, "cartesian") else rhs
    )
    lhs_cart = eqx.error_if(
        lhs_cart,
        lhs_cart.chart != rhs_cart.chart,
        "Cannot add vectors in different non-convertible representations.",
    )
    data = jtu.map(jnp.add, lhs_cart.data, rhs_cart.data, is_leaf=uq.is_any_quantity)
    result_cart = Vector(data, lhs_cart.chart, lhs.role)

    # Convert back to original representation if base point provided
    if at is not None and lhs.chart != lhs_cart.chart:
        return cast("Vector", result_cart.vconvert(lhs.chart, at))

    return result_cart


@plum.dispatch
def add(
    role_lhs: cxr.Acc,
    role_rhs: cxr.Acc,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Add Acc + Acc -> Acc.

    If both accelerations are in the same representation, operates directly.
    Otherwise, converts to Cartesian for the operation. If `at` is provided,
    converts the result back to lhs's representation.
    """
    # Same representation - direct component-wise addition
    if lhs.chart == rhs.chart:
        data = jtu.map(jnp.add, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Different representations - convert to Cartesian
    lhs_cart = (
        lhs.vconvert(lhs.chart.cartesian) if hasattr(lhs.chart, "cartesian") else lhs
    )
    rhs_cart = (
        rhs.vconvert(rhs.chart.cartesian) if hasattr(rhs.chart, "cartesian") else rhs
    )
    lhs_cart = eqx.error_if(
        lhs_cart,
        lhs_cart.chart != rhs_cart.chart,
        "Cannot add vectors in different non-convertible representations.",
    )
    data = jtu.map(jnp.add, lhs_cart.data, rhs_cart.data, is_leaf=uq.is_any_quantity)
    result_cart = Vector(data, lhs_cart.chart, lhs.role)

    # Convert back to original representation if base point provided
    if at is not None and lhs.chart != lhs_cart.chart:
        return cast("Vector", result_cart.vconvert(lhs.chart, at))

    return result_cart


# ===============================================
# Sub


# ===============================================
# Sub


@quax.register(jax.lax.sub_p)
def sub_p_absvecs(lhs: Vector, rhs: Vector, /) -> Vector:
    r"""Element-wise subtraction of two vectors with role semantics.

    Normative behavior:
    - If `lhs.chart == rhs.chart`, subtraction is performed component-wise and
      the result role is determined by the role-dispatch table (`sub(...)`).
    - If `lhs.chart != rhs.chart`, this function does **not** pick an implicit
      intermediate chart. Instead, it delegates to `sub(...)`, which will
      either:

      * perform a role-correct conversion (for `Point - Point`, via
        `point_transform` to a common chart), or
      * raise an error indicating that `at=` is required (for tangent-like
        roles or results).

    """
    # Fast path: same chart and leaf-wise subtraction. Role semantics are still
    # enforced by delegating to `sub(...)` when roles are not trivially closed.
    # If roles are the same physical-tangent role, we can subtract leaves
    # directly and keep the role.
    if (
        (lhs.chart == rhs.chart)
        and isinstance(lhs.role, r.AbstractPhysicalRole)
        and type(lhs.role) is type(rhs.role)
    ):
        data = jtu.map(jnp.subtract, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Delegate to the role-aware dispatcher. This ensures:
    # - `Point - Point -> Pos` uses the affine semantics,
    # - cross-chart tangent operations require `at=`,
    # - invalid role combinations raise.
    return sub(lhs.role, rhs.role, lhs, rhs, at=None)


@plum.dispatch.abstract
def sub(
    role_lhs: cxr.AbstractRole,
    role_rhs: cxr.AbstractRole,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Subtract two vectors."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
def sub(
    role_lhs: cxr.Point,
    role_rhs: cxr.Point,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Point - Point -> Pos (affine difference)."""
    # Embedded / non-Euclidean requires a base point for the resulting Pos.
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Subtraction", lhs.chart, at)

        if isinstance(lhs.chart, cxe.EmbeddedManifold):
            out = _embedded_point_point_to_pos(
                jnp.subtract, lhs=lhs, rhs=rhs, at=at_vec
            )
            return Vector(out, lhs.chart, cxr.pos)

        msg = (
            "Subtraction on intrinsic non-Euclidean "
            f"manifold {type(lhs.chart).__name__} "
            "is not yet implemented. Provide an embedding (EmbeddedManifold) "
            "or implement intrinsic log-map / parallel transport semantics."
        )
        raise NotImplementedError(msg)

    # Euclidean: same chart fast path.
    if lhs.chart == rhs.chart:
        return Vector(_leaf_binop(jnp.subtract, lhs.data, rhs.data), lhs.chart, cxr.pos)

    # Euclidean: convert rhs Point to lhs chart then subtract.
    rhs_in_lhs = _point_in_chart(rhs, lhs.chart)
    return Vector(
        _leaf_binop(jnp.subtract, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.pos
    )


@plum.dispatch
def sub(
    role_lhs: cxr.Point,
    role_rhs: cxr.Pos,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Subtraction", lhs.chart, at)

        if isinstance(lhs.chart, cxe.EmbeddedManifold):
            out = _embedded_point_tangent_update(
                jnp.subtract, point=lhs, tangent=rhs, at=at_vec
            )
            return Vector(out, lhs.chart, cxr.point)

        msg = (
            "Subtraction on intrinsic non-Euclidean "
            f"manifold {type(lhs.chart).__name__} is not yet implemented. "
            "Provide an embedding (EmbeddedManifold) or implement "
            "intrinsic exponential/log-map semantics."
        )
        raise NotImplementedError(msg)

    if lhs.chart == rhs.chart:
        return Vector(
            _leaf_binop(jnp.subtract, lhs.data, rhs.data), lhs.chart, cxr.point
        )

    rhs_in_lhs = _tangent_in_chart(rhs, lhs.chart, lhs)
    return Vector(
        _leaf_binop(jnp.subtract, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.point
    )


@plum.dispatch
def sub(
    role_lhs: cxr.Pos,
    role_rhs: cxr.Point,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> NoReturn:
    """Pos - Point is not allowed."""
    msg = "Cannot subtract Point from Pos. Use Point - Point - Pos, or Point - Pos."
    raise TypeError(msg)


@plum.dispatch
def sub(
    role_lhs: cxr.Pos,
    role_rhs: cxr.Pos,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Subtraction", lhs.chart, at)

        if isinstance(lhs.chart, cxe.EmbeddedManifold):
            out = _embedded_tangent_binop(jnp.subtract, lhs=lhs, rhs=rhs, at=at_vec)
            return Vector(out, lhs.chart, cxr.pos)

        msg = (
            f"Subtraction on intrinsic non-Euclidean "
            f"manifold {type(lhs.chart).__name__} is not yet implemented. "
            "Provide an embedding (EmbeddedManifold) or implement "
            "intrinsic parallel transport / log-map semantics."
        )
        raise NotImplementedError(msg)

    if lhs.chart == rhs.chart:
        return Vector(_leaf_binop(jnp.subtract, lhs.data, rhs.data), lhs.chart, cxr.pos)

    at_vec = _require_at_point("Subtraction", lhs.chart, at)
    at_in_lhs = _at_in_chart(at_vec, lhs.chart)
    rhs_in_lhs = _tangent_in_chart(rhs, lhs.chart, at_in_lhs)
    return Vector(
        _leaf_binop(jnp.subtract, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.pos
    )


@plum.dispatch
def sub(
    role_lhs: cxr.Vel,
    role_rhs: cxr.Vel,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Subtract Vel - Vel -> Vel.

    If both velocities are in the same representation, operates directly.
    Otherwise, converts to Cartesian for the operation. If `at` is provided,
    converts the result back to lhs's representation.
    """
    # Same representation - direct component-wise subtraction
    if lhs.chart == rhs.chart:
        data = jtu.map(jnp.subtract, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Different representations - convert to Cartesian
    lhs_cart = (
        lhs.vconvert(lhs.chart.cartesian) if hasattr(lhs.chart, "cartesian") else lhs
    )
    rhs_cart = (
        rhs.vconvert(rhs.chart.cartesian) if hasattr(rhs.chart, "cartesian") else rhs
    )
    lhs_cart = eqx.error_if(
        lhs_cart,
        lhs_cart.chart != rhs_cart.chart,
        "Cannot subtract vectors in different non-convertible representations.",
    )
    data = jtu.map(
        jnp.subtract, lhs_cart.data, rhs_cart.data, is_leaf=uq.is_any_quantity
    )
    result_cart = Vector(data, lhs_cart.chart, lhs.role)

    # Convert back to original representation if base point provided
    if at is not None and lhs.chart != lhs_cart.chart:
        return cast("Vector", result_cart.vconvert(lhs.chart, at))

    return result_cart


@plum.dispatch
def sub(
    role_lhs: cxr.Acc,
    role_rhs: cxr.Acc,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Subtract Acc - Acc -> Acc.

    If both accelerations are in the same representation, operates directly.
    Otherwise, converts to Cartesian for the operation. If `at` is provided,
    converts the result back to lhs's representation.
    """
    # Same representation - direct component-wise subtraction
    if lhs.chart == rhs.chart:
        data = jtu.map(jnp.subtract, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Different representations - convert to Cartesian
    lhs_cart = (
        lhs.vconvert(lhs.chart.cartesian) if hasattr(lhs.chart, "cartesian") else lhs
    )
    rhs_cart = (
        rhs.vconvert(rhs.chart.cartesian) if hasattr(rhs.chart, "cartesian") else rhs
    )
    lhs_cart = eqx.error_if(
        lhs_cart,
        lhs_cart.chart != rhs_cart.chart,
        "Cannot subtract vectors in different non-convertible representations.",
    )
    data = jtu.map(
        jnp.subtract, lhs_cart.data, rhs_cart.data, is_leaf=uq.is_any_quantity
    )
    result_cart = Vector(data, lhs_cart.chart, lhs.role)

    # Convert back to original representation if base point provided
    if at is not None and lhs.chart != lhs_cart.chart:
        return cast("Vector", result_cart.vconvert(lhs.chart, at))

    return result_cart


# ===============================================


@quax.register(jax.lax.mul_p)
def mul_p_absvecs(lhs: int | float | Array, rhs: Vector, /) -> Vector:
    """Element-wise multiplication of a scalar and a vector."""
    data = jtu.map(lambda v: jnp.multiply(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, chart=rhs.chart, role=rhs.role)


@quax.register(jax.lax.mul_p)
def mul_p_vecs(lhs: Vector, rhs: int | float | Array, /) -> Vector:
    """Element-wise multiplication of a vector and a scalar."""
    data = jtu.map(lambda v: jnp.multiply(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, chart=lhs.chart, role=lhs.role)


@quax.register(jax.lax.div_p)
def div_p_absvecs(lhs: int | float | Array, rhs: Vector, /) -> Vector:
    """Element-wise division of a scalar by a vector."""
    data = jtu.map(lambda v: jnp.divide(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, chart=rhs.chart, role=rhs.role)


@quax.register(jax.lax.div_p)
def div_p_vecs(lhs: Vector, rhs: int | float | Array, /) -> Vector:
    """Element-wise division of a vector by a scalar."""
    data = jtu.map(lambda v: jnp.divide(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, chart=lhs.chart, role=lhs.role)


# ===================================================================
# Unxt


@final
class ToUnitsOptions(Enum):
    """Options for the units argument of `Vector.uconvert`."""

    consistent = "consistent"
    """Convert to consistent units."""


@plum.dispatch
def uconvert(usys: u.AbstractUnitSystem, vec: Vector, /) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> usys = u.unitsystem("m", "s", "kg", "rad")

    >>> vec = cx.Vector.from_([1, 2, 3], "km")
    >>> print(u.uconvert(usys, vec))
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1000. 2000. 3000.]>

    """
    data = {k: u.uconvert(usys[u.dimension_of(v)], v) for k, v in vec.data.items()}
    return Vector(data, chart=vec.chart, role=vec.role)


@plum.dispatch
def uconvert(
    units: Mapping[u.AbstractDimension, u.AbstractUnit | str], vec: Vector, /
) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.Vector(
    ...     data={"x": u.Q(1, "m"), "y": u.Q(2, "km")},
    ...     chart=cxc.cart2d,
    ...     role=cxr.Point(),
    ... )
    >>> print(cart.uconvert({u.dimension("length"): "km"}))
    <Vector: chart=Cart2D, role=Point (x, y) [km]
        [0.001 2.]>

    This also works for vectors with different units:

    >>> sph = cx.Vector(
    ...     data={
    ...         "r": u.Q(1, "m"),
    ...         "theta": u.Q(45, "deg"),
    ...         "phi": u.Q(3, "rad"),
    ...     },
    ...     chart=cxc.sph3d,
    ...     role=cxr.Point(),
    ... )
    >>> print(sph.uconvert({u.dimension("length"): "km", u.dimension("angle"): "deg"}))
    <Vector: chart=Spherical3D, role=Point (r[km], theta[deg], phi[deg])
        [0.001 45. 171.887]>

    """
    # # Ensure `units_` is PT -> Unit
    units_ = {u.dimension(k): v for k, v in units.items()}
    data = {k: u.uconvert(units_[u.dimension_of(v)], v) for k, v in vec.data.items()}
    return Vector(data, chart=vec.chart, role=vec.role)


@plum.dispatch
def uconvert(units: Mapping[str, Any], vec: Vector, /) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.Vector(
    ...     data={"x": u.Q(1, "m"), "y": u.Q(2, "km")},
    ...     chart=cxc.cart2d,
    ...     role=cxr.Point(),
    ... )
    >>> print(cart.uconvert({"x": "km", "y": "m"}))
    <Vector: chart=Cart2D, role=Point (x[km], y[m])
        [0.001 2000.]>

    This also works for converting just some of the components:

    >>> print(cart.uconvert({"x": "km"}))
    <Vector: chart=Cart2D, role=Point (x, y) [km]
        [0.001 2.]>

    This also works for vectors with different units:

    >>> sph = cx.Vector(
    ...     data={
    ...         "r": u.Q(1, "m"),
    ...         "theta": u.Q(45, "deg"),
    ...         "phi": u.Q(3, "rad"),
    ...     },
    ...     chart=cxc.sph3d,
    ...     role=cxr.Point(),
    ... )
    >>> print(sph.uconvert({"r": "km", "theta": "rad"}))
    <Vector: chart=Spherical3D, role=Point (r[km], theta[rad], phi[rad])
        [0.001 0.785 3.]>

    """
    data = {  # (component: unit)
        k: u.uconvert(units.get(k, u.unit_of(v)), v)  # default to original unit
        for k, v in vec.data.items()
    }
    return Vector(data, chart=vec.chart, role=vec.role)


@plum.dispatch
def uconvert(flag: Literal[ToUnitsOptions.consistent], vec: Vector, /) -> Vector:
    """Convert the vector to a self-consistent set of units.

    Parameters
    ----------
    flag
        The vector is converted to consistent units by looking for the first
        quantity with each physical type and converting all components to
        the units of that quantity.
    vec
        The vector to convert.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.Vector.from_([1, 2, 3], "m")

    If all you want is to convert to consistent units, you can use
    ``"consistent"``:

    >>> print(cart.uconvert(cx.vecs.ToUnitsOptions.consistent))
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    >>> sph = cart.vconvert(cxc.sph3d)
    >>> print(sph.uconvert(cx.vecs.ToUnitsOptions.consistent))
    <Vector: chart=Spherical3D, role=Point (r[m], theta[rad], phi[rad])
        [3.742 0.641 1.107]>

    """
    dim2unit = {}
    units_ = {}
    for k, v in vec.data.items():
        pt = u.dimension_of(v)
        if pt not in dim2unit:
            dim2unit[pt] = u.unit_of(v)
        units_[k] = dim2unit[pt]

    data = {k: u.uconvert(units_[k], v) for k, v in vec.data.items()}
    return Vector(data, chart=vec.chart, role=vec.role)


@plum.dispatch
def uconvert(usys: str, vec: Vector, /) -> Vector:
    """Convert the vector to the given units system.

    Parameters
    ----------
    usys
        The units system to convert to, as a string.
    vec
        The vector to convert.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> usys = "galactic"
    >>> vector = cx.Vector.from_([1, 2, 3], "m")
    >>> print(u.uconvert(usys, vector))
    <Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
        [3.241e-20 6.482e-20 9.722e-20]>

    """
    usys = u.unitsystem(usys)
    return uconvert(usys, vec)


# ===================================================================
# as_pos


@plum.dispatch
def as_pos(
    pos: Vector,
    origin: None,
    /,
    *,
    at: Vector | None = None,
    chart: cxc.AbstractChart | None = None,  # type: ignore[type-arg]
) -> Vector:
    r"""Convert a position vector to a displacement from the origin.

    Mathematical Definition:
    Given a position $p$ and an origin $o$, the displacement is:

    $$
       \vec{d} = p - o \in T_o M
    $$
    For Euclidean spaces, if ``origin`` is ``None``, the coordinate origin
    is used (equivalent to reinterpreting the position vector as a
    displacement from the origin).

    Parameters
    ----------
    pos
        Position vector to convert. Must have ``Pos`` role.
    origin
        Origin point. If ``None``, uses the coordinate origin (zero vector).
        For embedded manifolds, this parameter is required.
    at
        Base point for tangent space evaluation when converting the resulting
        displacement to a different representation. If ``None`` and ``chart`` is
        requested, defaults to ``origin`` if provided, otherwise ``pos``.
    chart
        Target chart for the displacement vector. If provided, the
        displacement is converted to this chart.
    rep
        Target representation for the resulting displacement. If ``None``,
        returns displacement in ``pos.chart``. Conversion uses tangent/frame
        transformation evaluated at ``at``.

    Returns
    -------
    Vector
        Displacement vector with ``Displacement`` role.

    Raises
    ------
    TypeError
        If ``pos`` does not have ``Pos`` role, or if ``origin`` does not
        have ``Pos`` role when provided.
    NotImplementedError
        For intrinsic manifolds without embedding, or for embedded manifolds
        when proper parallel transport is not yet implemented.

    Notes
    -----
    **Euclidean case:**
        Displacements are computed by converting both points to a shared
        Cartesian representation (using position transform), subtracting
        componentwise, then optionally converting via ``Displacement.vconvert``
        to the requested chart.

    **Embedded manifold case:**
        Points are embedded to ambient Cartesian space, subtracted to get
        an ambient displacement, then optionally projected to the tangent
        space at ``at`` via ``Displacement.vconvert``.

    **Intrinsic manifold case:**
        Not yet implemented. Raises ``NotImplementedError``.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Convert a position to a displacement (from the coordinate origin):

    >>> pos = cx.Vector.from_([1, 2, 3], "m")
    >>> disp = cx.as_pos(pos)
    >>> disp.role
    <...Displacement object at ...>
    >>> disp["x"]
    Quantity(Array(1., dtype=float32), unit='m')

    Convert with an explicit origin:

    >>> origin = cx.Vector.from_([0.5, 0.5, 0.5], "m")
    >>> disp = cx.as_pos(pos, origin)
    >>> disp["x"]
    Quantity(Array(0.5, dtype=float32), unit='m')

    Request a specific representation (uses Displacement.vconvert):

    >>> pos_sph = pos.vconvert(cxc.sph3d)
    >>> disp_sph = cx.as_pos(pos_sph, origin, chart=cxc.cyl3d, at=pos_sph)
    >>> disp_sph.chart
    coordinax...Cylindrical3D

    """
    if not isinstance(pos.role, cxr.Point):
        msg = (
            f"Cannot convert vector with role {type(pos.role).__name__} "
            "to displacement."
        )
        raise TypeError(msg)

    is_embedded = isinstance(pos.chart, cxe.EmbeddedManifold)

    if not pos.chart.is_euclidean and not is_embedded:
        msg = (
            f"as_pos on intrinsic manifold {type(pos.chart).__name__} "
            "requires an embedding (not yet implemented)."
        )
        raise NotImplementedError(msg)

    # Compute displacement in a shared chart
    if is_embedded:
        msg = "Embedded manifolds require an explicit origin for as_pos."
        raise ValueError(msg)

    # Euclidean case: displacement from the Euclidean origin. Convert to
    # Cartesian using position transform, then interpret as displacement
    cart_chart = pos.chart.cartesian
    pos_cart = cast("Vector", pos.vconvert(cart_chart))

    # The displacement is just the position in Cartesian (origin is zero)
    disp_data = pos_cart.data
    disp_chart = cart_chart

    # Create displacement vector in computed chart
    disp_vec = Vector(disp_data, disp_chart, cxr.pos)

    # Convert to requested chart if specified
    if chart is not None and chart != disp_chart:
        # Determine base point for tangent transformation
        if at is None:
            at = origin if origin is not None else pos

        # Use Displacement.vconvert (tangent/frame transformation)
        disp_vec = cast("Vector", disp_vec.vconvert(chart, at))

    return disp_vec


@plum.dispatch
def as_pos(
    pos: Vector,
    origin: Vector,
    /,
    *,
    at: Vector | None = None,
    chart: cxc.AbstractChart | None = None,  # type: ignore[type-arg]
) -> Vector:
    r"""Convert a position vector to a displacement from the origin.

    Mathematical Definition:
    Given a position $p$ and an origin $o$, the displacement is:

    $$
       \vec{d} = p - o \in T_o M
    $$
    For Euclidean spaces, if ``origin`` is ``None``, the coordinate origin
    is used (equivalent to reinterpreting the position vector as a
    displacement from the origin).

    Parameters
    ----------
    pos
        Position vector to convert. Must have ``Pos`` role.
    origin
        Origin point. If ``None``, uses the coordinate origin (zero vector).
        For embedded manifolds, this parameter is required.
    at
        Base point for tangent space evaluation when converting the resulting
        displacement to a different representation. If ``None`` and ``chart`` is
        requested, defaults to ``origin`` if provided, otherwise ``pos``.
    chart
        Target chart for the displacement vector. If provided, the
        displacement is converted to this chart.
    rep
        Target representation for the resulting displacement. If ``None``,
        returns displacement in ``pos.chart``. Conversion uses tangent/frame
        transformation evaluated at ``at``.

    Returns
    -------
    Vector
        Displacement vector with ``Displacement`` role.

    Raises
    ------
    TypeError
        If ``pos`` does not have ``Pos`` role, or if ``origin`` does not
        have ``Pos`` role when provided.
    NotImplementedError
        For intrinsic manifolds without embedding, or for embedded manifolds
        when proper parallel transport is not yet implemented.

    Notes
    -----
    **Euclidean case:**
        Displacements are computed by converting both points to a shared
        Cartesian representation (using position transform), subtracting
        componentwise, then optionally converting via ``Displacement.vconvert``
        to the requested chart.

    **Embedded manifold case:**
        Points are embedded to ambient Cartesian space, subtracted to get
        an ambient displacement, then optionally projected to the tangent
        space at ``at`` via ``Displacement.vconvert``.

    **Intrinsic manifold case:**
        Not yet implemented. Raises ``NotImplementedError``.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Convert a position to a displacement (from the coordinate origin):

    >>> pos = cx.Vector.from_([1, 2, 3], "m")
    >>> disp = cx.as_pos(pos)
    >>> disp.role
    <...Displacement object at ...>
    >>> disp["x"]
    Quantity(Array(1., dtype=float32), unit='m')

    Convert with an explicit origin:

    >>> origin = cx.Vector.from_([0.5, 0.5, 0.5], "m")
    >>> disp = cx.as_pos(pos, origin)
    >>> disp["x"]
    Quantity(Array(0.5, dtype=float32), unit='m')

    Request a specific representation (uses Displacement.vconvert):

    >>> pos_sph = pos.vconvert(cxc.sph3d)
    >>> disp_sph = cx.as_pos(pos_sph, origin, chart=cxc.cyl3d, at=pos_sph)
    >>> disp_sph.chart
    coordinax...Cylindrical3D

    """
    if not isinstance(pos.role, cxr.Point):
        msg = (
            f"Cannot convert vector with role {type(pos.role).__name__} "
            "to displacement."
        )
        raise TypeError(msg)

    is_embedded = isinstance(pos.chart, cxe.EmbeddedManifold)

    if not pos.chart.is_euclidean and not is_embedded:
        msg = (
            f"as_pos on intrinsic manifold {type(pos.chart).__name__} "
            "requires an embedding (not yet implemented)."
        )
        raise NotImplementedError(msg)

    # Compute displacement in a shared chart
    # Validate origin
    if not isinstance(origin.role, cxr.Point):
        msg = f"Origin must have Point role, got {type(origin.role).__name__}."
        raise TypeError(msg)

    if is_embedded:  # compute displacement in ambient space
        # Embed both points to ambient Cartesian
        pos_ambient = cxe.embed_point(pos.chart, pos.data)
        origin_ambient = cxe.embed_point(origin.chart, origin.data)

        # Compute ambient displacement (Euclidean subtraction)
        disp_data = jtu.map(
            jnp.subtract, pos_ambient, origin_ambient, is_leaf=uq.is_any_quantity
        )
        # Get ambient Cartesian chart
        disp_chart = pos.chart.ambient_chart.cartesian
    else:
        # Euclidean case: convert both to shared Cartesian chart, subtract
        cart_chart = pos.chart.cartesian

        # Convert using Pos.vconvert (position transform)
        pos_cart = cast("Vector", pos.vconvert(cart_chart))
        origin_cart = cast("Vector", origin.vconvert(cart_chart))

        # Compute displacement in Cartesian
        disp_data = jtu.map(
            jnp.subtract,
            pos_cart.data,
            origin_cart.data,
            is_leaf=uq.is_any_quantity,
        )
        disp_chart = cart_chart

    # Create displacement vector in computed chart
    disp_vec = Vector(disp_data, disp_chart, cxr.pos)

    # Convert to requested chart if specified
    if chart is not None and chart != disp_chart:
        # Determine base point for tangent transformation
        if at is None:
            at = origin if origin is not None else pos

        # Use Displacement.vconvert (tangent/frame transformation)
        disp_vec = cast("Vector", disp_vec.vconvert(chart, at))

    return disp_vec


# =============================================================================
# Operator dispatch


@plum.dispatch
def apply_op(
    op: cxo.AbstractOperator,
    tau: Any,
    v: Vector,
    /,
    *,
    role: None = None,
    at: Any = None,
) -> Vector:
    """Apply an operator to a Vector.

    The role is inferred from the Vector's role attribute.
    """
    # Role must not be provided - it's inferred
    # (plum dispatch handles this via the `role: None = None` signature)

    # Get the base point data if provided
    at_data = at.data if (at is not None and hasattr(at, "data")) else at

    # Apply to the underlying data with the vector's role
    result_data = api.apply_op(op, tau, v.data, role=v.role, at=at_data)

    # Return a new Vector with the same chart and role
    return Vector(data=result_data, chart=v.chart, role=v.role)


# ===================================================================
# cdict dispatch


@plum.dispatch
def cdict(obj: Vector, /) -> Mapping[str, Any]:
    """Extract component dictionary from a Vector.

    Parameters
    ----------
    obj
        A Vector object

    Returns
    -------
    dict[str, Any]
        The component dictionary from the vector's data field.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> d = cx.cdict(vec)
    >>> list(d.keys())
    ['x', 'y', 'z']

    """
    return obj.data
