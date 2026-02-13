"""Vector."""

__all__ = ("Vector",)


from collections.abc import Mapping
from jaxtyping import Array, ArrayLike
from typing import Any, Generic, NoReturn, final
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

import coordinax.api as cxapi
import coordinax.charts as cxc
import coordinax.embeddings as cxe
import coordinax.roles as cxr
import coordinax_api as cxapi
from coordinax._src.constants import ACCELERATION, LENGTH, SPEED
from coordinax._src.custom_types import ComponentsKey, HasShape
from coordinax.objs._src.base import AbstractVectorLike
from coordinax.objs._src.mixins import AstropyRepresentationAPIMixin

ChartT = TypeVar(
    "ChartT", bound=cxc.AbstractChart[Any, Any], default=cxc.AbstractChart[Any, Any]
)
RoleT = TypeVar("RoleT", bound=cxr.AbstractRole, default=cxr.AbstractRole)
V = TypeVar("V", bound=HasShape, default=u.Q)


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

    Mathematical background:

    Let $M$ be a manifold and let $(U,\varphi)$ be a chart with
    coordinate map $\varphi: U \to \mathbb{R}^n$. Coordinax distinguishes:

    **Point** (role = ``Point`` / instance ``cxr.point``)
        A point $p \in M$ represented by its chart coordinates
        $q = \varphi(p)$. A point transforms by coordinate change:
        $q' = (\varphi' \circ \varphi^{-1})(q)$.

        In Euclidean charts, point coordinates may have *heterogeneous physical
        dimensions* (e.g. spherical $(r,\theta,\phi)$ mixes length and
        angle). This is expected.

    **Physical tangent vectors** (roles = ``PhysDisp``, ``PhysVel``, ``PhysAcc``)
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
        **Vel** (physical velocity), and **PhysAcc** (physical acceleration). The
        difference is their units: length, length/time, and length/time^2.

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
        A role flag instance (e.g. `cxr.point`, `cxr.phys_disp`, `cxr.phys_vel`,
        `cxr.phys_acc`) that selects the correct transformation semantics.

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
    ...               chart=cxc.cart3d, role=cxr.phys_disp)
    >>> p = cart  # base point (must have Point role)
    >>> d_sph = d.vconvert(cxc.sph3d, p)
    >>> d_sph.role
    PhysDisp()

    Add a displacement to a point (Euclidean case):

    >>> p2 = cart.add(d)
    >>> p2.role
    Point()

    Construct a velocity vector:

    >>> v = cx.Vector(
    ...     {"x": u.Q(1, "m/s"), "y": u.Q(2, "m/s"), "z": u.Q(3, "m/s")},
    ...     chart=cxc.cart3d,
    ...     role=cxr.phys_vel,
    ... )
    >>> v.role
    PhysVel()

    Notes
    -----
    Notes on units and array shape:

    - A `Vector` does **not** require that all components share one unit. This
      is essential for charts like spherical coordinates where point components
      naturally mix dimensions.
    - For physical tangent roles (``pos/vel/acc``), components are expected to
      be mutually compatible with a single physical dimension (length, speed,
      acceleration), even if expressed in different *units* (e.g. m vs km).
    - Batching is represented by broadcasting the component leaves; the
      conceptual shape of the `Vector` is `broadcast_shapes(*(v.shape for v in
      data.values()))`.

    Core operations:

    - Indexing: ``vec["x"]`` returns a component leaf.
    - Conversion: ``vec.vconvert(target_chart, at=...)`` converts the vector to
      `target_chart`. For ``Point`` this is a coordinate transform. For physical
      tangent roles it is a tangent basis transform evaluated at ``at``.
    - Addition: use ``vec.add(other, at=...)`` for role-aware addition. In
      Euclidean charts, ``Point + Pos -> Point`` and ``Pos + Pos -> Pos`` are
      supported; ``Point + Point`` is not.

    """

    data: dict[ComponentsKey, V]
    """The data for each component."""

    chart: ChartT = eqx.field(static=True)
    """The chart of the vector, e.g. `cxc.cart3d`."""

    role: RoleT = eqx.field(static=True)
    """The role, e.g. `cxr.point`, `cxr.phys_disp`, `cxr.phys_vel`, `cxr.phys_acc`."""

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
        kwargs.setdefault("include_params", False)

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
        if isinstance(self.chart, cxe.EmbeddedManifold):
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
            metric = cxapi.metric_of(self.chart)
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

        Mathematical definition:

        Vector addition follows affine geometry rules:

        - **PhysDisp + PhysDisp → PhysDisp**:
          Adding two tangent vectors yields another tangent vector.

        - **Point + PhysDisp → Point**:
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

            - ``PhysDisp.add(PhysDisp)`` → ``PhysDisp``
            - ``Point.add(PhysDisp)`` → ``Point``

        Raises
        ------
        TypeError
            If the role combination is not supported (e.g., ``Point + Point``).
        ValueError
            If ``at`` is required but not provided.

        Examples
        --------
        >>> import coordinax as cx
        >>> import unxt as u

        Adding two Pos (position-difference) vectors:

        >>> d1 = cx.Vector(
        ...     {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.phys_disp,
        ... )
        >>> d2 = cx.Vector(
        ...     {"x": u.Q(0.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.phys_disp,
        ... )
        >>> result = d1.add(d2)
        >>> result.role
        PhysDisp()

        Adding a Pos to a Point:

        >>> point = cx.Vector(
        ...     {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.point,
        ... )
        >>> result = point.add(d1)
        >>> result.role
        Point()

        """
        # Local import to avoid circular import with `register_quax`.
        from .register_quax import add as _add  # noqa: PLC0415

        return _add(self.role, other.role, self, other, at=at)

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
        anchored container (e.g. {class}`coordinax.PointedVector`).


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

            - ``Point.sub(Point)`` → ``PhysDisp`` (affine difference)
            - ``Pos.sub(Pos)`` → ``PhysDisp`` (vector subtraction)
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
        PhysDisp()

        Subtracting a Pos from a Point:

        >>> disp = cx.Vector(
        ...     {"x": u.Q(0.5, "m"), "y": u.Q(0.5, "m"), "z": u.Q(0.0, "m")},
        ...     cxc.cart3d, cxr.phys_disp,
        ... )
        >>> result = p1.sub(disp)
        >>> result.role
        Point()

        """
        # Local import to avoid circular import with `register_quax`.
        from .register_quax import sub as _sub  # noqa: PLC0415

        return _sub(self.role, other.role, self, other, at=at)

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
        The role flag (Point, PhysDisp, Vel, PhysAcc, etc.)
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
    if isinstance(role, cxr.PhysDisp):
        if dimension != LENGTH:
            msg = f"Pos role requires dimension=length, got dimension={dimension}"
            raise ValueError(msg)
        return
    if isinstance(role, cxr.PhysVel):
        if dimension != SPEED:  # length/time
            msg = (
                f"Vel role requires dimension=speed (length/time), "
                f"got dimension={dimension}"
            )
            raise ValueError(msg)
        return
    if isinstance(role, cxr.PhysAcc):
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
def from_(cls: type[Vector], obj: Vector, /) -> Any:
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
) -> Any:
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs, cxc.cart3d, cxr.point)
    >>> print(vec)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    >>> xs = {"x": u.Q([1, 2], "m"), "y": u.Q([3, 4], "m"), "z": u.Q([5, 6], "m")}
    >>> vec = cx.Vector.from_(xs, cxc.cart3d, cxr.phys_disp)
    >>> print(vec)
    <Vector: chart=Cart3D, role=PhysDisp (x, y, z) [m]
        [[1 3 5]
         [2 4 6]]>

    """
    return cls(data=obj, chart=chart, role=role)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Mapping, chart: cxc.AbstractChart, /) -> Any:  # type: ignore[type-arg]
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
    role = cxr.guess_role(dim)
    # Re-dispatch to the full constructor
    return cls(obj, chart, role)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Mapping, /) -> Any:  # type: ignore[type-arg]
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
    role = cxr.guess_role(dim)

    # Infer the representation from the keys
    chart = cxapi.guess_chart(obj)

    # Re-dispatch to the full constructor
    return cls(obj, chart, role)


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: u.AbstractQuantity,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    role: cxr.AbstractRole,
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
    role
        The role flag (Point, PhysDisp, Vel, PhysAcc)

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
    data = cxapi.cdict(obj, chart)

    # Construct the vector
    return cls(data, chart, role)


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: u.AbstractQuantity,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    /,
) -> Any:
    """Construct a vector from a quantity and chart, inferring role from dimension.

    The role is inferred from the physical dimension:
    - length → Point (affine point)
    - length/time → PhysVel (velocity)
    - length/time² → PhysAcc (acceleration)
    - other → error (role cannot be inferred)
    """
    # Map the components
    obj = jnp.atleast_1d(obj)
    data = cxapi.cdict(obj, chart)

    # Infer role from physical dimension
    dim = u.dimension_of(obj)
    try:
        role = cxr.guess_role(dim)
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
def from_(cls: type[Vector], obj: u.AbstractQuantity, /) -> Any:
    """Construct a Cartesian vector from a quantity, inferring role from dimension.

    The role is inferred from the physical dimension:
    - length → Point (affine point)
    - length/time → PhysVel (velocity)
    - length/time² → PhysAcc (acceleration)
    - other → error (role cannot be inferred)

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Length quantity → Point role:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> print(vec.role)
    Point()

    Velocity quantity → Velocity role:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m/s"))
    >>> print(vec.role)
    PhysVel()

    To override role, use Vector.from_(q, chart, role):

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m/s"), cxc.cart3d, cxr.point)
    >>> print(vec.role)
    Point()

    """
    obj = jnp.atleast_1d(obj)
    chart = cxapi.guess_chart(obj)
    return cls.from_(obj, chart)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: u.AbstractQuantity, role: cxr.AbstractRole, /) -> Any:
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
        The role flag (Point, PhysDisp, Vel, PhysAcc)

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

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"), cxr.phys_disp)
    >>> print(vec.role)
    PhysDisp()

    Construct a PhysVel from a velocity quantity:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m/s"), cxr.phys_vel)
    >>> print(vec.role)
    PhysVel()

    Attempting to construct a Pos from a velocity quantity raises an error:

    >>> try: cx.Vector.from_(u.Q([1, 2, 3], "m/s"), cxr.phys_disp)
    ... except Exception as e: print(e)
    Pos role requires dimension=length, got dimension=speed/velocity

    """
    obj = jnp.atleast_1d(obj)
    chart = cxapi.guess_chart(obj)
    return cls.from_(obj, chart, role)


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
    role: cxr.AbstractRole,
    /,
) -> Any:
    """Construct a cartesian vector from an array and unit."""
    return cls.from_(u.Q(obj, u.unit(unit)), chart, role)  # re-dispatch
