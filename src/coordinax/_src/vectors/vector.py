"""Vector."""
# mypy: disable-error-code=type-arg

__all__ = ("Vector",)

from enum import Enum

from collections.abc import Mapping
from jaxtyping import Array, ArrayLike, Bool
from typing import Any, Generic, Literal, NoReturn, cast
from typing_extensions import TypeVar, override

import equinox as eqx
import jax
import jax.tree as jtu
import numpy as np
import plum
import quax
import quax_blocks
import wadler_lindig as wl
from astropy.units import PhysicalType as Dimension
from zeroth import zeroth

import dataclassish
import quaxed.numpy as jnp
import unxt as u
import unxt.quantity as uq

import coordinax_api as cxapi
from .base import AbstractVectorLike
from .custom_types import HasShape, Shape
from .mixins import AstropyRepresentationAPIMixin
from coordinax import r
from coordinax._src.representations.base import REPRESENTATION_CLASSES
from coordinax._src.representations.roles import DIM_TO_ROLE_MAP

RepT = TypeVar("RepT", bound=r.AbstractRep, default=r.AbstractRep)
RoleT = TypeVar("RoleT", bound=r.AbstractRoleFlag, default=r.AbstractRoleFlag)
V = TypeVar("V", bound=HasShape, default=u.Q)


class Vector(
    # IPythonReprMixin,
    AstropyRepresentationAPIMixin,
    quax_blocks.NumpyInvertMixin[Any],
    quax_blocks.LaxLenMixin,
    AbstractVectorLike,
    Generic[RepT, RoleT, V],
):
    """A vector."""

    data: Mapping[str, V]
    """The data for each """

    rep: RepT
    """The representation of the vector, e.g. `cx.r.cart3d`."""

    role: RoleT
    """The role, e.g. `cx.r.pos`, `cx.r.vel`, `cx.r.acc`."""

    def _check_init(self) -> None:
        # Pass a check to self.rep.check_data
        self.rep.check_data(self.data)

    @override
    def __getitem__(self, key: str) -> V:  # type: ignore[override]
        return self.data[key]

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, vector_form: bool = False, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig docstring for the vector.

        Parameters
        ----------
        vector_form
            If True, return the vector form of the docstring.
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
                [("data", self.data), ("rep", self.rep), ("role", self.role)], **kwargs
            )
            return wl.bracketed(
                begin=wl.TextDoc(f"{self.__class__.__name__}("),
                docs=docs,
                sep=wl.comma,
                end=wl.TextDoc(")"),
                indent=kwargs.get("indent", 4),
            )

        rep_name = type(self.rep).__name__
        if isinstance(self.rep, r.EmbeddedManifold):
            chart_name = type(self.rep.chart_kind).__name__
            ambient_name = type(self.rep.ambient_kind).__name__
            rep_name = f"{rep_name}({chart_name} -> {ambient_name})"
        role_name = type(self.role).__name__
        metric = r.metric_of(self.rep)
        metric_name = repr(metric)

        comps = self.rep.components
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

        header = (
            f"<{self.__class__.__name__}: rep={rep_name}, role={role_name}, "
            f"metric={metric_name} {comps_doc}"
        )
        if unit_doc:
            header = f"{header} {unit_doc}"
        return wl.TextDoc(header + values_doc)

    # ===============================================================
    # Vector API

    @plum.dispatch
    def vconvert(
        self,
        target: r.AbstractRep,  # type: ignore[type-arg]
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
    # Misc

    def norm(self, *args: "Vector") -> u.AbstractQuantity:
        msg = "TODO"
        raise NotImplementedError(msg)
        # return self.rep.norm(self.data, *args)


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
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 2 3]>

    """
    return cls.from_(obj.data, obj.rep, obj.role)  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector], obj: Mapping, rep: r.AbstractRep, role: r.AbstractRoleFlag, /
) -> Vector:
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs, cx.r.cart3d, cx.r.pos)
    >>> print(vec)
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 2 3]>

    >>> xs = {"x": u.Q([1, 2], "m"), "y": u.Q([3, 4], "m"), "z": u.Q([5, 6], "m")}
    >>> vec = cx.Vector.from_(xs, cx.r.cart3d, cx.r.pos)
    >>> print(vec)
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [[1 3 5]
         [2 4 6]]>

    """
    return cls(data=obj, rep=rep, role=role)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Mapping, rep: r.AbstractRep, /) -> Vector:
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs, cx.r.cart3d)
    >>> print(vec)
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 2 3]>

    >>> xs = {"x": u.Q([1, 2], "m"), "y": u.Q([3, 4], "m"), "z": u.Q([5, 6], "m")}
    >>> vec = cx.Vector.from_(xs, cx.r.cart3d)
    >>> print(vec)
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [[1 3 5]
         [2 4 6]]>

    """
    # Infer the role from the dimensionality
    dim = u.dimension_of(obj[zeroth(obj)])
    role = DIM_TO_ROLE_MAP[dim]()
    # Re-dispatch to the full constructor
    return cls(obj, rep, role)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Mapping, /) -> Vector:
    """Construct a vector from just a mapping.

    Note that this is a pretty limited constructor and can only match
    `coordinax.r.AbstractFixedComponentsRep` representations, since those have
    fixed component names.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs)
    >>> print(vec)
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 2 3]>

    """
    # Infer the role from the dimensionality
    dim = u.dimension_of(obj[zeroth(obj)])
    role = DIM_TO_ROLE_MAP[dim]()

    # Infer the representation from the keys
    obj_keys = set(obj.keys())
    rep = None
    for rep_cls in REPRESENTATION_CLASSES:
        rep_instance = rep_cls()
        if set(rep_instance.components) == obj_keys:
            rep = rep_instance
            break

    if rep is None:
        msg = f"Cannot infer representation from keys {obj_keys}"
        raise ValueError(msg)

    # Re-dispatch to the full constructor
    return cls(obj, rep, role)


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: u.AbstractQuantity,
    rep: r.AbstractRep,
    role: r.AbstractRoleFlag,
    /,
) -> Vector:
    """Construct a vector from a quantity and representation."""
    # Map the components
    obj = jnp.atleast_1d(obj)
    obj = eqx.error_if(  # Check the dimensionality
        obj,
        obj.shape[-1] != rep.dimensionality,
        f"Cannot construct {cls} from {obj.shape[-1]} components.",
    )
    data = {k: obj[..., i] for i, k in enumerate(rep.components)}

    # Construct the vector
    return cls(data, rep, role)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: u.AbstractQuantity, rep: r.AbstractRep, /) -> Vector:
    """Construct a vector from a quantity and representation."""
    # Map the components
    obj = jnp.atleast_1d(obj)
    obj = eqx.error_if(  # Check the dimensionality
        obj,
        obj.shape[-1] != rep.dimensionality,
        f"Cannot construct {cls} from {obj.shape[-1]} components.",
    )
    data = {k: obj[..., i] for i, k in enumerate(rep.components)}

    # Determine the role
    role = DIM_TO_ROLE_MAP[u.dimension_of(obj)]()

    # Construct the vector
    return cls(data, rep, role)


_SHAPE_DIM_MAP = {
    0: r.cart0d,
    1: r.cart1d,
    2: r.cart2d,
    3: r.cart3d,
}


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: u.AbstractQuantity, /) -> Vector:
    """Construct a Cartesian vector from a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Pos 3D:

    >>> vec = cx.Vector.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 2 3]>

    Vel 3D:

    >>> vec = cx.Vector.from_(u.Quantity([1, 2, 3], "m/s"))
    >>> print(vec)
    <Vector: rep=Cart3D, role=Vel, metric=EuclideanMetric(n=3) (x, y, z) [m / s]
        [1 2 3]>

    Acc 3D:

    >>> vec = cx.Vector.from_(u.Quantity([1, 2, 3], "m/s2"))
    >>> print(vec)
    <Vector: rep=Cart3D, role=Acc, metric=EuclideanMetric(n=3) (x, y, z) [m / s2]
        [1 2 3]>

    """
    obj = jnp.atleast_1d(obj)
    dim = u.dimension_of(obj)

    # Determine the representation
    try:
        rep = _SHAPE_DIM_MAP[obj.shape[-1]]
    except KeyError as e:
        msg = (
            f"Cannot construct {cls} from quantity "
            f"with shape {obj.shape} and dimension {dim}."
        )
        raise ValueError(msg) from e

    return cls.from_(obj, rep)


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
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.Vector.from_(xs, "meter")
    >>> print(vec)
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
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
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.Vector.from_(xs, "meter")
    >>> print(vec)
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    return cls.from_(u.Q.from_(obj, unit))  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    rep: r.AbstractRep,
    /,
) -> Vector:
    """Construct a cartesian vector from an array and unit."""
    return cls.from_(u.Q.from_(obj, unit), rep)  # re-dispatch


@Vector.from_.dispatch
def from_(
    cls: type[Vector],
    obj: ArrayLike | list[Any],
    unit: u.AbstractUnit | str,
    rep: r.AbstractRep,
    role: r.AbstractRoleFlag,
    /,
) -> Vector:
    """Construct a cartesian vector from an array and unit."""
    return cls.from_(u.Q.from_(obj, unit), rep, role)  # re-dispatch


##############################################################################
# Vector conversion


@plum.dispatch
def vconvert(role: r.Pos, to_rep: r.AbstractRep, from_vec: Vector, /) -> Vector:
    """Convert a vector from one representation to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 1, 1], "m")
    >>> print(vec)
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cx.vconvert(cx.r.sph3d, vec)
    >>> print(sph_vec)
    <Vector: rep=Spherical3D, role=Pos, metric=EuclideanMetric(n=3) (r[m], theta[rad], phi[rad])
        [1.732 0.955 0.785]>

    """
    from_vec = eqx.error_if(
        from_vec,
        not isinstance(from_vec.role, r.Pos),
        "from_vec is not a position vector and requires a position vector "
        "for the change-of-basis.",
    )
    # Call the `vconvert` function on the data from the vector's kind
    p = cxapi.vconvert(from_vec.role, to_rep, from_vec.rep, from_vec.data)
    # Return a new vector
    return Vector(data=p, rep=to_rep, role=role)


@plum.dispatch
def vconvert(to_rep: r.AbstractRep, from_vec: Vector, /) -> Vector:
    """Convert a vector from one representation to another."""
    # Redispatch to the role-specific version
    return cxapi.vconvert(from_vec.role, to_rep, from_vec)


@plum.dispatch
def vconvert(
    role: r.Vel | r.Acc,
    to_rep: r.AbstractRep,
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

    >>> sph_vvec = cx.vconvert(cx.r.sph3d, vvec, qvec)

    """
    # Checks
    from_dif = eqx.error_if(
        from_dif,
        isinstance(from_dif.role, r.Pos),
        "'from_dif' must be a differential vector",
    )
    from_pos = eqx.error_if(
        from_pos,
        not isinstance(from_pos.role, r.Pos),
        "'from_pos' must be a position vector",
    )

    # Convert the position to the differential's representation
    from_pos = from_pos.vconvert(from_dif.rep)

    # Call the `vconvert` function on the data from the vector's kind
    p = cxapi.vconvert(
        from_dif.role, to_rep, from_dif.rep, from_dif.data, from_pos.data
    )
    # Return a new vector
    return Vector(data=p, rep=to_rep, role=from_dif.role)


@plum.dispatch
def vconvert(to_rep: r.AbstractRep, from_dif: Vector, from_pos: Vector, /) -> Vector:
    """Convert a vector from one representation to another."""
    # Redispatch to the role-specific version
    return cxapi.vconvert(from_dif.role, to_rep, from_dif, from_pos)


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

    >>> print(dataclassish.replace(vec, z=u.Quantity(4, "km")))
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 2 4000]>

    """
    rep = kwargs.pop("rep", obj.rep)
    role = kwargs.pop("role", obj.role)
    return Vector(data=dataclassish.replace(obj.data, **kwargs), rep=rep, role=role)


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
    <Vector: rep=Cart3D, role=Acc, metric=EuclideanMetric(n=3) (x, y, z) [m / s2]
        [1 2 3]>

    >>> print(jnp.broadcast_to(a, (1, 3)))
    <Vector: rep=Cart3D, role=Acc, metric=EuclideanMetric(n=3) (x, y, z) [m / s2]
        [[1 2 3]]>

    """
    c_shape = shape[:-1]
    return Vector(
        jtu.map(lambda v: jnp.broadcast_to(v, c_shape), operand.data),
        rep=operand.rep,
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
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1. 2. 3.]>

    """
    convert_p = quax.quaxify(jax.lax.convert_element_type_p.bind)
    data = jtu.map(lambda v: convert_p(v, **kwargs), operand.data)
    return Vector(data, rep=operand.rep, role=operand.role)


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


@quax.register(jax.lax.add_p)
def add_p_absvecs(lhs: Vector, rhs: Vector, /) -> Vector:
    """Element-wise addition of two vectors."""
    rhs = cast("Vector", rhs.vconvert(lhs.rep))
    data = jtu.map(jnp.add, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, lhs.rep)


@quax.register(jax.lax.sub_p)
def sub_p_absvecs(lhs: Vector, rhs: Vector, /) -> Vector:
    """Element-wise subtraction of two vectors."""
    rhs = cast("Vector", rhs.vconvert(lhs.rep))
    data = jtu.map(jnp.subtract, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, lhs.rep)


@quax.register(jax.lax.mul_p)
def mul_p_absvecs(lhs: int | float | Array, rhs: Vector, /) -> Vector:
    """Element-wise multiplication of a scalar and a vector."""
    data = jtu.map(lambda v: jnp.multiply(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, rhs.rep)


@quax.register(jax.lax.mul_p)
def mul_p_vecs(lhs: Vector, rhs: int | float | Array, /) -> Vector:
    """Element-wise multiplication of a vector and a scalar."""
    data = jtu.map(lambda v: jnp.multiply(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, lhs.rep)


@quax.register(jax.lax.div_p)
def div_p_absvecs(lhs: int | float | Array, rhs: Vector, /) -> Vector:
    """Element-wise division of a scalar by a vector."""
    data = jtu.map(lambda v: jnp.divide(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, rhs.rep)


@quax.register(jax.lax.div_p)
def div_p_vecs(lhs: Vector, rhs: int | float | Array, /) -> Vector:
    """Element-wise division of a vector by a scalar."""
    data = jtu.map(lambda v: jnp.divide(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, lhs.rep)


# ===================================================================
# Unxt


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
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1000. 2000. 3000.]>

    """
    data = {k: u.uconvert(usys[u.dimension_of(v)], v) for k, v in vec.data.items()}
    return Vector(data, rep=vec.rep, role=vec.role)


@plum.dispatch
def uconvert(units: Mapping[Dimension, u.AbstractUnit | str], vec: Vector, /) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.Vector(
    ...     data={"x": u.Quantity(1, "m"), "y": u.Quantity(2, "km")},
    ...     rep=cx.r.cart2d,
    ...     role=cx.r.Pos(),
    ... )
    >>> print(cart.uconvert({u.dimension("length"): "km"}))
    <Vector: rep=Cart2D, role=Pos, metric=EuclideanMetric(n=2) (x, y) [km]
        [0.001 2.]>

    This also works for vectors with different units:

    >>> sph = cx.Vector(
    ...     data={
    ...         "r": u.Quantity(1, "m"),
    ...         "theta": u.Quantity(45, "deg"),
    ...         "phi": u.Quantity(3, "rad"),
    ...     },
    ...     rep=cx.r.sph3d,
    ...     role=cx.r.Pos(),
    ... )
    >>> print(sph.uconvert({u.dimension("length"): "km", u.dimension("angle"): "deg"}))
    <Vector: rep=Spherical3D, role=Pos, metric=EuclideanMetric(n=3) (r[km], theta[deg], phi[deg])
        [0.001 45. 171.887]>

    """
    # # Ensure `units_` is PT -> Unit
    units_ = {u.dimension(k): v for k, v in units.items()}
    data = {k: u.uconvert(units_[u.dimension_of(v)], v) for k, v in vec.data.items()}
    return Vector(data, rep=vec.rep, role=vec.role)


@plum.dispatch
def uconvert(units: Mapping[str, Any], vec: Vector, /) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.Vector(
    ...     data={"x": u.Quantity(1, "m"), "y": u.Quantity(2, "km")},
    ...     rep=cx.r.cart2d,
    ...     role=cx.r.Pos(),
    ... )
    >>> print(cart.uconvert({"x": "km", "y": "m"}))
    <Vector: rep=Cart2D, role=Pos, metric=EuclideanMetric(n=2) (x[km], y[m])
        [0.001 2000.]>

    This also works for converting just some of the components:

    >>> print(cart.uconvert({"x": "km"}))
    <Vector: rep=Cart2D, role=Pos, metric=EuclideanMetric(n=2) (x, y) [km]
        [0.001 2.]>

    This also works for vectors with different units:

    >>> sph = cx.Vector(
    ...     data={
    ...         "r": u.Quantity(1, "m"),
    ...         "theta": u.Quantity(45, "deg"),
    ...         "phi": u.Quantity(3, "rad"),
    ...     },
    ...     rep=cx.r.sph3d,
    ...     role=cx.r.Pos(),
    ... )
    >>> print(sph.uconvert({"r": "km", "theta": "rad"}))
    <Vector: rep=Spherical3D, role=Pos, metric=EuclideanMetric(n=3) (r[km], theta[rad], phi[rad])
        [0.001 0.785 3.]>

    """
    data = {  # (component: unit)
        k: u.uconvert(units.get(k, u.unit_of(v)), v)  # default to original unit
        for k, v in vec.data.items()
    }
    return Vector(data, rep=vec.rep, role=vec.role)


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
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [m]
        [1 2 3]>

    >>> sph = cart.vconvert(cx.r.sph3d)
    >>> print(sph.uconvert(cx.vecs.ToUnitsOptions.consistent))
    <Vector: rep=Spherical3D, role=Pos, metric=EuclideanMetric(n=3) (r[m], theta[rad], phi[rad])
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
    return Vector(data, rep=vec.rep, role=vec.role)


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
    <Vector: rep=Cart3D, role=Pos, metric=EuclideanMetric(n=3) (x, y, z) [kpc]
        [3.241e-20 6.482e-20 9.722e-20]>

    """
    usys = u.unitsystem(usys)
    return uconvert(usys, vec)
