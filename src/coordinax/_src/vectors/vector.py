"""Vector."""
# mypy: disable-error-code=type-arg

__all__ = ("Vector",)

import functools as ft
from enum import Enum

from collections.abc import Mapping
from jaxtyping import Array, ArrayLike, Bool
from typing import Any, Generic, Literal, NoReturn, TypeVar, cast
from typing_extensions import override

import equinox as eqx
import jax
import jax.tree as jtu
import plum
import quax
import quax_blocks
from astropy.units import PhysicalType as Dimension

import dataclassish
import quaxed.numpy as jnp
import unxt as u
import unxt.quantity as uq
from unxt.quantity import BareQuantity, is_any_quantity

import coordinax_api as cxapi
from .base import AbstractVectorLike
from .custom_types import HasShape, PDict, Shape
from .representations import core as r
from coordinax._src.custom_types import Unit

RepT = TypeVar("RepT", bound=r.AbstractRepresentation)
V = TypeVar("V", bound=HasShape)


class Vector(
    # IPythonReprMixin,
    # AstropyRepresentationAPIMixin,
    quax_blocks.NumpyInvertMixin[Any],
    quax_blocks.LaxLenMixin,
    AbstractVectorLike,
    Generic[RepT, V],
):
    """A vector."""

    data: Mapping[str, V]
    """The data for each """

    kind: RepT

    def _check_init(self) -> None:
        # Pass a check to self.kind.check_data
        self.kind.check_data(self.data)

    @override
    def __getitem__(self, key: str) -> V:  # type: ignore[override]
        return self.data[key]

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
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")

        >>> try: vec.materialise()
        ... except RuntimeError as e: print(e)
        Refusing to materialise `CartesianPos3D`.

        """
        msg = f"Refusing to materialise `{type(self).__name__}`."
        raise RuntimeError(msg)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        shapes = [v.shape for v in self.data.values()]
        return jnp.broadcast_shapes(*shapes)

    def norm(self, *args: "Vector") -> u.AbstractQuantity:
        msg = "TODO"
        raise NotImplementedError(msg)
        # return self.kind.norm(self.data, *args)


# ===================================================================
# Constructors


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Mapping, rep: r.AbstractRepresentation, /) -> Vector:
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> vec = cx.Vector.from_(xs, cx.r.cartpos3d)
    >>> print(vec)

    >>> xs = {"x": u.Q([1, 2], "m"), "y": u.Q([3, 4], "m"), "z": u.Q([5, 6], "m")}
    >>> vec = cx.Vector.from_(xs, cx.r.cartpos3d)
    >>> print(vec)

    """
    return cls(obj, rep)


@Vector.from_.dispatch
def from_(
    cls: type[Vector], obj: u.AbstractQuantity, rep: r.AbstractRepresentation, /
) -> Vector:
    """Construct a vector from a quantity and representation."""
    # Ensure the object is at least 1D
    obj = jnp.atleast_1d(obj)

    # Check the dimensions
    if obj.shape[-1] != rep.dimensionality:
        msg = f"Cannot construct {cls} from {obj.shape[-1]} components."
        raise ValueError(msg)

    # Map the components
    comps = {k: obj[..., i] for i, k in enumerate(rep.components)}

    # Construct the vector from the mapping
    return cls.from_(comps, rep)


_SHAPE_DIM_MAP = {
    (1, "length"): r.cartpos1d,
    (1, "speed"): r.cartvel1d,
    (1, "acceleration"): r.cartacc1d,
    (2, "length"): r.cartpos2d,
    (2, "speed"): r.cartvel2d,
    (2, "acceleration"): r.cartacc2d,
    (3, "length"): r.cartpos3d,
    (3, "speed"): r.cartvel3d,
    (3, "acceleration"): r.cartacc3d,
}


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: u.AbstractQuantity, /) -> Vector:
    """Construct a vector from a quantity.

    This will fail for most non-position vectors, except Cartesian vectors,
    since they generally do not have the same dimensions, nor can be converted
    from a Cartesian vector without additional information.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.nvecs as cxv

    Pos 3D:

    >>> vec = cxv.Vector.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <CartesianPos3D: (x, y, z) [m]
        [1 2 3]>

    Vel 3D:

    >>> vec = cxv.Vector.from_(u.Quantity([1, 2, 3], "m/s"))
    >>> print(vec)
    <CartesianVel3D: (x, y, z) [m / s]
        [1 2 3]>

    Acc 3D:

    >>> vec = cxv.Vector.from_(u.Quantity([1, 2, 3], "m/s2"))
    >>> print(vec)
    <CartesianAcc3D: (x, y, z) [m / s2]
        [1 2 3]>

    """
    obj = jnp.atleast_1d(obj)
    dim = u.dimension_of(obj)

    if (obj.shape[-1], dim) not in _SHAPE_DIM_MAP:
        msg = (
            f"Cannot construct {cls} from quantity "
            f"with shape {obj.shape} and dimension {dim}."
        )
        raise ValueError(msg)

    rep = _SHAPE_DIM_MAP[(obj.shape[-1], dim)]

    return cls.from_(obj, rep)


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: ArrayLike | list[Any], unit: Unit | str, /) -> Vector:
    """Construct a vector from an array and unit.

    The ``ArrayLike[Any, (*#batch, N), "..."]`` is expected to have the
    components as the last dimension.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "meter")
    >>> print(vec)
    <CartesianPos3D: (x, y, z) [m]
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.CartesianPos3D.from_(xs, "meter")
    >>> print(vec)
    <CartesianPos3D: (x, y, z) [m]
        [[1 2 3]
         [4 5 6]]>

    """
    obj = u.Quantity.from_(jnp.asarray(obj), unit)
    return cls.from_(obj)  # re-dispatch


@Vector.from_.dispatch
def from_(cls: type[Vector], obj: Vector, /) -> Vector:
    """Construct a vector from another vector.

    Examples
    --------
    >>> import coordinax.nvecs as cxv
    >>> vec1 = cxv.Vector.from_([1, 2, 3], "m")
    >>> vec2 = cxv.Vector.from_(vec1)
    >>> print(vec2)

    """
    return cls.from_(obj.data, obj.kind)  # re-dispatch


# ===================================================================


@plum.dispatch
def vconvert(to_rep: r.AbstractRepresentation, from_vec: Vector, /) -> Vector:
    """Convert a vector from one representation to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.nvecs as cxv

    >>> vec = cxv.Vector.from_([1, 1, 1], "m")
    >>> print(vec)
    <CartesianPos3D: (x, y, z) [m]
        [1 1 1]>

    >>> sph_vec = cxv.vconvert(cxv.r.sphericalpos, vec)
    >>> print(sph_vec)
    <SphericalPos: (r, theta, phi) [m]
        [1.73205081 0.95531662 0.78539816]>

    """
    # Call the `vconvert` function on the data from the vector's kind
    p = vconvert(to_rep, from_vec.kind, from_vec.data)
    # Return a new vector
    return Vector(p, to_rep)


@plum.dispatch.multi(
    (r.CartPos1D, r.CartPos1D, PDict),
    (r.CartVel1D, r.CartVel1D, PDict),
    (r.CartAcc1D, r.CartAcc1D, PDict),
    (r.RadialPos, r.RadialPos, PDict),
    (r.RadialVel, r.RadialVel, PDict),
    (r.RadialAcc, r.RadialAcc, PDict),
    (r.CartPos2D, r.CartPos2D, PDict),
    (r.CartVel2D, r.CartVel2D, PDict),
    (r.CartAcc2D, r.CartAcc2D, PDict),
    (r.PolarPos, r.PolarPos, PDict),
    (r.PolarVel, r.PolarVel, PDict),
    (r.PolarAcc, r.PolarAcc, PDict),
    (r.TwoSpherePos, r.TwoSpherePos, PDict),
    (r.TwoSphereVel, r.TwoSphereVel, PDict),
    (r.TwoSphereAcc, r.TwoSphereAcc, PDict),
    (r.CartPos3D, r.CartPos3D, PDict),
    (r.CartVel3D, r.CartVel3D, PDict),
    (r.CartAcc3D, r.CartAcc3D, PDict),
    (r.SphericalPos, r.SphericalPos, PDict),
    (r.SphericalVel, r.SphericalVel, PDict),
    (r.SphericalAcc, r.SphericalAcc, PDict),
    (r.CylindricalPos, r.CylindricalPos, PDict),
    (r.CylindricalVel, r.CylindricalVel, PDict),
    (r.CylindricalAcc, r.CylindricalAcc, PDict),
    (r.ProlateSpheroidalPos, r.ProlateSpheroidalPos, PDict),
    (r.ProlateSpheroidalVel, r.ProlateSpheroidalVel, PDict),
    (r.ProlateSpheroidalAcc, r.ProlateSpheroidalAcc, PDict),
    (r.CartSpaceTime, r.CartSpaceTime, PDict),
    # (r.SpaceTime, r.SpaceTime, PDict),  # Not true
    # (r.PoincarePolarRep, r.PoincarePolarRep, PDict),  # Not true
    (r.CartPosND, r.CartPosND, PDict),
    (r.CartVelND, r.CartVelND, PDict),
    (r.CartAccND, r.CartAccND, PDict),
)
def vconvert(
    to_rep: r.AbstractRepresentation, from_rep: r.AbstractRepresentation, p: PDict, /
) -> Mapping:
    return p


@plum.dispatch(precedence=-1)
def vconvert(to_rep: r.AbstractPos, from_rep: r.AbstractPos, p: PDict, /) -> Mapping:
    """AbstractPos -> CartesianPos -> AbstractPos."""
    p = vconvert(to_rep.cartesian_type, from_rep, p)
    return vconvert(to_rep, from_rep.cartesian_type, p)


@plum.dispatch
def vconvert(to_rep: r.CartPos1D, from_rep: r.RadialPos, p: PDict, /) -> Mapping:
    """RadialPos -> CartesianPos1D.

    The `r` coordinate is converted to the `x` coordinate of the 1D system.
    """
    return {"x": p["r"]}


@plum.dispatch
def vconvert(to_rep: r.RadialPos, from_rep: r.CartPos1D, p: PDict, /) -> Mapping:
    """CartesianPos1D -> RadialPos.

    The `x` coordinate is converted to the `r` coordinate of the 1D system.
    """
    del to_rep, from_rep  # Unused
    return {"r": p["x"]}


@plum.dispatch
def vconvert(to_rep: r.PolarPos, from_rep: r.CartPos2D, p: PDict, /) -> Mapping:
    """CartesianPos2D -> PolarPos.

    The `x` and `y` coordinates are converted to the `r` and `theta` coordinates
    of the 2D polar system.
    """
    del to_rep, from_rep  # Unused
    r_ = jnp.hypot(p["x"], p["y"])
    theta = jnp.arctan2(p["y"], p["x"])
    return {"r": r_, "theta": theta}


@plum.dispatch
def vconvert(to_rep: r.CartPos2D, from_rep: r.PolarPos, p: PDict, /) -> Mapping:
    """PolarPos -> CartesianPos2D.

    The `r` and `theta` coordinates are converted to the `x` and `y` coordinates
    of the 2D Cartesian system.
    """
    del to_rep, from_rep  # Unused
    x = p["r"] * jnp.cos(p["theta"])
    y = p["r"] * jnp.sin(p["theta"])
    return {"x": x, "y": y}


@plum.dispatch
def vconvert(to_rep: r.CylindricalPos, from_rep: r.CartPos3D, p: PDict, /) -> Mapping:
    """CartesianPos3D -> CylindricalPos."""
    del to_rep, from_rep  # Unused
    rho = jnp.hypot(p["x"], p["y"])
    phi = jnp.atan2(p["y"], p["x"])
    return {"rho": rho, "phi": phi, "z": p["z"]}


@plum.dispatch
def vconvert(to_rep: r.SphericalPos, from_rep: r.CartPos3D, p: PDict, /) -> Mapping:
    """CartesianPos3D -> SphericalPos."""
    del to_rep, from_rep  # Unused
    r_ = jnp.sqrt(p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2)
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r_ == 0, jnp.ones(r_.shape), p["z"] / r_))
    # atan2 handles the case when x = y = 0, returning phi = 0
    phi = jnp.atan2(p["y"], p["x"])
    return {"r": r_, "theta": theta, "phi": phi}


@plum.dispatch
def vconvert(
    to_rep: r.AbstractSphericalPos, from_rep: r.CartPos3D, p: PDict, /
) -> Mapping:
    """CartesianPos3D -> AbstractSphericalPos."""
    p = vconvert(r.SphericalPos, from_rep, p)
    return vconvert(to_rep, r.SphericalPos, p)


@plum.dispatch
def vconvert(to_rep: r.CartPos3D, from_rep: r.CylindricalPos, p: PDict, /) -> Mapping:
    """CylindricalPos -> CartesianPos3D."""
    del to_rep, from_rep  # Unused
    x = p["rho"] * jnp.cos(p["phi"])
    y = p["rho"] * jnp.sin(p["phi"])
    return {"x": x, "y": y, "z": p["z"]}


@plum.dispatch
def vconvert(
    to_rep: r.SphericalPos, from_rep: r.CylindricalPos, p: PDict, /
) -> Mapping:
    """CylindricalPos -> SphericalPos."""
    del to_rep, from_rep  # Unused
    r_ = jnp.hypot(p["rho"], p["z"])
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r_ == 0, jnp.ones(r_.shape), p["z"] / r_))
    return {"r": r_, "theta": theta, "phi": p["phi"]}


@plum.dispatch.multi(
    (type[r.LonLatSphericalPos], type[r.CylindricalPos], Mapping),
    (type[r.MathSphericalPos], type[r.CylindricalPos], Mapping),
)
def vconvert(
    to_rep: r.AbstractSphericalPos, from_rep: r.CylindricalPos, p: PDict, /
) -> Mapping:
    """CylindricalPos -> SphericalPos -> AbstractSphericalPos."""
    p = vconvert(r.SphericalPos, from_rep, p)
    return vconvert(to_rep, r.SphericalPos, p)


@plum.dispatch
def vconvert(to_rep: r.CartPos3D, from_rep: r.SphericalPos, p: PDict, /) -> Mapping:
    """SphericalPos -> CartesianPos3D."""
    del to_rep, from_rep  # Unused
    r, theta, phi = p["r"], p["theta"], p["phi"]
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return {"x": x, "y": y, "z": z}


@plum.dispatch
def vconvert(
    to_rep: r.CylindricalPos, from_rep: r.SphericalPos, p: PDict, /
) -> Mapping:
    """SphericalPos -> CylindricalPos."""
    del to_rep, from_rep  # Unused
    rho = p["r"] * jnp.sin(p["theta"])
    z = p["r"] * jnp.cos(p["theta"])
    return {"rho": rho, "phi": p["phi"], "z": z}


@plum.dispatch
def vconvert(
    to_rep: r.LonLatSphericalPos, from_rep: r.SphericalPos, p: PDict, /
) -> Mapping:
    """SphericalPos -> LonLatSphericalPos."""
    del to_rep, from_rep  # Unused
    lat = (
        u.Quantity(90, "deg")
        if isinstance(p["theta"], u.AbstractQuantity)
        else jnp.pi / 2
    ) - p["theta"]
    return {"lon": p["phi"], "lat": lat, "distance": p["r"]}


@plum.dispatch.multi(
    (type[r.LonLatSphericalVel], type[r.SphericalVel], Mapping),
    (type[r.LonLatSphericalAcc], type[r.SphericalAcc], Mapping),
)
def vconvert(to_rep: r.AbstractVel, from_rep: r.AbstractVel, p: PDict, /) -> Mapping:
    del to_rep, from_rep  # Unused
    return {"distance": p["r"], "lat": -p["theta"], "lon": p["phi"]}


@plum.dispatch.multi(
    (type[r.MathSphericalPos], type[r.SphericalPos], Mapping),
    (type[r.SphericalPos], type[r.MathSphericalPos], Mapping),
    (type[r.MathSphericalVel], type[r.SphericalVel], Mapping),
    (type[r.SphericalVel], type[r.MathSphericalVel], Mapping),
    (type[r.MathSphericalAcc], type[r.SphericalAcc], Mapping),
    (type[r.SphericalAcc], type[r.MathSphericalAcc], Mapping),
)
def vconvert(
    to_rep: r.AbstractRepresentation, from_rep: r.AbstractRepresentation, p: PDict, /
) -> Mapping:
    del to_rep, from_rep  # Unused
    return {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}


# ---------------------------------------------------------
# differentials


@plum.dispatch
def vconvert(
    to_rep: r.AbstractRepresentation, from_dif: Vector, from_pos: Vector, /
) -> Vector:
    """Convert a vector from one differential to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.nvecs as cxv

    >>> qvec = cxv.Vector.from_([1, 1, 1], "m")
    >>> vvec = cxv.Vector.from_([10, 10, 10], "m/s")

    >>> sph_vvec = cxv.vconvert(cxv.r.sphericalvel, vvec, qvec)
    >>> print(sph_vvec)

    """
    # Checks
    from_dif = eqx.error_if(
        from_dif,
        not isinstance(from_dif.kind, r.AbstractVel | r.AbstractAcc),
        "from_dif must be a velocity or acceleration vector",
    )
    from_pos = eqx.error_if(
        from_pos,
        not isinstance(from_pos.kind, r.AbstractPos),
        "from_pos must be a position vector",
    )

    # Transform the position to the type required by the differential to
    # construct the Jacobian. E.g. if we are transforming CartVel1D ->
    # RadialVel, we need the Jacobian of the CartPos1D -> RadialPos so the
    # position must be transformed to CartPos1D.
    # TODO: generalize to work with higher-order differentials
    n = -2 if isinstance(from_dif.kind, r.AbstractAcc) else -1
    in_pos = from_dif.kind.time_nth_derivative(n=n)
    from_posv = vconvert(in_pos, from_pos)

    p = vconvert(to_rep, from_dif.kind, from_dif.data, from_posv.data)
    # Return a new vector
    return Vector(p, to_rep)


jac_pos_fn_scalar = jax.jit(
    jax.jacfwd(cxapi.vconvert, argnums=2), static_argnums=(0, 1)
)


def jac_pos_fn(
    to_rep: r.AbstractPos, from_rep: r.AbstractPos, data: Mapping, /
) -> Mapping:
    return jax.vmap(jac_pos_fn_scalar, in_axes=(None, None, 0))(
        to_rep, from_rep, jtu.map(jnp.atleast_1d, data)
    )


def compute_jac(
    to_rep: r.AbstractPos, from_rep: r.AbstractPos, data: Mapping, /
) -> dict[str, dict[str, u.AbstractQuantity]]:
    """Compute the Jacobian of the transformation."""
    # Compute the Jacobian of the transformation.
    # NOTE: this is using Quantities. Using raw arrays is ~20x faster.
    jac = jac_pos_fn(to_rep, from_rep, data)
    # Restructure the Jacobian:
    # from: ``{to_k: Quantity({from_k: Quantity(dto/dfrom, u_from)}, u_to)}``
    # to  : ``{to_k: {from_k: Quantity(dto/dfrom, u_to/u_from)}}``.
    jac = {
        out_k: {
            k: BareQuantity(v.value, out_v.unit / v.unit)
            for k, v in out_v.value.items()
        }
        for out_k, out_v in jac.items()
    }
    return jac  # noqa: RET504


is_q_or_arr = lambda x: is_any_quantity(x) or eqx.is_array(x)  # noqa: E731


@ft.partial(jax.jit, inline=True)
def inner_dot(
    inner: dict[str, u.AbstractQuantity],
    vec: dict[str, u.AbstractQuantity],
) -> u.AbstractQuantity:
    """Dot product of two dicts.

    This is a helper function for the `dot_jac_vec` function.

    Parameters
    ----------
    inner
        The first dict.
        The structure is ``{from_k: Quantity(v, unit)}``.
    vec
        The second dict.
        The structure is ``{from_k: Quantity(v, unit)}``.

    """
    return jtu.reduce(
        jnp.add,
        jtu.map(jnp.multiply, inner, vec, is_leaf=is_q_or_arr),
        is_leaf=is_q_or_arr,
    )


@jax.jit
def dot_jac_vec(
    jac: dict[str, dict[str, u.AbstractQuantity]], vec: dict[str, u.AbstractQuantity]
) -> dict[str, u.AbstractQuantity]:
    """Dot product of a Jacobian dict and a vector dict.

    This is a helper function for the `vconvert` function.

    Parameters
    ----------
    jac
        The Jacobian of the transformation.
        The structure is ``{to_k: {from_k: Quantity(v, unit)}}``.
    vec
        The vector to transform.
        The structure is ``{from_k: Quantity(v, unit)}``.

    Examples
    --------
    >>> import unxt as u

    >>> J = {"r": {"x": u.Quantity(1, ""), "y": u.Quantity(2, "")},
    ...      "phi": {"x": u.Quantity(3, "rad/km"), "y": u.Quantity(4, "rad/km")}}

    >>> v = {"x": u.Quantity(1, "km"), "y": u.Quantity(2, "km")}

    >>> dot_jac_vec(J, v)
    {'phi': Quantity(Array(11, dtype=int32, ...), unit='rad'),
     'r': Quantity(Array(5, dtype=int32, ...), unit='km')}

    """
    # TODO: rewrite this by separating the units and the values
    return {k: inner_dot(inner, vec) for k, inner in jac.items()}


@plum.dispatch.multi(
    (r.AbstractVel, r.AbstractVel, PDict, PDict),
    (r.AbstractAcc, r.AbstractAcc, PDict, PDict),
)
def vconvert(
    to_dif: r.AbstractRepresentation,
    from_dif: r.AbstractRepresentation,
    p_dif: PDict,
    p_pos: PDict,
    /,
) -> PDict:
    # Compute the Jacobian of the position transformation.
    # The position is assumed to be in the type required by the differential to
    # construct the Jacobian. E.g. for CartVel1D -> RadialVel, we need the
    # Jacobian of the CartPos1D -> RadialPos transform.
    n = -2 if isinstance(to_dif, r.AbstractAcc) else -1
    to_pos = cast("r.AbstractPos", to_dif.time_nth_derivative(n))
    from_pos = cast("r.AbstractPos", from_dif.time_nth_derivative(n))
    jac = compute_jac(to_pos, from_pos, p_pos)

    # Transform the differential by dotting with the Jacobian.
    to_p_dif = dot_jac_vec(jac, p_dif)

    # Reshape the output to the shape of the input
    shape = jnp.broadcast_shapes(*[v.shape for v in p_dif.values()])
    to_p_dif = jtu.map(lambda x: jnp.reshape(x, shape), to_p_dif)

    return to_p_dif  # noqa: RET504


# ===================================================================
# Dataclassish


@plum.dispatch
def replace(obj: Vector, /, **kwargs: Any) -> Vector:
    """Replace fields of a vector.

    Examples
    --------
    >>> import dataclassish
    >>> import unxt as u
    >>> import coordinax.nvecs as cxv

    >>> vec = cxv.Vector.from_([1, 2, 3], "m")
    >>> vec

    >>> dataclassish.replace(vec, z=u.Quantity(4, "km"))
    <CartesianPos3D: (x, y, z) [m]
        [1 2 4000]>

    """
    kind = kwargs.pop("kind", obj.kind)
    return Vector(data=dataclassish.replace(obj.data, **kwargs), kind=kind)


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

    >>> a = cx.vecs.Vector.from_([1, 2, 3], "m/s2")
    >>> print(a)
    <CartesianAcc3D: (x, y, z) [m / s2]
        [1 2 3]>

    >>> print(jnp.broadcast_to(a, (1, 3)))
    <CartesianAcc3D: (x, y, z) [m / s2]
        [[1 2 3]]>

    """
    c_shape = shape[:-1]
    return Vector(
        jtu.map(lambda v: jnp.broadcast_to(v, c_shape), operand.data),
        operand.kind,
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

    >>> qlax.convert_element_type(vec, float)
    Vector(data=Quantity([1., 2., 3.], unit='m'))

    """
    convert_p = quax.quaxify(jax.lax.convert_element_type_p.bind)
    data = jtu.map(lambda v: convert_p(v, **kwargs), operand.data)
    return Vector(data, operand.kind)


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
    rhs = cast("Vector", rhs.vconvert(lhs.kind))
    data = jtu.map(jnp.add, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, lhs.kind)


@quax.register(jax.lax.sub_p)
def sub_p_absvecs(lhs: Vector, rhs: Vector, /) -> Vector:
    """Element-wise subtraction of two vectors."""
    rhs = cast("Vector", rhs.vconvert(lhs.kind))
    data = jtu.map(jnp.subtract, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, lhs.kind)


@quax.register(jax.lax.mul_p)
def mul_p_absvecs(lhs: int | float | Array, rhs: Vector, /) -> Vector:
    """Element-wise multiplication of a scalar and a vector."""
    data = jtu.map(lambda v: jnp.multiply(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, rhs.kind)


@quax.register(jax.lax.mul_p)
def mul_p_vecs(lhs: Vector, rhs: int | float | Array, /) -> Vector:
    """Element-wise multiplication of a vector and a scalar."""
    data = jtu.map(lambda v: jnp.multiply(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, lhs.kind)


@quax.register(jax.lax.div_p)
def div_p_absvecs(lhs: int | float | Array, rhs: Vector, /) -> Vector:
    """Element-wise division of a scalar by a vector."""
    data = jtu.map(lambda v: jnp.divide(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, rhs.kind)


@quax.register(jax.lax.div_p)
def div_p_vecs(lhs: Vector, rhs: int | float | Array, /) -> Vector:
    """Element-wise division of a vector by a scalar."""
    data = jtu.map(lambda v: jnp.divide(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, lhs.kind)


# ===================================================================
# Unxt


class ToUnitsOptions(Enum):
    """Options for the units argument of `Vector.uconvert`."""

    consistent = "consistent"
    """Convert to consistent units."""


@plum.dispatch
def uconvert(usys: u.AbstractUnitSystem, vector: Vector, /) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> usys = u.unitsystem("m", "s", "kg", "rad")

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(u.uconvert(usys, vec))
    <CartesianPos3D: (x, y, z) [m]
        [1000. 2000. 3000.]>

    """
    data = {k: u.uconvert(usys[u.dimension_of(v)], v) for k, v in vector.data.items()}
    return Vector(data, vector.kind)


@plum.dispatch
def uconvert(units: Mapping[Dimension, Unit | str], vector: Vector, /) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.vecs.CartesianPos2D(x=u.Quantity(1, "m"), y=u.Quantity(2, "km"))
    >>> cart.uconvert({u.dimension("length"): "km"})
    CartesianPos2D(x=Quantity(0.001, unit='km'), y=Quantity(2, unit='km'))

    This also works for vectors with different units:

    >>> sph = cx.SphericalPos(r=u.Quantity(1, "m"), theta=u.Quantity(45, "deg"),
    ...                       phi=u.Quantity(3, "rad"))
    >>> sph.uconvert({u.dimension("length"): "km", u.dimension("angle"): "deg"})
    SphericalPos(
      r=Distance(0.001, unit='km'),
      theta=Angle(45, unit='deg'),
      phi=Angle(171.88734, unit='deg')
    )

    """
    # # Ensure `units_` is PT -> Unit
    units_ = {u.dimension(k): v for k, v in units.items()}
    data = {k: u.uconvert(units_[u.dimension_of(v)], v) for k, v in vector.data.items()}
    return Vector(data, vector.kind)


@plum.dispatch
def uconvert(units: Mapping[str, Any], vector: Vector, /) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.vecs.CartesianPos2D(x=u.Quantity(1, "m"), y=u.Quantity(2, "km"))
    >>> cart.uconvert({"x": "km", "y": "m"})
    CartesianPos2D(x=Quantity(0.001, unit='km'), y=Quantity(2000., unit='m'))

    This also works for converting just some of the components:

    >>> cart.uconvert({"x": "km"})
    CartesianPos2D(x=Quantity(0.001, unit='km'), y=Quantity(2, unit='km'))

    This also works for vectors with different units:

    >>> sph = cx.SphericalPos(r=u.Quantity(1, "m"), theta=u.Quantity(45, "deg"),
    ...                       phi=u.Quantity(3, "rad"))
    >>> sph.uconvert({"r": "km", "theta": "rad"})
    SphericalPos(
      r=Distance(0.001, unit='km'),
      theta=Angle(0.7853982, unit='rad'),
      phi=Angle(3., unit='rad')
    )

    """
    data = {  # (component: unit)
        k: u.uconvert(units.get(k, u.unit_of(v)), v)  # default to original unit
        for k, v in vector.data.items()
    }
    return Vector(data, vector.kind)


@plum.dispatch
def uconvert(flag: Literal[ToUnitsOptions.consistent], vector: Vector, /) -> Vector:
    """Convert the vector to a self-consistent set of units.

    Parameters
    ----------
    flag
        The vector is converted to consistent units by looking for the first
        quantity with each physical type and converting all components to
        the units of that quantity.
    vector
        The vector to convert.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "m")

    If all you want is to convert to consistent units, you can use
    ``"consistent"``:

    >>> cart.uconvert(cx.vecs.ToUnitsOptions.consistent)
    CartesianPos2D(x=Quantity(1, unit='m'), y=Quantity(2, unit='m'))

    >>> sph = cart.vconvert(cx.SphericalPos)
    >>> sph.uconvert(cx.vecs.ToUnitsOptions.consistent)
    SphericalPos(
      r=Distance(2.236068, unit='m'),
      theta=Angle(1.5707964, unit='rad'),
      phi=Angle(1.1071488, unit='rad')
    )

    """
    dim2unit = {}
    units_ = {}
    for k, v in vector.data.items():
        pt = u.dimension_of(v)
        if pt not in dim2unit:
            dim2unit[pt] = u.unit_of(v)
        units_[k] = dim2unit[pt]

    data = {k: u.uconvert(units_[k], v) for k, v in vector.data.items()}
    return Vector(data, vector.kind)


@plum.dispatch
def uconvert(usys: str, vector: Vector, /) -> Vector:
    """Convert the vector to the given units system.

    Parameters
    ----------
    usys
        The units system to convert to, as a string.
    vector
        The vector to convert.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> usys = "galactic"
    >>> vector = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> u.uconvert(usys, vector)
    CartesianPos3D(
      x=Quantity(3.2407793e-20, unit='kpc'),
      y=Quantity(6.4815585e-20, unit='kpc'),
      z=Quantity(9.722338e-20, unit='kpc')
    )

    """
    usys = u.unitsystem(usys)
    return uconvert(usys, vector)
