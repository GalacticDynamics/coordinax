"""Galilean coordinate transformations."""
# ruff: noqa: N806

__all__ = ("Rotate",)


from collections.abc import Mapping
from dataclasses import replace
from typing import Any, TypeAlias, final

import equinox as eqx
import jax
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array, Shaped
from plum import convert, dispatch
from quax import quaxify

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractOperator
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors.base import ToUnitsOptions
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel
from coordinax._src.vectors.d3 import AbstractPos3D

vec_matmul = quaxify(jax.numpy.vectorize(jax.numpy.matmul, signature="(3,3),(3)->(3)"))

RotationMatrix: TypeAlias = Shaped[Array, "3 3"]


def converter(x: Any) -> Array:
    """Convert the input to a rotation matrix."""
    if isinstance(x, Rotate):
        out = x.rotation
    elif isinstance(x, u.Quantity):
        out = u.ustrip("", x)
    else:
        out = x
    return jnp.asarray(out)


@final
class Rotate(AbstractOperator):
    r"""Operator for Galilean rotations.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t, R \mathbf{x})

    where :math:`R` is the rotation matrix.  Note this is NOT time dependent.

    Parameters
    ----------
    rotation : Array[float, (3, 3)]
        The rotation matrix.

    Raises
    ------
    ValueError
        If the rotation matrix is not orthogonal.

    Notes
    -----
    The Galilean rotation is not a time-dependent transformation.  This is part
    of the inhomogeneous Galilean group, which is the group of transformations
    that leave the space-time interval invariant.

    Examples
    --------
    We start with the required imports:

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a rotation operator:

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cx.ops.Rotate(Rz)
    >>> op
    Rotate(rotation=i32[3,3])

    Translation operators can be applied to a Quantity[float, (N, 3), "...]:

    >>> q = u.Quantity([1, 0, 0], "m")
    >>> t = u.Quantity(1, "s")
    >>> newt, newq = op(t, q)
    >>> newq
    Quantity(Array([0, 1, 0], dtype=int32), unit='m')

    The time is not affected by the rotation.

    >>> newt
    Quantity(Array(1, dtype=int32, ...), unit='s')

    This also works for a batch of vectors:

    >>> q = u.Quantity([[1, 0, 0], [0, 1, 0]], "m")
    >>> t = u.Quantity(0, "s")

    >>> newt, newq = op(t, q)
    >>> newq
    Quantity(Array([[ 0,  1,  0],
                    [-1,  0,  0]], dtype=int32), unit='m')

    Translation operators can be applied to `vector.AbstractPos3D`:

    >>> q = cx.CartesianPos3D.from_(q)  # from the previous example
    >>> newt, newq = op(t, q)
    >>> newq.x
    Quantity(Array([ 0, -1], dtype=int32), unit='m')
    >>> newq.norm().value.round(2)
    Array([1., 1.], dtype=float32)

    """

    rotation: Shaped[Array, "3 3"] = eqx.field(converter=converter)
    """The rotation vector."""

    #: Tolerance check on the rotation matrix.
    check_tol: Mapping[str, Any] = eqx.field(
        converter=dict, default=(("atol", 1e-7),), repr=False, static=True
    )

    # -----------------------------------------------------

    @classmethod
    def from_euler(
        cls: "type[Rotate]",
        seq: str,
        angles: u.Quantity["angle"] | u.Angle,
        /,
    ) -> "Rotate":
        """Initialize from Euler angles.

        See `jax.scipy.spatial.transform.Rotation.from_euler`.
        `XYZ` are intrinsic rotations, `xyz` are extrinsic rotations.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.ops.Rotate.from_euler("z", u.Quantity(90, "deg"))
        >>> op.rotation.round(2)
        Array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)

        """
        R = Rotation.from_euler(seq, u.ustrip("deg", angles), degrees=True).as_matrix()

        return cls(rotation=R)

    @classmethod
    @AbstractOperator.from_.dispatch  # type: ignore[misc]
    def from_(cls: "type[Rotate]", obj: Rotation, /) -> "Rotate":
        """Initialize from a `jax.scipy.spatial.transform.Rotation`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from jax.scipy.spatial.transform import Rotation
        >>> import coordinax as cx

        >>> R = Rotation.from_euler("z", 90, degrees=True)
        >>> op = cx.ops.Rotate.from_(R)

        >>> jnp.allclose(op.rotation, R.as_matrix())
        Array(True, dtype=bool)

        """
        return cls(rotation=obj.as_matrix())

    # -----------------------------------------------------

    @property
    def inverse(self) -> "Rotate":
        """The inverse of the operator.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
        >>> op = cx.ops.Rotate(Rz)
        >>> op.inverse
        Rotate(rotation=i32[3,3])

        >>> jnp.allclose(op.rotation, op.inverse.rotation.T)
        Array(True, dtype=bool)

        """
        return replace(self, rotation=self.rotation.T)

    # -----------------------------------------------------
    # Arithmetic operations

    def __neg__(self: "Rotate") -> "Rotate":
        """Negate the rotation.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
        >>> op = cx.ops.Rotate(Rz)
        >>> print((-op).rotation)
        [[ 0  1  0]
         [-1  0  0]
         [ 0  0 -1]]

        """
        return replace(self, rotation=-self.rotation)

    def __matmul__(self: "Rotate", other: Any, /) -> Any:
        """Combine two Galilean rotations.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        Two rotations can be combined:

        >>> theta1 = u.Quantity(45, "deg")
        >>> Rz1 = jnp.asarray([[jnp.cos(theta1), -jnp.sin(theta1), 0],
        ...                   [jnp.sin(theta1), jnp.cos(theta1),  0],
        ...                   [0,             0,              1]])
        >>> op1 = cx.ops.Rotate(Rz1)

        >>> theta2 = u.Quantity(90, "deg")
        >>> Rz2 = jnp.asarray([[jnp.cos(theta2), -jnp.sin(theta2), 0],
        ...                   [jnp.sin(theta2), jnp.cos(theta2),  0],
        ...                   [0,             0,              1]])
        >>> op2 = cx.ops.Rotate(Rz2)

        >>> op3 = op1 @ op2
        >>> op3
        Rotate(rotation=f32[3,3])

        >>> jnp.allclose(op3.rotation, op1.rotation @ op2.rotation)
        Array(True, dtype=bool)

        """
        return Rotate(rotation=self.rotation @ other.rotation)


# ============================================================================
# Call dispatches


@dispatch
def operate(
    self: "Rotate",
    q: Shaped[u.Quantity["length"], "*batch 3"],
    /,
    **__: Any,
) -> Shaped[u.Quantity["length"], "*batch 3"]:
    """Apply the rotation to the coordinates.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cx.ops.Rotate(Rz)

    >>> q = u.Quantity([1, 0, 0], "m")
    >>> op(q)
    Quantity(Array([0, 1, 0], dtype=int32), unit='m')

    THere's a related dispatch that also takes a time argument:

    >>> t = u.Quantity(1, "s")
    >>> newt, newq = op(t, q)
    >>> newq
    Quantity(Array([0, 1, 0], dtype=int32), unit='m')

    The time is not affected by the rotation.
    >>> newt
    Quantity(Array(1, dtype=int32, ...), unit='s')

    """
    return vec_matmul(self.rotation, q)


@dispatch
def operate(self: "Rotate", q: AbstractPos3D, /, **__: Any) -> AbstractPos3D:
    """Apply the rotation to the coordinates.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cx.ops.Rotate(Rz)

    >>> q = cx.CartesianPos3D.from_([1, 0, 0], "m")
    >>> newq = op(q)
    >>> newq.x
    Quantity(Array(0, dtype=int32), unit='m')

    """
    return self.rotation @ q


@dispatch
def operate(
    self: Rotate, t: u.Quantity["time"], q: AbstractPos3D, /
) -> tuple[u.Quantity["time"], AbstractPos3D]:
    """Apply the rotation to the coordinates.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cx.ops.Rotate(Rz)

    >>> q = cx.CartesianPos3D.from_([1, 0, 0], "m")
    >>> t = u.Quantity(1, "s")
    >>> newt, newq = op(t, q)
    >>> newq.x
    Quantity(Array(0, dtype=int32), unit='m')

    The time is not affected by the rotation.
    >>> newt
    Quantity(Array(1, dtype=int32, ...), unit='s')

    """
    return t, self(q)


@jax.jit
@dispatch
def operate(
    self: Rotate, qvec: AbstractPos, pvec: AbstractVel, /, **__: Any
) -> tuple[AbstractPos, AbstractVel]:
    r"""Apply the rotation to the coordinates and velocities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> R_z = cx.ops.Rotate.from_euler("z", u.Quantity(90, "deg"))

    >>> q = cx.CartesianPos3D.from_([1, 0, 0], "m")
    >>> p = cx.CartesianVel3D.from_([1, 0, 0], "m/s")

    >>> newq, newp = R_z(q, p)
    >>> print(newq, newp, sep="\n")
    <CartesianPos3D: (x, y, z) [m]
        [0. 1. 0.]>
    <CartesianVel3D: (x, y, z) [m / s]
        [0. 1. 0.]>

    """
    # Rotate the position.
    newqvec = self(qvec)

    # TODO: figure out how to do this without converting back to arrays.
    # XVel -> CartVel -> Q -> R@Q -> CartVel -> XVel
    cart_cls = pvec.cartesian_type
    pcvec = pvec.vconvert(cart_cls, qvec)
    p = convert(pcvec.uconvert(ToUnitsOptions.consistent), u.Quantity)
    newp = vec_matmul(self.rotation, p)
    newpcvec = cart_cls.from_(newp)
    newpvec = newpcvec.vconvert(type(pvec), newqvec)

    return newqvec, newpvec


@dispatch
def operate(
    self: Rotate,
    q: u.Quantity["length"],
    p: u.Quantity["speed"],
    /,
    **__: Any,
) -> tuple[u.Quantity["length"], u.Quantity["speed"]]:
    r"""Apply the rotation to the coordinates and velocities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> R_z = cx.ops.Rotate(jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]]))

    >>> q = u.Quantity([1., 0, 0], "m")
    >>> p = u.Quantity([1., 0, 0], "m/s")

    >>> newq, newp = R_z(q, p)
    >>> newq, newp
    (Quantity(Array([0., 1., 0.], dtype=float32), unit='m'),
     Quantity(Array([0., 1., 0.], dtype=float32), unit='m / s'))

    """
    newq = self(q)
    newp = vec_matmul(self.rotation, p)
    return newq, newp


# ============================================================================
# Simplification


@dispatch
def simplify(op: Rotate, /, **kwargs: Any) -> AbstractOperator:
    """Simplify the Galilean rotation operator.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    An operator with a non-identity rotation matrix is not simplified:

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cx.ops.Rotate(Rz)
    >>> cx.ops.simplify(op)
    Rotate(rotation=i32[3,3])

    An operator with an identity rotation matrix is simplified:

    >>> op = cx.ops.Rotate(jnp.eye(3))
    >>> cx.ops.simplify(op)
    Identity()

    """
    if jnp.allclose(op.rotation, jnp.eye(3), **kwargs):
        return Identity()
    return op


# @dispatch
# def simplify(op1: Rotate, op2: Rotate) -> Rotate:
#     """Combine two Galilean rotations.

#     Examples
#     --------
#     >>> import quaxed.numpy as jnp
#     >>> import unxt as u
#     >>> import coordinax as cx

#     Two rotations can be combined:

#     >>> theta1 = u.Quantity(45, "deg")
#     >>> Rz1 = jnp.asarray([[jnp.cos(theta1), -jnp.sin(theta1), 0],
#     ...                   [jnp.sin(theta1), jnp.cos(theta1),  0],
#     ...                   [0,             0,              1]])
#     >>> op1 = cx.ops.Rotate(Rz1)

#     >>> theta2 = u.Quantity(60, "deg")
#     >>> Rz2 = jnp.asarray([[jnp.cos(theta2), -jnp.sin(theta2), 0],
#     ...                   [jnp.sin(theta2), jnp.cos(theta2),  0],
#     ...                   [0,             0,              1]])
#     >>> op2 = cx.ops.Rotate(Rz2)

#     >>> op3 = cx.ops.simplify(op1, op2)
#     >>> op3
#     Rotate(rotation=f32[3,3])

#     >>> jnp.allclose(op3.rotation, op1.rotation @ op2.rotation)
#     Array(True, dtype=bool)

#     """
#     return op1 @ op2
