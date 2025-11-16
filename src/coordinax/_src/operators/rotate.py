"""Galilean coordinate transformations."""
# ruff:noqa: F401

__all__ = ("Rotate",)


from collections.abc import Callable
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

from .base import AbstractOperator, Neg
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors.base import ToUnitsOptions
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel
from coordinax._src.vectors.d3 import AbstractPos3D

vec_matmul = quaxify(jax.numpy.vectorize(jax.numpy.matmul, signature="(3,3),(3)->(3)"))

RMatrix: TypeAlias = Shaped[Array, "3 3"]


def converter(x: Any, /) -> RMatrix | Callable[[Any], RMatrix]:
    """Convert the input to a rotation matrix."""
    if isinstance(x, Rotate):
        out = x.R
    elif callable(x):
        return x
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

    R: Shaped[Array, "3 3"] | Callable[[Any], RMatrix] = eqx.field(converter=converter)
    """The rotation vector."""

    # -----------------------------------------------------

    @classmethod
    def from_euler(
        cls: type["Rotate"], seq: str, angles: u.Quantity["angle"] | u.Angle, /
    ) -> "Rotate":
        """Initialize from Euler angles.

        See `jax.scipy.spatial.transform.Rotation.from_euler`.
        `XYZ` are intrinsic rotations, `xyz` are extrinsic rotations.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.ops.Rotate.from_euler("z", u.Quantity(90, "deg"))
        >>> op.R.round(2)
        Array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)

        """
        R = Rotation.from_euler(seq, u.ustrip("deg", angles), degrees=True).as_matrix()
        return cls(R)

    @classmethod
    @AbstractOperator.from_.dispatch  # type: ignore[misc]
    def from_(cls: type["Rotate"], obj: Rotation, /) -> "Rotate":
        """Initialize from a `jax.scipy.spatial.transform.Rotation`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from jax.scipy.spatial.transform import Rotation
        >>> import coordinax as cx

        >>> R = Rotation.from_euler("z", 90, degrees=True)
        >>> op = cx.ops.Rotate.from_(R)

        >>> jnp.allclose(op.R, R.as_matrix())
        Array(True, dtype=bool)

        """
        return cls(obj.as_matrix())

    # -----------------------------------------------------

    @classmethod
    def operate(cls, params: dict[str, Any], arg: Any, /, **__: Any) -> Any:
        """Apply the :class:`coordinax.ops.Rotate` operation.

        This is the identity operation, which does nothing to the input.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.ops as cxo

        >>> q = u.Quantity([1, 2, 3], "km")
        >>> R = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> cxo.operate(cxo.Rotate, {"R": R}, q)

        >>> vec = cxo.CartesianPos3D.from_([1, 2, 3], "km")
        >>> cxo.operate(cxo.Rotate, {"R": R}, vec)

        """
        return params["R"] @ arg
        # return vec_matmul(params["R"], arg)

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
        Rotate(R=i32[3,3])

        >>> jnp.allclose(op.R, op.inverse.R.T)
        Array(True, dtype=bool)

        """
        return replace(self, R=self.R.T)

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
        >>> print((-op).R)
        [[ 0  1  0]
         [-1  0  0]
         [ 0  0 -1]]

        """
        R = Neg(self.R) if callable(self.R) else -self.R
        return replace(self, R=R)

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
        Rotate(R=f32[3,3])

        >>> jnp.allclose(op3.R, op1.R @ op2.R)
        Array(True, dtype=bool)

        """
        return replace(self, R=self.R @ other.R)


# ============================================================================
# Simplification


@dispatch
def simplify(op: Rotate, /, **kwargs: Any) -> AbstractOperator:
    """Simplify the Galilean rotation operator.

    Examples
    --------
    >>> import quaxed.numpy as jnp
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

    When two rotations are combined that cancel each other out, the result
    simplifies to an :class:`coordinax.ops.Identity`:

    >>> op = (  cx.ops.Rotate.from_euler("z", u.Quantity(45, "deg"))
    ...       @ cx.ops.Rotate.from_euler("z", u.Quantity(-45, "deg")))
    >>> cx.ops.simplify(op)
    Identity()

    """
    if jnp.allclose(op.R, jnp.eye(3), **kwargs):
        return Identity()  # type: ignore[no-untyped-call]
    return op
