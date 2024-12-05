# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanRotation"]


from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
import jax
from jaxtyping import Array, Shaped
from plum import convert, dispatch
from quax import quaxify

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractGalileanOperator
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors.base import ToUnitsOptions
from coordinax._src.vectors.d3 import AbstractPos3D, CartesianPos3D

vec_matmul = quaxify(jax.numpy.vectorize(jax.numpy.matmul, signature="(3,3),(3)->(3)"))


def converter(x: Any) -> Array:
    """Convert the input to a rotation matrix."""
    if isinstance(x, GalileanRotation):
        out = x.rotation
    elif isinstance(x, u.Quantity):
        out = u.ustrip("", x)
    else:
        out = x
    return jnp.asarray(out)


@final
class GalileanRotation(AbstractGalileanOperator):
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

    >>> theta = jnp.pi / 4  # 45 degrees
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                   [jnp.sin(theta), jnp.cos(theta),  0],
    ...                   [0,              0,               1]])
    >>> op = cx.operators.GalileanRotation(Rz)
    >>> op
    GalileanRotation(rotation=f32[3,3])

    Translation operators can be applied to a Quantity[float, (N, 3), "...]:

    >>> q = u.Quantity([1, 0, 0], "m")
    >>> t = u.Quantity(1, "s")
    >>> newq, newt = op(q, t)
    >>> newq
    Quantity['length'](Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

    The time is not affected by the rotation.

    >>> newt
    Quantity['time'](Array(1, dtype=int32, ...), unit='s')

    This also works for a batch of vectors:

    >>> q = u.Quantity([[1, 0, 0], [0, 1, 0]], "m")
    >>> t = u.Quantity(0, "s")

    >>> newq, newt = op(q, t)
    >>> newq
    Quantity['length'](Array([[ 0.70710677,  0.70710677,  0.        ],
                              [-0.70710677,  0.70710677,  0.        ]], dtype=float32),
                       unit='m')

    Translation operators can be applied to :class:`vector.AbstractPos3D`:

    >>> q = cx.CartesianPos3D.from_(q)  # from the previous example
    >>> newq, newt = op(q, t)
    >>> newq.x
    Quantity['length'](Array([ 0.70710677, -0.70710677], dtype=float32), unit='m')
    >>> newq.norm().value.round(2)
    Array([1., 1.], dtype=float32)

    """

    rotation: Shaped[Array, "3 3"] = eqx.field(converter=converter)
    """The rotation vector."""

    check_tol: Mapping[str, Any] = eqx.field(
        default_factory=lambda: {"atol": 1e-7}, repr=False, static=True
    )

    # TODO: fix so that it works in jitted contexts
    # def __check_init__(self) -> None:
    #     # Check that the rotation matrix is orthogonal.
    #     _ = eqx.error_if(
    #         self.rotation,
    #         not jnp.allclose(
    #             self.rotation @ self.rotation.T, jnp.eye(3), **self.check_tol
    #         ),
    #         "The rotation matrix must be orthogonal.",
    #     )

    # -----------------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean rotation is an inertial-frame preserving transform.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.operators as cxo

        >>> theta = u.Quantity(45, "deg")
        >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
        ...                  [jnp.sin(theta), jnp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = cxo.GalileanRotation(Rz)
        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanRotation":
        """The inverse of the operator.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.operators as cxo

        >>> theta = u.Quantity(45, "deg")
        >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
        ...                  [jnp.sin(theta), jnp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = cxo.GalileanRotation(Rz)
        >>> op.inverse
        GalileanRotation(rotation=f32[3,3])

        >>> jnp.allclose(op.rotation, op.inverse.rotation.T)
        Array(True, dtype=bool)

        """
        return replace(self, rotation=self.rotation.T)

    # -----------------------------------------------------

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(
        self: "GalileanRotation", q: Shaped[u.Quantity["length"], "*batch 3"], /
    ) -> Shaped[u.Quantity["length"], "*batch 3"]:
        """Apply the rotation to the coordinates.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.operators as cxo

        >>> theta = u.Quantity(45, "deg")
        >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
        ...                  [jnp.sin(theta), jnp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = cxo.GalileanRotation(Rz)

        >>> q = u.Quantity([1, 0, 0], "m")
        >>> op(q)
        Quantity[...](Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

        THere's a related dispatch that also takes a time argument:

        >>> t = u.Quantity(1, "s")
        >>> newq, newt = op(q, t)
        >>> newq
        Quantity[...](Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

        The time is not affected by the rotation.
        >>> newt
        Quantity['time'](Array(1, dtype=int32, ...), unit='s')

        """
        return vec_matmul(self.rotation, q)

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(self: "GalileanRotation", q: AbstractPos3D, /) -> AbstractPos3D:
        """Apply the rotation to the coordinates.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        >>> theta = u.Quantity(45, "deg")
        >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
        ...                  [jnp.sin(theta), jnp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = cx.operators.GalileanRotation(Rz)

        >>> q = cx.CartesianPos3D.from_([1, 0, 0], "m")
        >>> newq = op(q)
        >>> newq.x
        Quantity['length'](Array(0.70710677, dtype=float32), unit='m')

        """
        vec = convert(  # Array[float, (N, 3)]
            q.represent_as(CartesianPos3D).uconvert(ToUnitsOptions.consistent),
            u.Quantity,
        )
        rcart = CartesianPos3D.from_(vec_matmul(self.rotation, vec))
        return rcart.represent_as(type(q))

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(
        self: "GalileanRotation", q: AbstractPos3D, t: u.Quantity["time"], /
    ) -> tuple[AbstractPos3D, u.Quantity["time"]]:
        """Apply the rotation to the coordinates.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        >>> theta = u.Quantity(45, "deg")
        >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
        ...                  [jnp.sin(theta), jnp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = cx.operators.GalileanRotation(Rz)

        >>> q = cx.CartesianPos3D.from_([1, 0, 0], "m")
        >>> t = u.Quantity(1, "s")
        >>> newq, newt = op(q, t)
        >>> newq.x
        Quantity['length'](Array(0.70710677, dtype=float32), unit='m')

        The time is not affected by the rotation.
        >>> newt
        Quantity['time'](Array(1, dtype=int32, ...), unit='s')

        """
        return self(q), t


@dispatch  # type: ignore[misc]
def simplify_op(op: GalileanRotation, /, **kwargs: Any) -> AbstractOperator:
    """Simplify the Galilean rotation operator.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.operators as cxo

    An operator with a non-identity rotation matrix is not simplified:

    >>> theta = u.Quantity(45, "deg")
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                  [jnp.sin(theta), jnp.cos(theta),  0],
    ...                  [0,             0,              1]])
    >>> op = cxo.GalileanRotation(Rz)
    >>> cxo.simplify_op(op)
    GalileanRotation(rotation=f32[3,3])

    An operator with an identity rotation matrix is simplified:

    >>> op = cxo.GalileanRotation(jnp.eye(3))
    >>> cxo.simplify_op(op)
    Identity()

    """
    if jnp.allclose(op.rotation, jnp.eye(3), **kwargs):
        return Identity()
    return op
