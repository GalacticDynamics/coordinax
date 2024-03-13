# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanRotationOperator"]


from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped
from plum import convert

import quaxed.array_api as xp
import quaxed.numpy as qnp
from unxt import Quantity

from .base import AbstractGalileanOperator
from coordinax._d3.base import Abstract3DVector
from coordinax._d3.builtin import Cartesian3DVector
from coordinax.operators._base import AbstractOperator, op_call_dispatch
from coordinax.operators._funcs import simplify_op
from coordinax.operators._identity import IdentityOperator

vec_matmul = qnp.vectorize(jnp.matmul, signature="(3,3),(3)->(3)")


def converter(x: Any) -> Array:
    """Convert the input to a rotation matrix."""
    if isinstance(x, GalileanRotationOperator):
        out = x.rotation
    elif isinstance(x, Quantity):
        out = x.to_value("")
    else:
        out = x
    return jnp.asarray(out)


@final
class GalileanRotationOperator(AbstractGalileanOperator):
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
    >>> from unxt import Quantity
    >>> import coordinax.operators as co

    We can then create a rotation operator:

    >>> theta = jnp.pi / 4  # 45 degrees
    >>> Rz = jnp.asarray([[jnp.cos(theta), -jnp.sin(theta), 0],
    ...                   [jnp.sin(theta), jnp.cos(theta),  0],
    ...                   [0,              0,               1]])
    >>> op = co.GalileanRotationOperator(Rz)
    >>> op
    GalileanRotationOperator(rotation=f32[3,3])

    Translation operators can be applied to a Quantity[float, (N, 3), "...]:

    >>> q = Quantity([1, 0, 0], "m")
    >>> t = Quantity(1, "s")
    >>> newq, newt = op(q, t)
    >>> newq
    Quantity['length'](Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

    The time is not affected by the rotation.

    >>> newt
    Quantity['time'](Array(1, dtype=int32, ...), unit='s')

    This also works for a batch of vectors:

    >>> q = Quantity([[1, 0, 0], [0, 1, 0]], "m")
    >>> t = Quantity(0, "s")

    >>> newq, newt = op(q, t)
    >>> newq
    Quantity['length'](Array([[ 0.70710677,  0.70710677,  0.        ],
                              [-0.70710677,  0.70710677,  0.        ]], dtype=float32),
                       unit='m')

    Translation operators can be applied to :class:`vector.Abstract3DVector`:

    >>> from coordinax import Cartesian3DVector
    >>> q = Cartesian3DVector.constructor(q)  # from the previous example
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

    def __check_init__(self) -> None:
        # Check that the rotation matrix is orthogonal.
        _ = eqx.error_if(
            self.rotation,
            not jnp.allclose(
                self.rotation @ self.rotation.T, jnp.eye(3), **self.check_tol
            ),
            "The rotation matrix must be orthogonal.",
        )

    # -----------------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean rotation is an inertial-frame preserving transform.

        Examples
        --------
        >>> import quaxed.array_api as xp
        >>> from unxt import Quantity
        >>> from coordinax.operators import GalileanRotationOperator

        >>> theta = Quantity(45, "deg")
        >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
        ...                  [xp.sin(theta), xp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = GalileanRotationOperator(Rz)
        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanRotationOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> import quaxed.array_api as xp
        >>> from unxt import Quantity
        >>> from coordinax.operators import GalileanRotationOperator

        >>> theta = Quantity(45, "deg")
        >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
        ...                  [xp.sin(theta), xp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = GalileanRotationOperator(Rz)
        >>> op.inverse
        GalileanRotationOperator(rotation=f32[3,3])

        >>> jnp.allclose(op.rotation, op.inverse.rotation.T)
        Array(True, dtype=bool)

        """
        return replace(self, rotation=self.rotation.T)

    # -----------------------------------------------------

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanRotationOperator", q: Shaped[Quantity["length"], "*batch 3"], /
    ) -> Shaped[Quantity["length"], "*batch 3"]:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> import quaxed.array_api as xp
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian3DVector, CartesianDifferential3D
        >>> from coordinax.operators import GalileanRotationOperator

        >>> theta = Quantity(45, "deg")
        >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
        ...                  [xp.sin(theta), xp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = GalileanRotationOperator(Rz)

        >>> q = Quantity([1, 0, 0], "m")
        >>> t = Quantity(1, "s")
        >>> newq, newt = op(q, t)
        >>> newq
        Quantity[...](Array([0.70710677, 0.70710677, 0. ], dtype=float32), unit='m')

        The time is not affected by the rotation.
        >>> newt
        Quantity['time'](Array(1, dtype=int32, ...), unit='s')

        """
        return vec_matmul(self.rotation, q)

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanRotationOperator", q: Abstract3DVector, /
    ) -> Abstract3DVector:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> import quaxed.array_api as xp
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian3DVector, CartesianDifferential3D
        >>> from coordinax.operators import GalileanRotationOperator

        >>> theta = Quantity(45, "deg")
        >>> Rz = xp.asarray([[xp.cos(theta), -xp.sin(theta), 0],
        ...                  [xp.sin(theta), xp.cos(theta),  0],
        ...                  [0,             0,              1]])
        >>> op = GalileanRotationOperator(Rz)

        >>> q = Cartesian3DVector.constructor(Quantity([1, 0, 0], "m"))
        >>> t = Quantity(1, "s")
        >>> newq, newt = op(q, t)
        >>> newq.x
        Quantity['length'](Array(0.70710677, dtype=float32), unit='m')

        The time is not affected by the rotation.
        >>> newt
        Quantity['time'](Array(1, dtype=int32, ...), unit='s')

        """
        vec = convert(  # Array[float, (N, 3)]
            q.represent_as(Cartesian3DVector).to_units("consistent"), Quantity
        )
        rcart = Cartesian3DVector.constructor(vec_matmul(self.rotation, vec))
        return rcart.represent_as(type(q))

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanRotationOperator",
        q: Abstract3DVector,
        t: Quantity["time"],
        /,
    ) -> tuple[Abstract3DVector, Quantity["time"]]:
        return self(q), t

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanRotationOperator",
        q: Abstract3DVector,
        t: Quantity["time"],
        /,
    ) -> tuple[Abstract3DVector, Quantity["time"]]:
        return self(q), t


@simplify_op.register
def _simplify_op_rotation(
    op: GalileanRotationOperator, /, **kwargs: Any
) -> AbstractOperator:
    if jnp.allclose(op.rotation, xp.eye(3), **kwargs):
        return IdentityOperator()
    return op
