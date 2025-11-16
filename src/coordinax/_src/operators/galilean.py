"""Galilean coordinate transformations."""

__all__ = ("GalileanOp",)


from typing import TYPE_CHECKING, Any, TypeAlias, cast, final, overload

import equinox as eqx
from plum import dispatch

import quaxed.numpy as jnp
from dataclassish.converters import Unless

from .add import Add
from .rotate import Rotate
from coordinax._src.operators import api
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.composite import AbstractCompositeOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.operators.pipe import Pipe

if TYPE_CHECKING:
    from typing import Self


@final
class GalileanOp(AbstractCompositeOperator):
    r"""Operator for general Galilean transformations.

    In the transformed frame the coordinates are given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t+s, R \mathbf{x} + \mathbf{v} t + \mathbf{a})

    where :math:`R` is the rotation matrix, :math:`\mathbf{v}` is the boost
    velocity, :math:`\mathbf{a}` is the spatial translationl, and :math:`s` is
    the time translation. This is equivalent to a sequential operation of 1. a
    rotation, 2. a translation, 3. a boost.

    Parameters
    ----------
    rotation
        The in-frame spatial rotation. This is a `coordinax.ops.Rotate`
    translation
        The spatial translation of the frame. See
        `coordinax.ops.Add` for alternative inputs to construct
        this parameter.
    velocity
        The boost to the frame. See `coordinax.ops.Add` for
        alternative inputs to construct this parameter.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanOp(
    ...     translation=u.Quantity([0., 2., 3., 4.], "km"),
    ...     velocity=u.Quantity([1., 2., 3.], "km/s"))
    >>> op
    GalileanOp(
      translation=Add( delta_t=..., delta_q=... ),
      velocity=Add(CartesianVel3D( ... ))
    )

    Note that the translation is a `coordinax.ops.Add` with a
    `coordinax.vecs.FourVector` translation, and the velocity is a
    `coordinax.ops.Add` with a `coordinax.vecs.AbstractVel` velocity.
    We can also construct them directly, which allows for other vector types.

    >>> op = cx.ops.GalileanOp(
    ...     translation=cx.ops.Add(
    ...         delta_t=u.Quantity(2.5, "Gyr"),
    ...         delta_q=cx.SphericalPos(r=u.Quantity(1, "km"),
    ...                                 theta=u.Quantity(90, "deg"),
    ...                                 phi=u.Quantity(0, "rad") ) ),
    ...     velocity=cx.ops.Add(
    ...         cx.CartesianVel3D.from_([1, 2, 3], "km/s") )
    ... )
    >>> op
    GalileanOp(
      translation=Add( delta_t=..., delta_q=SphericalPos( ... ) ),
      velocity=Add(CartesianVel3D( ... ))
    )

    Galilean operators can be applied to `coordinax.vecs.FourVector`:

    >>> w = cx.FourVector.from_([0, 0, 0, 0], "km")
    >>> new = op(w)
    >>> new.t.ustrip("Gyr").round(2)
    Array(2.5, dtype=float32, ...)
    >>> print(new.q)
    <CartesianPos3D: (x, y, z) [km]
        [7.889e+16 1.578e+17 2.367e+17]>

    Also the Galilean operators can also be applied to `vector.AbstractPos3D`
    and `unxt.Quantity`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> t = u.Quantity(0, "s")
    >>> newt, newq = op(t, q)
    >>> print(newq)
    <CartesianPos3D: (x, y, z) [km]
        [7.889e+16 1.578e+17 2.367e+17]>

    >>> newt
    Quantity(Array(7.8894005e+16, dtype=float32, ...), unit='s')

    """

    rotation: Rotate = eqx.field(
        default=Rotate(jnp.eye(3)),
        converter=Unless(Rotate, Rotate.from_),
    )
    """The in-frame spatial rotation."""

    translation: Add = eqx.field(
        default=Add.from_([0, 0, 0, 0], "km"),
        converter=Unless(Add, converter=Add.from_),
    )
    """The temporal + spatial translation.

    The translation vector [T, Q].  This parameters accepts either a
    `coordinax.ops.Add` instance or any input that can be used
    to construct a `coordinax.vecs.FourVector`, using
    `coordinax.vecs.FourVector.from_`. See `coordinax.vecs.FourVector` for
    details.

    """

    velocity: Add = eqx.field(
        default=Add.from_([0, 0, 0], "km/s"),
        converter=Unless(Add, converter=Add.from_),
    )
    """The boost to the frame.

    This parameters accepts either a `coordinax.ops.Add`
    instance or any input that can be used to construct one. See
    `coordinax.ops.Add.from_` for details.
    """

    @property  # type: ignore[misc]
    def operators(  # type: ignore[override]
        self,
    ) -> tuple[Rotate, Add, Add]:
        """Rotate -> translation -> boost."""
        return (self.rotation, self.translation, self.velocity)

    @overload
    def __getitem__(self, key: int) -> AbstractOperator: ...

    @overload
    def __getitem__(self, key: slice) -> "Self": ...

    def __getitem__(self, key: int | slice) -> AbstractOperator | Pipe:
        """Getitem from the operators.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanOp(
        ...     translation=u.Quantity([0., 2., 3., 4.], "km"),
        ...     velocity=u.Quantity([1., 2., 3.], "km/s"))

        >>> op[0]
        Rotate(rotation=f32[3,3])

        >>> op[1:]
        Pipe(( Add( delta_t=..., delta_q=... ),
               Add(CartesianVel3D( ... ))   ))

        """
        if isinstance(key, int):
            return self.operators[key]
        return Pipe(self.operators[key])  # type: ignore[arg-type]


# ===================================================================
# Simplification

SimplifyOpR: TypeAlias = GalileanOp | Pipe | Add | Rotate | Identity


@dispatch
def simplify(op: GalileanOp, /, **kwargs: Any) -> SimplifyOpR:
    """Simplify a Galilean operator.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    This Galilean operator cannot be simplified:

    >>> op = cxo.GalileanOp(
    ...     translation=u.Quantity([0., 2., 3., 4.], "km"),
    ...     velocity=u.Quantity([1., 2., 3.], "km/s"),
    ...     rotation=jnp.eye(3).at[0, 2].set(1),
    ... )
    >>> op
    GalileanOp(
      rotation=Rotate(rotation=f32[3,3]),
      translation=Add( delta_t=..., delta_q=... ),
      velocity=Add(CartesianVel3D( ... ))
    )

    >>> cxo.simplify(op) is op
    True

    This Galilean operator can be simplified in all its components except the
    translation:

    >>> op = cxo.GalileanOp(translation=u.Quantity([0., 2., 3., 4.], "km"))
    >>> cxo.simplify(op)
    Add( delta_t=..., delta_q=CartesianPos3D( ... ) )

    """
    simple_ops = [api.simplify(x, **kwargs) for x in op.operators]
    if any(
        not isinstance(x, type(orig))
        for x, orig in zip(simple_ops, op.operators, strict=True)
    ):
        return cast("SimplifyOpR", Pipe(tuple(simple_ops)).simplify())

    return op
