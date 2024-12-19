# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanSpatialTranslation"]


from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
import jax
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AbstractQuantity

from .base import AbstractGalileanOperator
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors.base import AbstractPos, AbstractVel
from coordinax._src.vectors.d1 import CartesianPos1D
from coordinax._src.vectors.d2 import CartesianPos2D
from coordinax._src.vectors.d3 import CartesianPos3D
from coordinax._src.vectors.d4 import FourVector

##############################################################################
# Spatial Translations


def converter(x: Any) -> AbstractPos:
    """Convert for the spatial translation operator."""
    out: AbstractPos | None = None
    if isinstance(x, GalileanSpatialTranslation):
        out = x.translation
    elif isinstance(x, AbstractPos):
        out = x
    elif isinstance(x, AbstractQuantity):
        shape: tuple[int, ...] = x.shape
        match shape:
            case (1,):
                out = CartesianPos1D.from_(x)
            case (2,):
                out = CartesianPos2D.from_(x)
            case (3,):
                out = CartesianPos3D.from_(x)
            case _:
                msg = f"Cannot convert {x} to a spatial translation vector."
                raise TypeError(msg)

    if out is None:
        msg = f"Cannot convert {x} to a spatial translation vector."
        raise TypeError(msg)

    return out


@final
class GalileanSpatialTranslation(AbstractGalileanOperator):
    r"""Operator for Galilean spatial translations.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t+s, \mathbf{x} + \mathbf {a})

    Parameters
    ----------
    translation : :class:`vector.AbstractPos3D`
        The spatial translation vector. This parameters accepts either a
        :class:`vector.AbstractPos3D` instance or uses
        :meth:`vector.CartesianPos3D.from_` to enable a variety of more
        convenient input types to create a Cartesian vector. See
        :class:`vector.CartesianPos3D` for details.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1.0, 2.0, 3.0], "km")
    >>> op
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    Note that the translation is a :class:`vector.CartesianPos3D`, which was
    constructed from a 1D array, using :meth:`vector.CartesianPos3D.from_`. We
    can also construct it directly, which allows for other vector types.

    >>> shift = cx.SphericalPos(r=u.Quantity(1.0, "km"),
    ...                         theta=u.Quantity(jnp.pi/2, "rad"),
    ...                         phi=u.Quantity(0, "rad"))
    >>> op = cx.ops.GalileanSpatialTranslation(shift)
    >>> op
    GalileanSpatialTranslation(SphericalPos( ... ))

    Translation operators can be applied to :class:`vector.AbstractPos`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> op(q)
    CartesianPos3D( ... )

    And to :class:`~unxt.Quantity`:

    >>> q = u.Quantity([0, 0, 0], "km")
    >>> op(q).value.round(2)
    Array([ 1.,  0., -0.], dtype=float32)

    :class:`coordinax.ops.GalileanSpatialTranslation` can be used
    for other dimensional vectors as well:

    - 1D:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1], "km")
    >>> q = u.Quantity([0], "km")
    >>> op(q)
    Quantity['length'](Array([1], dtype=int32), unit='km')

    >>> vec = cx.vecs.CartesianPos1D.from_(q).vconvert(cx.vecs.RadialPos)
    >>> op(vec)
    RadialPos(r=Distance(value=i32[], unit=Unit("km")))

    - 2D:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2], "km")
    >>> q = u.Quantity([0, 0], "km")
    >>> op(q)
    Quantity['length'](Array([1, 2], dtype=int32), unit='km')

    >>> vec = cx.vecs.CartesianPos2D.from_(q).vconvert(cx.vecs.PolarPos)
    >>> op(vec)
    PolarPos( r=Distance(value=f32[], unit=Unit("km")),
              phi=Angle(value=f32[], unit=Unit("rad")) )

    - 3D:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "km")
    >>> q = u.Quantity([0, 0, 0], "km")
    >>> op(q)
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='km')

    >>> vec = cx.CartesianPos3D.from_(q).vconvert(cx.SphericalPos)
    >>> op(vec)
    SphericalPos( r=Distance(value=f32[], unit=Unit("km")),
                  theta=Angle(value=f32[], unit=Unit("rad")),
                  phi=Angle(value=f32[], unit=Unit("rad")) )

    Many operators are time dependent and require a time argument. This operator
    is time independent and will pass through the time argument:

    >>> t = u.Quantity(0, "Gyr")
    >>> op(q, t)[1] is t
    True

    """

    translation: AbstractPos = eqx.field(converter=converter)
    """The spatial translation.

    This parameters accepts either a :class:`vector.AbstractVector` instance or
    uses a Cartesian vector from_ to enable a variety of more convenient
    input types to create a Cartesian vector. See
    :class:`vector.CartesianPos3D.from_` for an example when doing a 3D
    translation.
    """

    # -------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean translation is an inertial frame-preserving transformation.

        Examples
        --------
        >>> import coordinax.ops as cxo

        >>> op = GalileanSpatialTranslation.from_([1, 1, 1], "km")

        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanSpatialTranslation":
        """The inverse of the operator.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "km")

        >>> op.inverse
        GalileanSpatialTranslation(CartesianPos3D( ... ))

        >>> print(op.inverse.translation)
        <CartesianPos3D (x[km], y[km], z[km])
            [-1 -1 -1]>

        """
        return GalileanSpatialTranslation(-self.translation)

    # -------------------------------------------

    @AbstractOperator.__call__.dispatch  # type: ignore[attr-defined, misc]
    def __call__(
        self: "GalileanSpatialTranslation", q: AbstractPos, /, **__: Any
    ) -> AbstractPos:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "km")

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> t = u.Quantity(0, "Gyr")
        >>> newq = op(q)
        >>> print(newq)
        <CartesianPos3D (x[km], y[km], z[km])
            [2 3 4]>

        """
        return q + self.translation

    # -------------------------------------------
    # Arithmetic operations

    def __neg__(self: "GalileanSpatialTranslation") -> "GalileanSpatialTranslation":
        """Negate the rotation.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "km")
        >>> print((-op).translation)
        <CartesianPos3D (x[km], y[km], z[km])
            [-1 0 0]>

        """
        return replace(self, translation=-self.translation)

    # -------------------------------------------
    # Python special methods

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.translation!r})"


# ======================================================================
# Call dispatches


@AbstractOperator.__call__.dispatch
def call(
    self: GalileanSpatialTranslation,
    q: AbstractPos,
    t: u.Quantity["time"],
    /,
    **__: Any,
) -> tuple[AbstractPos, u.Quantity["time"]]:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "km")

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> t = u.Quantity(0, "Gyr")
    >>> newq, newt = op(q, t)
    >>> print(newq)
    <CartesianPos3D (x[km], y[km], z[km])
        [2 3 4]>

    This spatial translation is time independent.

    >>> op(q, u.Quantity(1, "Gyr"))[0].x == newq.x
    Array(True, dtype=bool)

    """
    return q + self.translation, t


@AbstractOperator.__call__.dispatch
def call(self: GalileanSpatialTranslation, v4: FourVector, /, **__: Any) -> AbstractPos:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "km")

    >>> v4 = cx.FourVector.from_([0, 1, 2, 3], "km")
    >>> newv4 = op(v4)
    >>> print(newv4)
    <FourVector (t[s], q=(x[km], y[km], z[km]))
        [0. 2. 3. 4.]>

    """
    return replace(v4, q=v4.q + self.translation)


@jax.jit
@AbstractOperator.__call__.dispatch
def call(
    self: GalileanSpatialTranslation, qvec: AbstractPos, pvec: AbstractVel, /, **__: Any
) -> tuple[AbstractPos, AbstractVel]:
    r"""Apply the translation to the coordinates.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "km")

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> p = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> newq, newp = op(q, p)
    >>> print(newq, newp, sep="\n")
    <CartesianPos3D (x[km], y[km], z[km])
        [1 1 1]>
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [1. 2. 3.]>

    """
    newqvec = self(qvec)

    # TODO: figure out how to do this in general, then all these dispatches
    # can be consolidated. And do it on vectors, not the quantities.
    #
    # Translate the velocity (this operator will have no effect on the
    # velocity).
    # 1. convert to a Quantity in Cartesian coordinates.
    q = convert(qvec.vconvert(qvec._cartesian_cls), u.Quantity)  # noqa: SLF001
    p = convert(pvec.vconvert(pvec._cartesian_cls, q), u.Quantity)  # noqa: SLF001
    # 1.5 cast to float dtype  # TODO: more careful casting
    q, p = q.astype(float, copy=False), p.astype(float, copy=False)
    # 2. create the Jacobian of the operation on the position
    jac = u.experimental.jacfwd(self.__call__, argnums=0, units=(q.unit,))(q)
    # 3. apply the Jacobian to the velocity
    newp = jac @ p
    # 4. convert the Quantity back to a Cartesian vector
    newpvec = pvec._cartesian_cls.from_(newp)  # noqa: SLF001
    # 5. convert the Quantity to the original vector type
    newpvec = newpvec.vconvert(type(pvec), newqvec)

    return newqvec, newpvec


@AbstractOperator.__call__.dispatch
def call(
    self: GalileanSpatialTranslation,
    q: u.Quantity["length"],
    p: u.Quantity["speed"],
    /,
    **__: Any,
) -> tuple[u.Quantity["length"], u.Quantity["speed"]]:
    r"""Apply the translation to the coordinates.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "km")

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> p = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> newq, newp = op(q, p)
    >>> print(newq, newp, sep="\n")
    <CartesianPos3D (x[km], y[km], z[km])
        [1 1 1]>
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [1. 2. 3.]>

    """
    newq = self(q)

    # TODO: figure out how to do this in general, then all these dispatches
    # can be consolidated. And do it on vectors, not the quantities.
    #
    # Translate the velocity (this operator will have no effect on the
    # velocity).
    # 2. create the Jacobian of the operation on the position
    jac = u.experimental.jacfwd(self.__call__, argnums=0, units=(q.unit,))(q)
    # 3. apply the Jacobian to the velocity
    newp = jac @ p

    return newq, newp


# ======================================================================
# Simplification


@dispatch
def simplify_op(
    op: GalileanSpatialTranslation, /, **kwargs: Any
) -> GalileanSpatialTranslation | Identity:
    """Simplify a Galilean spatial translation operator.

    Examples
    --------
    >>> import coordinax as cx

    An operator with real effect cannot be simplified:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "m")
    >>> cx.ops.simplify_op(op)
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    An operator with no effect can be simplified:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([0, 0, 0], "m")
    >>> cx.ops.simplify_op(op)
    Identity()

    """
    # Check if the translation is zero.
    if jnp.allclose(
        convert(op.translation, u.Quantity).value, jnp.zeros((3,)), **kwargs
    ):
        return Identity()
    return op


@dispatch
def simplify_op(
    op1: GalileanSpatialTranslation, op2: GalileanSpatialTranslation, /
) -> GalileanSpatialTranslation:
    """Combine two spatial translations into a single translation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op1 = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "km")
    >>> op2 = cx.ops.GalileanSpatialTranslation.from_([0, 1, 0], "km")

    >>> op3 = cx.ops.simplify_op(op1, op2)
    >>> op3
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    >>> op3.translation == op1.translation + op2.translation
    Array(True, dtype=bool)

    """
    return GalileanSpatialTranslation(op1.translation + op2.translation)
