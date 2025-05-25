# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanSpatialTranslation"]


from dataclasses import replace
from typing import Any, Literal, cast, final

import equinox as eqx
import jax
import wadler_lindig as wl
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractGalileanOperator
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel
from coordinax._src.vectors.d1 import CartesianPos1D
from coordinax._src.vectors.d2 import CartesianPos2D
from coordinax._src.vectors.d3 import CartesianPos3D
from coordinax._src.vectors.d4 import FourVector
from coordinax._src.vectors.private_api import spatial_component

##############################################################################
# Spatial Translations


def converter(x: Any) -> AbstractPos:
    """Convert for the spatial translation operator."""
    out: AbstractPos | None
    if isinstance(x, GalileanSpatialTranslation):
        out = x.translation
    elif isinstance(x, AbstractPos):
        out = x
    elif isinstance(x, u.AbstractQuantity):
        shape: tuple[int, ...] = x.shape
        match shape:
            case (1,):
                out = cast(AbstractPos, CartesianPos1D.from_(x))
            case (2,):
                out = cast(AbstractPos, CartesianPos2D.from_(x))
            case (3,):
                out = cast(AbstractPos, CartesianPos3D.from_(x))
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
    translation
        The spatial translation vector. This parameters accepts either a
        `vector.AbstractPos3D` instance or uses
        `coordinax.vecs.CartesianPos3D.from_` to enable a variety of more
        convenient input types to create a Cartesian vector. See
        `coordinax.vecs.CartesianPos3D` for details.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1.0, 2.0, 3.0], "km")
    >>> op
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    Note that the translation is a `coordinax.vecs.CartesianPos3D`, which was
    constructed from a 1D array, using `coordinax.vecs.CartesianPos3D.from_`. We
    can also construct it directly, which allows for other vector types.

    >>> shift = cx.SphericalPos(r=u.Quantity(1.0, "km"),
    ...                         theta=u.Quantity(jnp.pi/2, "rad"),
    ...                         phi=u.Quantity(0, "rad"))
    >>> op = cx.ops.GalileanSpatialTranslation(shift)
    >>> op
    GalileanSpatialTranslation(SphericalPos( ... ))

    Translation operators can be applied to any `vector.AbstractPos`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> op(q)
    CartesianPos3D( ... )

    Actually, the operator is very flexible and can be applied to many types of
    input. Let's work up the type ladder:

    - `unxt.Quantity`:

    >>> q = u.Quantity([0, 0, 0], "km")
    >>> op(q).value.round(2)
    Array([ 1.,  0., -0.], dtype=float32)

    `coordinax.ops.GalileanSpatialTranslation` can be used for other dimensional
    vectors as well:

    - 1D:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1], "km")
    >>> q = u.Quantity([0], "km")
    >>> op(q)
    Quantity(Array([1], dtype=int32), unit='km')

    >>> vec = cx.vecs.CartesianPos1D.from_(q).vconvert(cx.vecs.RadialPos)
    >>> op(vec)
    RadialPos(r=Distance(1, unit='km'))

    - 2D:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2], "km")
    >>> q = u.Quantity([0, 0], "km")
    >>> op(q)
    Quantity(Array([1, 2], dtype=int32), unit='km')

    >>> vec = cx.vecs.CartesianPos2D.from_(q).vconvert(cx.vecs.PolarPos)
    >>> op(vec)
    PolarPos(r=Distance(2.236068, unit='km'), phi=Angle(1.1071488, unit='rad'))

    - 3D:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "km")
    >>> q = u.Quantity([0, 0, 0], "km")
    >>> op(q)
    Quantity(Array([1, 2, 3], dtype=int32), unit='km')

    >>> vec = cx.CartesianPos3D.from_(q).vconvert(cx.SphericalPos)
    >>> op(vec)
    SphericalPos(
      r=Distance(3.7416575, unit='km'),
      theta=Angle(0.64052236, unit='rad'),
      phi=Angle(1.1071488, unit='rad')
    )

    Many operators are time dependent and require a time argument. This operator
    is time independent and will pass through the time argument:

    >>> t = u.Quantity(0, "Gyr")
    >>> op(t, q)[0] is t
    True

    `coordinax.ops.GalileanSpatialTranslation` can be applied to other input
    types. Let's work up the type ladder:

    - `jax.Array`: Note that since the operator is unitful but the

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "km")

    >>> q = jnp.array([0, 0, 0])

    """

    translation: AbstractPos = eqx.field(converter=converter)
    """The spatial translation.

    This parameters accepts either a `vector.AbstractVector` instance or
    uses a Cartesian vector from_ to enable a variety of more convenient
    input types to create a Cartesian vector. See
    `coordinax.vecs.CartesianPos3D.from_` for an example when doing a 3D
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
        <CartesianPos3D: (x, y, z) [km]
            [-1 -1 -1]>

        """
        return GalileanSpatialTranslation(-self.translation)

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
        <CartesianPos3D: (x, y, z) [km]
            [-1 0 0]>

        """
        return replace(self, translation=-self.translation)

    # -------------------------------------------

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation."""
        return (
            wl.TextDoc(f"{self.__class__.__name__}(")
            + wl.pdoc(self.translation, **kwargs)
            + wl.TextDoc(")")
        )


# ======================================================================
# Call dispatches

# ---------------------------
# Fundamental dispatches


@AbstractOperator.__call__.dispatch
def call(self: GalileanSpatialTranslation, q: AbstractPos, /, **__: Any) -> AbstractPos:
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
    <CartesianPos3D: (x, y, z) [km]
        [2 3 4]>

    """
    return cast(AbstractPos, q + self.translation)


@AbstractOperator.__call__.dispatch
def call(
    self: GalileanSpatialTranslation,
    t: u.AbstractQuantity,
    q: AbstractPos,
    /,
    **__: Any,
) -> tuple[u.AbstractQuantity, AbstractPos]:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "km")

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> t = u.Quantity(0, "Gyr")
    >>> newt, newq = op(t, q)
    >>> print(newq)
    <CartesianPos3D: (x, y, z) [km]
        [2 3 4]>

    This spatial translation is time independent.

    >>> t = u.Quantity(1, "Gyr")
    >>> op(t, q)[1].x == newq.x
    Array(True, dtype=bool)

    """
    return t, q + self.translation


# ---------------------------


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
    <FourVector: (t[s], q=(x, y, z) [km])
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
    <CartesianPos3D: (x, y, z) [km]
        [1 1 1]>
    <CartesianVel3D: (x, y, z) [km / s]
        [1. 2. 3.]>

    >>> q = cx.CartesianPos3D.from_([[0, 0, 0], [0, 1, 0]], "km")
    >>> newq, newp = op(q, p)
    >>> print(newq, newp, sep="\n")
    <CartesianPos3D: (x, y, z) [km]
        [[1 1 1]
         [1 2 1]]>
    <CartesianVel3D: (x, y, z) [km / s]
        [[1. 2. 3.]
         [1. 2. 3.]]>

    """
    newqvec = self(qvec)

    # TODO: figure out how to do this in general, then all these dispatches
    # can be consolidated. And do it on vectors, not the quantities.
    #
    # Translate the velocity (this operator will have no effect on the
    # velocity).
    # 1. convert to a Quantity in Cartesian coordinates.
    qvec = spatial_component(qvec)
    pvec = eqx.error_if(
        pvec,
        pvec._dimensionality() != qvec._dimensionality(),
        "The position and velocity vectors must have the same dimensionality.",
    )

    qcart_cls = qvec.cartesian_type
    pcart_cls = pvec.cartesian_type
    q = convert(qvec.vconvert(qcart_cls), u.Quantity)
    p = convert(pvec.vconvert(pcart_cls, qvec), u.Quantity)
    # 1.5 flatten all but the last axis  # TODO: not need to flatten
    batch = jnp.broadcast_shapes(q.shape[:-1], p.shape[:-1])
    q, p = jnp.reshape(q, (-1, q.shape[-1])), jnp.reshape(p, (-1, q.shape[-1]))
    # 1.5 cast to float dtype  # TODO: more careful casting
    q, p = q.astype(float, copy=False), p.astype(float, copy=False)
    # 2. create the Jacobian of the operation on the position
    jac = jax.vmap(u.experimental.jacfwd(self.__call__, argnums=0, units=(q.unit,)))(q)
    # 3. apply the Jacobian to the velocity
    newp = jnp.einsum("bmn,bn->bm", jac, p)
    newp = jnp.reshape(newp, (*batch, newp.shape[-1]))
    # 4. convert the Quantity back to a Cartesian vector
    newpvec = pcart_cls.from_(newp)
    # 5. convert the Quantity to the original vector type
    newpvec = newpvec.vconvert(type(pvec), newqvec)

    return newqvec, newpvec


@AbstractOperator.__call__.dispatch
def call(
    self: GalileanSpatialTranslation,
    q: u.AbstractQuantity,
    p: u.Quantity["speed"],
    /,
    **__: Any,
) -> tuple[u.AbstractQuantity, u.AbstractQuantity]:
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
    <CartesianPos3D: (x, y, z) [km]
        [1 1 1]>
    <CartesianVel3D: (x, y, z) [km / s]
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
