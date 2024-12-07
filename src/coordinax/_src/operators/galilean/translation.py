# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanSpatialTranslation", "GalileanTranslation"]


from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractGalileanOperator
from coordinax._src.distances import AbstractDistance
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors.base import AbstractPos
from coordinax._src.vectors.d1 import CartesianPos1D
from coordinax._src.vectors.d2 import CartesianPos2D
from coordinax._src.vectors.d3 import AbstractPos3D, CartesianPos3D
from coordinax._src.vectors.d4 import FourVector

##############################################################################
# Spatial Translations


def _converter_spatialtranslation(x: Any) -> AbstractPos:
    """Convert to a spatial translation vector."""
    out: AbstractPos | None = None
    if isinstance(x, GalileanSpatialTranslation):
        out = x.translation
    elif isinstance(x, AbstractPos):
        out = x
    elif isinstance(x, u.Quantity | AbstractDistance):
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

    >>> shift = u.Quantity([1.0, 2.0, 3.0], "kpc")
    >>> op = cx.ops.GalileanSpatialTranslation(shift)
    >>> op
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    Note that the translation is a :class:`vector.CartesianPos3D`, which was
    constructed from a 1D array, using :meth:`vector.CartesianPos3D.from_`. We
    can also construct it directly, which allows for other vector types.

    >>> shift = cx.SphericalPos(r=u.Quantity(1.0, "kpc"),
    ...                         theta=u.Quantity(jnp.pi/2, "rad"),
    ...                         phi=u.Quantity(0, "rad"))
    >>> op = cx.ops.GalileanSpatialTranslation(shift)
    >>> op
    GalileanSpatialTranslation(SphericalPos( ... ))

    Translation operators can be applied to :class:`vector.AbstractPos`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "kpc")
    >>> op(q)
    CartesianPos3D( ... )

    And to :class:`~unxt.Quantity`:

    >>> q = u.Quantity([0, 0, 0], "kpc")
    >>> op(q).value.round(2)
    Array([ 1.,  0., -0.], dtype=float32)

    :class:`coordinax.operators.GalileanSpatialTranslation` can be used
    for other dimensional vectors as well:

    - 1D:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1], "kpc")
    >>> q = u.Quantity([0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1.], dtype=float32), unit='kpc')

    >>> vec = cx.vecs.CartesianPos1D.from_(q).represent_as(cx.vecs.RadialPos)
    >>> op(vec)
    RadialPos(r=Distance(value=f32[], unit=Unit("kpc")))

    - 2D:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2], "kpc")
    >>> q = u.Quantity([0, 0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='kpc')

    >>> vec = cx.vecs.CartesianPos2D.from_(q).represent_as(cx.vecs.PolarPos)
    >>> op(vec)
    PolarPos( r=Distance(value=f32[], unit=Unit("kpc")),
              phi=Angle(value=f32[], unit=Unit("rad")) )

    - 3D:

    >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 2, 3], "kpc")
    >>> q = u.Quantity([0, 0, 0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')

    >>> vec = cx.CartesianPos3D.from_(q).represent_as(cx.SphericalPos)
    >>> op(vec)
    SphericalPos( r=Distance(value=f32[], unit=Unit("kpc")),
                  theta=Angle(value=f32[], unit=Unit("rad")),
                  phi=Angle(value=f32[], unit=Unit("rad")) )

    Many operators are time dependent and require a time argument. This operator
    is time independent and will pass through the time argument:

    >>> t = u.Quantity(0, "Gyr")
    >>> op(q, t)[1] is t
    True

    """

    translation: AbstractPos = eqx.field(converter=_converter_spatialtranslation)
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

        >>> op = GalileanSpatialTranslation.from_([1, 1, 1], "kpc")

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

        >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "kpc")

        >>> op.inverse
        GalileanSpatialTranslation(CartesianPos3D( ... ))

        >>> print(op.inverse.translation)
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [-1. -1. -1.]>

        """
        return GalileanSpatialTranslation(-self.translation)

    # -------------------------------------------

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(self: "GalileanSpatialTranslation", q: AbstractPos, /) -> AbstractPos:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> shift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> op = cx.ops.GalileanSpatialTranslation(shift)

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> t = u.Quantity(0, "Gyr")
        >>> newq = op(q)
        >>> print(newq)
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [2. 3. 4.]>

        """
        return q + self.translation

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(
        self: "GalileanSpatialTranslation",
        q: AbstractPos,
        t: u.Quantity["time"],
        /,
    ) -> tuple[AbstractPos, u.Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> shift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> op = cx.ops.GalileanSpatialTranslation(shift)

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> t = u.Quantity(0, "Gyr")
        >>> newq, newt = op(q, t)
        >>> print(newq)
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [2. 3. 4.]>

        This spatial translation is time independent.

        >>> op(q, u.Quantity(1, "Gyr"))[0].x == newq.x
        Array(True, dtype=bool)

        """
        return q + self.translation, t

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(self: "GalileanSpatialTranslation", v4: FourVector, /) -> AbstractPos:
        """Apply the translation to the coordinates."""  # TODO: docstring
        return replace(v4, q=v4.q + self.translation)

    # -------------------------------------------
    # Arithmetic operations

    def __neg__(self: "GalileanSpatialTranslation") -> "GalileanSpatialTranslation":
        """Negate the rotation.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "kpc")
        >>> print((-op).translation)
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [-1. -0. -0.]>

        """
        return replace(self, translation=-self.translation)

    # -------------------------------------------
    # Python special methods

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.translation!r})"


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

    >>> op1 = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "kpc")
    >>> op2 = cx.ops.GalileanSpatialTranslation.from_([0, 1, 0], "kpc")

    >>> op3 = cx.ops.simplify_op(op1, op2)
    >>> op3
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    >>> op3.translation == op1.translation + op2.translation
    Array(True, dtype=bool)

    """
    return GalileanSpatialTranslation(op1.translation + op2.translation)


##############################################################################


@final
class GalileanTranslation(AbstractGalileanOperator):
    r"""Operator for spatio-temporal translations.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t+s, \mathbf{x} + \mathbf {a})

    where :math:`a \in R^3` and :math:`s \in R`.  Therefore for a potential
    :math:`\Phi(t,\mathbf{x})` in the translated frame the potential is given by
    the subtraction of the translation.

    Parameters
    ----------
    translation : :class:`vector.FourVector`
        The translation vector [T, Q].  This parameters uses
        :meth:`vector.FourVector.from_` to enable a variety of more
        convenient input types. See :class:`vector.FourVector` for details.

    Examples
    --------
    We start with the required imports:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a translation operator:

    >>> op = cx.ops.GalileanTranslation.from_([1.0, 2.0, 3.0, 4.0], "kpc")
    >>> op
    GalileanTranslation(FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("kpc s / km")),
        q=CartesianPos3D( ... ) ))

    Note that the translation is a :class:`vector.FourVector`, which was
    constructed from a 1D array, using :meth:`vector.FourVector.from_`. We
    can also construct it directly, which allows for other vector types.

    >>> qshift = cx.SphericalPos(r=u.Quantity(1.0, "kpc"),
    ...                          theta=u.Quantity(jnp.pi/2, "rad"),
    ...                          phi=u.Quantity(0, "rad"))
    >>> shift = cx.FourVector(u.Quantity(1.0, "Gyr"), qshift)
    >>> op = cx.ops.GalileanTranslation(shift)
    >>> op
    GalileanTranslation(FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("Gyr")),
        q=SphericalPos( ... ) ))

    Translation operators can be applied to :class:`vector.FourVector`:

    >>> w = cx.FourVector.from_([0, 0, 0, 0], "kpc")
    >>> op(w)
    FourVector(
      t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("kpc s / km")),
      q=CartesianPos3D( ... )
    )

    Also to :class:`vector.AbstractPos3D` and :class:`unxt.Quantity`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> newq, newt = op(q, t)
    >>> newq.x
    Quantity['length'](Array(1., dtype=float32), unit='kpc')
    >>> newt
    Quantity['time'](Array(1., dtype=float32), unit='Gyr')

    """

    translation: FourVector = eqx.field(converter=FourVector.from_)
    """The temporal + spatial translation.

    The translation vector [T, Q].  This parameters uses
    :meth:`vector.FourVector.from_` to enable a variety of more convenient
    input types. See :class:`vector.FourVector` for details.
    """

    # -------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean translation is an inertial-frame preserving transformation.

        Examples
        --------
        >>> import coordinax as cx

        >>> shift = cx.FourVector.from_([0, 1, 1, 1], "kpc")
        >>> op = cx.ops.GalileanTranslation(shift)

        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanTranslation":
        """The inverse of the operator.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> shift = FourVector(u.Quantity(1, "Gyr"), qshift)
        >>> op = cx.ops.GalileanTranslation(shift)

        >>> op.inverse
        GalileanTranslation(FourVector( ... ))

        >>> op.inverse.translation.q.x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return GalileanTranslation(-self.translation)

    # -------------------------------------------

    @AbstractOperator.__call__.dispatch
    def __call__(self: "GalileanTranslation", x: FourVector, /) -> FourVector:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.ops as cxo

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> shift = FourVector(u.Quantity(1, "Gyr"), qshift)
        >>> op = cx.ops.GalileanTranslation(shift)

        Construct a vector to translate, using the convenience from_ (the
        0th component is :math:`c * t`, the rest are spatial components):

        >>> w = cx.FourVector.from_([0, 1, 2, 3], "kpc")
        >>> w.t
        Quantity['time'](Array(0., dtype=float32), unit='kpc s / km')

        Apply the translation operator:

        >>> new = op(w)
        >>> new.x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        >>> new.t.uconvert("Gyr")
        Quantity['time'](Array(0.99999994, dtype=float32), unit='Gyr')

        """
        return x + self.translation

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(
        self: "GalileanTranslation",
        x: AbstractPos3D,
        t: u.Quantity["time"],
        /,
    ) -> tuple[AbstractPos3D, u.Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.ops as cxo

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> tshift = u.Quantity(1, "Gyr")
        >>> shift = cx.FourVector(tshift, qshift)
        >>> op = cx.ops.GalileanTranslation(shift)

        Construct a vector to translate

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> t = u.Quantity(1, "Gyr")
        >>> newq, newt = op(q, t)

        >>> newq.x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        >>> newt
        Quantity['time'](Array(2., dtype=float32), unit='Gyr')

        """
        return (x + self.translation.q, t + self.translation.t)

    # -------------------------------------------
    # Python special methods

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.translation!r})"


@dispatch
def simplify_op(
    op: GalileanTranslation, /, **kwargs: Any
) -> GalileanTranslation | Identity:
    """Simplify a Galilean translation operator.

    Examples
    --------
    >>> import coordinax as cx

    An operator with real effect cannot be simplified:

    >>> op = cx.ops.GalileanTranslation.from_([3e8, 1, 0, 0], "m")
    >>> cx.ops.simplify_op(op)
    GalileanTranslation(FourVector( ... ))

    An operator with no effect can be simplified:

    >>> op = cx.ops.GalileanTranslation.from_([0, 0, 0, 0], "m")
    >>> cx.ops.simplify_op(op)
    Identity()

    """
    # Check if the translation is zero.
    if jnp.allclose(
        convert(op.translation, u.Quantity).value, jnp.zeros((4,)), **kwargs
    ):
        return Identity()
    # TODO: Check if the translation is purely spatial.

    return op


# TODO: show op3.translation = op1.translation + op2.translation
@dispatch
def simplify_op(
    op1: GalileanTranslation, op2: GalileanTranslation, /
) -> GalileanTranslation:
    """Combine two translations into a single translation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> qshift = cx.CartesianPos3D.from_([1, 0, 0], "kpc")
    >>> tshift = u.Quantity(1, "Gyr")
    >>> op1 = cx.ops.GalileanTranslation(FourVector(tshift, qshift))

    >>> qshift = cx.CartesianPos3D.from_([0, 1, 0], "kpc")
    >>> tshift = u.Quantity(1, "Gyr")
    >>> op2 = cx.ops.GalileanTranslation(FourVector(tshift, qshift))

    >>> op3 = cx.ops.simplify_op(op1, op2)
    >>> op3
    GalileanTranslation(FourVector( ... ))

    """
    return GalileanTranslation(op1.translation + op2.translation)
