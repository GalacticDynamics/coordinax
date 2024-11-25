# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanSpatialTranslationOperator", "GalileanTranslationOperator"]


from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractGalileanOperator
from coordinax._src.d4.spacetime import FourVector
from coordinax._src.operators.base import AbstractOperator, op_call_dispatch
from coordinax._src.operators.identity import IdentityOperator
from coordinax._src.vectors.base import AbstractPos
from coordinax._src.vectors.d1 import CartesianPos1D
from coordinax._src.vectors.d2 import CartesianPos2D
from coordinax._src.vectors.d3 import AbstractPos3D, CartesianPos3D

##############################################################################
# Spatial Translations


def _converter_spatialtranslation(x: Any) -> AbstractPos:
    """Convert to a spatial translation vector."""
    out: AbstractPos | None = None
    if isinstance(x, GalileanSpatialTranslationOperator):
        out = x.translation
    elif isinstance(x, AbstractPos):
        out = x
    elif isinstance(x, u.Quantity):
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
class GalileanSpatialTranslationOperator(AbstractGalileanOperator):
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
    >>> import coordinax.operators as cxo

    We can then create a spatial translation operator:

    >>> shift = u.Quantity([1.0, 2.0, 3.0], "kpc")
    >>> op = cxo.GalileanSpatialTranslationOperator(shift)
    >>> op
    GalileanSpatialTranslationOperator( translation=CartesianPos3D( ... ) )

    Note that the translation is a :class:`vector.CartesianPos3D`, which was
    constructed from a 1D array, using :meth:`vector.CartesianPos3D.from_`. We
    can also construct it directly, which allows for other vector types.

    >>> shift = cx.SphericalPos(r=u.Quantity(1.0, "kpc"),
    ...                         theta=u.Quantity(jnp.pi/2, "rad"),
    ...                         phi=u.Quantity(0, "rad"))
    >>> op = cxo.GalileanSpatialTranslationOperator(shift)
    >>> op
    GalileanSpatialTranslationOperator( translation=SphericalPos( ... ) )

    Translation operators can be applied to :class:`vector.AbstractPos`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "kpc")
    >>> op(q)
    CartesianPos3D( ... )

    And to :class:`~unxt.Quantity`:

    >>> q = u.Quantity([0, 0, 0], "kpc")
    >>> op(q).value.round(2)
    Array([ 1.,  0., -0.], dtype=float32)

    :class:`coordinax.operators.GalileanSpatialTranslationOperator` can be used
    for other dimensional vectors as well:

    - 1D:

    >>> op = cxo.GalileanSpatialTranslationOperator.from_([1], "kpc")
    >>> q = u.Quantity([0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1.], dtype=float32), unit='kpc')

    >>> vec = cx.CartesianPos1D.from_(q).represent_as(cx.RadialPos)
    >>> op(vec)
    RadialPos(r=Distance(value=f32[], unit=Unit("kpc")))

    - 2D:

    >>> op = cxo.GalileanSpatialTranslationOperator.from_([1, 2], "kpc")
    >>> q = u.Quantity([0, 0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='kpc')

    >>> vec = cx.CartesianPos2D.from_(q).represent_as(cx.PolarPos)
    >>> op(vec)
    PolarPos( r=Distance(value=f32[], unit=Unit("kpc")),
              phi=Angle(value=f32[], unit=Unit("rad")) )

    - 3D:

    >>> op = cxo.GalileanSpatialTranslationOperator.from_([1, 2, 3], "kpc")
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
        >>> import coordinax.operators as cxo

        >>> op = GalileanSpatialTranslationOperator.from_([1, 1, 1], "kpc")

        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanSpatialTranslationOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> import coordinax.operators as cxo

        >>> op = cxo.GalileanSpatialTranslationOperator.from_([1, 1, 1], "kpc")

        >>> op.inverse
        GalileanSpatialTranslationOperator( translation=CartesianPos3D( ... ) )

        >>> print(op.inverse.translation)
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [-1. -1. -1.]>

        """
        return GalileanSpatialTranslationOperator(-self.translation)

    # -------------------------------------------

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanSpatialTranslationOperator", q: AbstractPos, /
    ) -> AbstractPos:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.operators as cxo

        >>> shift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> op = cxo.GalileanSpatialTranslationOperator(shift)

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> t = u.Quantity(0, "Gyr")
        >>> newq = op(q)
        >>> print(newq)
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [2. 3. 4.]>

        """
        return q + self.translation

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanSpatialTranslationOperator",
        q: AbstractPos,
        t: u.Quantity["time"],
        /,
    ) -> tuple[AbstractPos, u.Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.operators as cxo

        >>> shift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> op = cxo.GalileanSpatialTranslationOperator(shift)

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

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanSpatialTranslationOperator", v4: FourVector, /
    ) -> AbstractPos:
        """Apply the translation to the coordinates."""  # TODO: docstring
        return replace(v4, q=v4.q + self.translation)


@dispatch  # type: ignore[misc]
def simplify_op(
    op: GalileanSpatialTranslationOperator, /, **kwargs: Any
) -> AbstractOperator:
    """Simplify a Galilean spatial translation operator.

    Examples
    --------
    >>> import coordinax.operators as co

    An operator with real effect cannot be simplified:

    >>> op = co.GalileanSpatialTranslationOperator.from_([1, 0, 0], "m")
    >>> co.simplify_op(op)
    GalileanSpatialTranslationOperator(
      translation=CartesianPos3D( ... )
    )

    An operator with no effect can be simplified:

    >>> op = co.GalileanSpatialTranslationOperator.from_([0, 0, 0], "m")
    >>> co.simplify_op(op)
    IdentityOperator()

    """
    # Check if the translation is zero.
    if jnp.allclose(
        convert(op.translation, u.Quantity).value, jnp.zeros((3,)), **kwargs
    ):
        return IdentityOperator()
    return op


##############################################################################


@final
class GalileanTranslationOperator(AbstractGalileanOperator):
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
    >>> import coordinax.operators as cxo

    We can then create a translation operator:

    >>> op = cxo.GalileanTranslationOperator.from_([1.0, 2.0, 3.0, 4.0], "kpc")
    >>> op
    GalileanTranslationOperator(
      translation=FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("kpc s / km")),
        q=CartesianPos3D( ... ) )
    )

    Note that the translation is a :class:`vector.FourVector`, which was
    constructed from a 1D array, using :meth:`vector.FourVector.from_`. We
    can also construct it directly, which allows for other vector types.

    >>> qshift = cx.SphericalPos(r=u.Quantity(1.0, "kpc"),
    ...                          theta=u.Quantity(jnp.pi/2, "rad"),
    ...                          phi=u.Quantity(0, "rad"))
    >>> shift = cx.FourVector(u.Quantity(1.0, "Gyr"), qshift)
    >>> op = cxo.GalileanTranslationOperator(shift)
    >>> op
    GalileanTranslationOperator(
      translation=FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("Gyr")),
        q=SphericalPos( ... ) )
    )

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
        >>> import coordinax.operators as cxo

        >>> shift = cx.FourVector.from_([0, 1, 1, 1], "kpc")
        >>> op = cxo.GalileanTranslationOperator(shift)

        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanTranslationOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.operators as cxo

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> shift = FourVector(u.Quantity(1, "Gyr"), qshift)
        >>> op = cxo.GalileanTranslationOperator(shift)

        >>> op.inverse
        GalileanTranslationOperator( translation=FourVector( ... ) )

        >>> op.inverse.translation.q.x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return GalileanTranslationOperator(-self.translation)

    # -------------------------------------------

    @op_call_dispatch
    def __call__(self: "GalileanTranslationOperator", x: FourVector, /) -> FourVector:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.operators as cxo

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> shift = FourVector(u.Quantity(1, "Gyr"), qshift)
        >>> op = cxo.GalileanTranslationOperator(shift)

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

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanTranslationOperator",
        x: AbstractPos3D,
        t: u.Quantity["time"],
        /,
    ) -> tuple[AbstractPos3D, u.Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.operators as cxo

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> tshift = u.Quantity(1, "Gyr")
        >>> shift = cx.FourVector(tshift, qshift)
        >>> op = cxo.GalileanTranslationOperator(shift)

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


@dispatch  # type: ignore[misc]
def simplify_op(op: GalileanTranslationOperator, /, **kwargs: Any) -> AbstractOperator:
    """Simplify a Galilean translation operator.

    Examples
    --------
    >>> import coordinax.operators as co

    An operator with real effect cannot be simplified:

    >>> op = co.GalileanTranslationOperator.from_([3e8, 1, 0, 0], "m")
    >>> co.simplify_op(op)
    GalileanTranslationOperator(
      translation=FourVector( ... )
    )

    An operator with no effect can be simplified:

    >>> op = co.GalileanTranslationOperator.from_([0, 0, 0, 0], "m")
    >>> co.simplify_op(op)
    IdentityOperator()

    """
    # Check if the translation is zero.
    if jnp.allclose(
        convert(op.translation, u.Quantity).value, jnp.zeros((4,)), **kwargs
    ):
        return IdentityOperator()
    # TODO: Check if the translation is purely spatial.

    return op
