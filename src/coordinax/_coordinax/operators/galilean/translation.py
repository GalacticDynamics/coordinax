# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanSpatialTranslationOperator", "GalileanTranslationOperator"]


from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
import jax.numpy as jnp
from plum import convert

import quaxed.array_api as xp
from unxt import Quantity

from .base import AbstractGalileanOperator
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.d1.cartesian import CartesianPosition1D
from coordinax._coordinax.d2.cartesian import CartesianPosition2D
from coordinax._coordinax.d3.base import AbstractPosition3D
from coordinax._coordinax.d3.cartesian import CartesianPosition3D
from coordinax._coordinax.d4.spacetime import FourVector
from coordinax._coordinax.operators.base import AbstractOperator, op_call_dispatch
from coordinax._coordinax.operators.funcs import simplify_op
from coordinax._coordinax.operators.identity import IdentityOperator

##############################################################################
# Spatial Translations


def _converter_spatialtranslation(x: Any) -> AbstractPosition:
    """Convert to a spatial translation vector."""
    out: AbstractPosition | None = None
    if isinstance(x, GalileanSpatialTranslationOperator):
        out = x.translation
    elif isinstance(x, AbstractPosition):
        out = x
    elif isinstance(x, Quantity):
        shape: tuple[int, ...] = x.shape
        match shape:
            case (1,):
                out = CartesianPosition1D.constructor(x)
            case (2,):
                out = CartesianPosition2D.constructor(x)
            case (3,):
                out = CartesianPosition3D.constructor(x)
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
    translation : :class:`vector.AbstractPosition3D`
        The spatial translation vector. This parameters accepts either a
        :class:`vector.AbstractPosition3D` instance or uses
        :meth:`vector.CartesianPosition3D.constructor` to enable a variety of more
        convenient input types to create a Cartesian vector. See
        :class:`vector.CartesianPosition3D` for details.

    Examples
    --------
    We start with the required imports:

    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> shift = Quantity([1.0, 2.0, 3.0], "kpc")
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)
    >>> op
    GalileanSpatialTranslationOperator( translation=CartesianPosition3D( ... ) )

    Note that the translation is a :class:`vector.CartesianPosition3D`, which was
    constructed from a 1D array, using
    :meth:`vector.CartesianPosition3D.constructor`. We can also construct it
    directly, which allows for other vector types.

    >>> shift = cx.SphericalPosition(r=Quantity(1.0, "kpc"),
    ...                              theta=Quantity(xp.pi/2, "rad"),
    ...                              phi=Quantity(0, "rad"))
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)
    >>> op
    GalileanSpatialTranslationOperator( translation=SphericalPosition( ... ) )

    Translation operators can be applied to :class:`vector.AbstractPosition`:

    >>> q = cx.CartesianPosition3D.constructor([0, 0, 0], "kpc")
    >>> op(q)
    CartesianPosition3D( ... )

    And to :class:`~unxt.Quantity`:

    >>> q = Quantity([0, 0, 0], "kpc")
    >>> op(q).value.round(2)
    Array([ 1.,  0., -0.], dtype=float32)

    :class:`coordinax.operators.GalileanSpatialTranslationOperator` can be used
    for other dimensional vectors as well:

    - 1D:

    >>> shift = Quantity([1], "kpc")
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)
    >>> q = Quantity([0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1.], dtype=float32), unit='kpc')

    >>> vec = cx.CartesianPosition1D.constructor(q).represent_as(cx.RadialPosition)
    >>> op(vec)
    RadialPosition(r=Distance(value=f32[], unit=Unit("kpc")))

    - 2D:

    >>> shift = Quantity([1, 2], "kpc")
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)
    >>> q = Quantity([0, 0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='kpc')

    >>> vec = cx.CartesianPosition2D.constructor(q).represent_as(cx.PolarPosition)
    >>> op(vec)
    PolarPosition( r=Distance(value=f32[], unit=Unit("kpc")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )

    - 3D:

    >>> shift = Quantity([1, 2, 3], "kpc")
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)
    >>> q = Quantity([0, 0, 0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')

    >>> vec = cx.CartesianPosition3D.constructor(q).represent_as(cx.SphericalPosition)
    >>> op(vec)
    SphericalPosition( r=Distance(value=f32[], unit=Unit("kpc")),
                     theta=Quantity[...](value=f32[], unit=Unit("rad")),
                     phi=Quantity[...](value=f32[], unit=Unit("rad")) )

    Many operators are time dependent and require a time argument. This operator
    is time independent and will pass through the time argument:

    >>> t = Quantity(0, "Gyr")
    >>> op(q, t)[1] is t
    True

    """

    translation: AbstractPosition = eqx.field(converter=_converter_spatialtranslation)
    """The spatial translation.

    This parameters accepts either a :class:`vector.AbstracVector` instance or
    uses a Cartesian vector constructor to enable a variety of more convenient
    input types to create a Cartesian vector. See
    :class:`vector.CartesianPosition3D.constructor` for an example when doing a 3D
    translation.
    """

    # -------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean translation is an inertial frame-preserving transformation.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanSpatialTranslationOperator

        >>> shift = cx.CartesianPosition3D.constructor([1, 1, 1], "kpc")
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanSpatialTranslationOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanSpatialTranslationOperator

        >>> shift = cx.CartesianPosition3D.constructor([1, 1, 1], "kpc")
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> op.inverse
        GalileanSpatialTranslationOperator( translation=CartesianPosition3D( ... ) )

        >>> op.inverse.translation.x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return GalileanSpatialTranslationOperator(-self.translation)

    # -------------------------------------------

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanSpatialTranslationOperator", q: AbstractPosition, /
    ) -> AbstractPosition:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanSpatialTranslationOperator

        >>> shift = cx.CartesianPosition3D.constructor([1, 1, 1], "kpc")
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
        >>> t = Quantity(0, "Gyr")
        >>> newq = op(q)
        >>> newq.x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        """
        return q + self.translation

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanSpatialTranslationOperator",
        q: AbstractPosition,
        t: Quantity["time"],
        /,
    ) -> tuple[AbstractPosition, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanSpatialTranslationOperator

        >>> shift = cx.CartesianPosition3D.constructor([1, 1, 1], "kpc")
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
        >>> t = Quantity(0, "Gyr")
        >>> newq, newt = op(q, t)
        >>> newq.x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        This spatial translation is time independent.

        >>> op(q, Quantity(1, "Gyr"))[0].x == newq.x
        Array(True, dtype=bool)

        """
        return q + self.translation, t

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanSpatialTranslationOperator", v4: FourVector, /
    ) -> AbstractPosition:
        """Apply the translation to the coordinates."""  # TODO: docstring
        return replace(v4, q=v4.q + self.translation)


@simplify_op.register
def _simplify_op_spatialtranslation(
    op: GalileanSpatialTranslationOperator, /, **kwargs: Any
) -> AbstractOperator:
    """Simplify a spatial translation operator."""
    # Check if the translation is zero.
    if jnp.allclose(convert(op.translation, Quantity).value, xp.zeros((3,)), **kwargs):
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
        :meth:`vector.FourVector.constructor` to enable a variety of more
        convenient input types. See :class:`vector.FourVector` for details.

    Examples
    --------
    We start with the required imports:

    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import coordinax.operators as co

    We can then create a translation operator:

    >>> op = GalileanTranslationOperator(Quantity([1.0, 2.0, 3.0, 4.0], "kpc"))
    >>> op
    GalileanTranslationOperator(
      translation=FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("kpc s / km")),
        q=CartesianPosition3D( ... ) )
    )

    Note that the translation is a :class:`vector.FourVector`, which was
    constructed from a 1D array, using :meth:`vector.FourVector.constructor`. We
    can also construct it directly, which allows for other vector types.

    >>> qshift = cx.SphericalPosition(r=Quantity(1.0, "kpc"),
    ...                               theta=Quantity(xp.pi/2, "rad"),
    ...                               phi=Quantity(0, "rad"))
    >>> op = GalileanTranslationOperator(FourVector(t=Quantity(1.0, "Gyr"), q=qshift))
    >>> op
    GalileanTranslationOperator(
      translation=FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("Gyr")),
        q=SphericalPosition( ... ) )
    )

    Translation operators can be applied to :class:`vector.FourVector`:

    >>> w = FourVector.constructor([0, 0, 0, 0], "kpc")
    >>> op(w)
    FourVector(
      t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("kpc s / km")),
      q=CartesianPosition3D( ... )
    )

    Also to :class:`vector.AbstractPosition3D` and :class:`unxt.Quantity`:

    >>> q = cx.CartesianPosition3D.constructor([0, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")
    >>> newq, newt = op(q, t)
    >>> newq.x
    Quantity['length'](Array(1., dtype=float32), unit='kpc')
    >>> newt
    Quantity['time'](Array(1., dtype=float32), unit='Gyr')

    """

    translation: FourVector = eqx.field(converter=FourVector.constructor)
    """The temporal + spatial translation.

    The translation vector [T, Q].  This parameters uses
    :meth:`vector.FourVector.constructor` to enable a variety of more convenient
    input types. See :class:`vector.FourVector` for details.
    """

    # -------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean translation is an inertial-frame preserving transformation.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanTranslationOperator

        >>> shift = cx.FourVector.constructor([0, 1, 1, 1], "kpc")
        >>> op = GalileanTranslationOperator(shift)

        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanTranslationOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanSpatialTranslationOperator

        >>> qshift = cx.CartesianPosition3D.constructor([1, 1, 1], "kpc")
        >>> tshift = Quantity(1, "Gyr")
        >>> shift = FourVector(tshift, qshift)
        >>> op = GalileanTranslationOperator(shift)

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
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanTranslationOperator

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPosition3D.constructor([1, 1, 1], "kpc")
        >>> tshift = Quantity(1, "Gyr")
        >>> shift = FourVector(tshift, qshift)
        >>> op = GalileanTranslationOperator(shift)

        Construct a vector to translate, using the convenience constructor (the
        0th component is :math:`c * t`, the rest are spatial components):

        >>> w = cx.FourVector.constructor([0, 1, 2, 3], "kpc")
        >>> w.t
        Quantity['time'](Array(0., dtype=float32), unit='kpc s / km')

        Apply the translation operator:

        >>> new = op(w)
        >>> new.x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        >>> new.t.to_units("Gyr")
        Quantity['time'](Array(0.99999994, dtype=float32), unit='Gyr')

        """
        return x + self.translation

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanTranslationOperator",
        x: AbstractPosition3D,
        t: Quantity["time"],
        /,
    ) -> tuple[AbstractPosition3D, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanTranslationOperator

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPosition3D.constructor([1, 1, 1], "kpc")
        >>> tshift = Quantity(1, "Gyr")
        >>> shift = cx.FourVector(tshift, qshift)
        >>> op = GalileanTranslationOperator(shift)

        Construct a vector to translate

        >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
        >>> t = Quantity(1, "Gyr")
        >>> newq, newt = op(q, t)

        >>> newq.x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        >>> newt
        Quantity['time'](Array(2., dtype=float32), unit='Gyr')

        """
        return (x + self.translation.q, t + self.translation.t)


@simplify_op.register
def _simplify_op_tranlation(
    op: GalileanTranslationOperator, /, **kwargs: Any
) -> AbstractOperator:
    """Simplify a translation operator."""
    # Check if the translation is zero.
    if jnp.allclose(convert(op.translation, Quantity).value, xp.zeros((4,)), **kwargs):
        return IdentityOperator()
    # Check if the translation is purely spatial.
    if op.translation[0] == 0:
        return GalileanSpatialTranslationOperator(op.translation[1:])
    return op
