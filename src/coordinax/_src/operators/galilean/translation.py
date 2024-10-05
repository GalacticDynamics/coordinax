# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanSpatialTranslationOperator", "GalileanTranslationOperator"]


from dataclasses import replace
from typing import Any, Literal, final
from typing_extensions import override

import equinox as eqx
from jaxtyping import ArrayLike
from plum import convert, dispatch

import quaxed.numpy as jnp
from unxt import Quantity

from .base import AbstractGalileanOperator
from coordinax._src.base import AbstractPos
from coordinax._src.d1.cartesian import CartesianPos1D
from coordinax._src.d2.cartesian import CartesianPos2D
from coordinax._src.d3.base import AbstractPos3D
from coordinax._src.d3.cartesian import CartesianPos3D
from coordinax._src.d4.spacetime import FourVector
from coordinax._src.operators.base import AbstractOperator, op_call_dispatch
from coordinax._src.operators.funcs import simplify_op
from coordinax._src.operators.identity import IdentityOperator

##############################################################################
# Spatial Translations


def _converter_spatialtranslation(x: Any) -> AbstractPos:
    """Convert to a spatial translation vector."""
    out: AbstractPos | None = None
    if isinstance(x, GalileanSpatialTranslationOperator):
        out = x.translation
    elif isinstance(x, AbstractPos):
        out = x
    elif isinstance(x, Quantity):
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
    We start with the required imports:

    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    We can then create a spatial translation operator:

    >>> shift = Quantity([1.0, 2.0, 3.0], "kpc")
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)
    >>> op
    GalileanSpatialTranslationOperator( translation=CartesianPos3D( ... ) )

    Note that the translation is a :class:`vector.CartesianPos3D`, which was
    constructed from a 1D array, using :meth:`vector.CartesianPos3D.from_`. We
    can also construct it directly, which allows for other vector types.

    >>> shift = cx.SphericalPos(r=Quantity(1.0, "kpc"),
    ...                              theta=Quantity(jnp.pi/2, "rad"),
    ...                              phi=Quantity(0, "rad"))
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)
    >>> op
    GalileanSpatialTranslationOperator( translation=SphericalPos( ... ) )

    Translation operators can be applied to :class:`vector.AbstractPos`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "kpc")
    >>> op(q)
    CartesianPos3D( ... )

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

    >>> vec = cx.CartesianPos1D.from_(q).represent_as(cx.RadialPos)
    >>> op(vec)
    RadialPos(r=Distance(value=f32[], unit=Unit("kpc")))

    - 2D:

    >>> shift = Quantity([1, 2], "kpc")
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)
    >>> q = Quantity([0, 0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='kpc')

    >>> vec = cx.CartesianPos2D.from_(q).represent_as(cx.PolarPos)
    >>> op(vec)
    PolarPos( r=Distance(value=f32[], unit=Unit("kpc")),
                 phi=Quantity[...](value=f32[], unit=Unit("rad")) )

    - 3D:

    >>> shift = Quantity([1, 2, 3], "kpc")
    >>> op = cx.operators.GalileanSpatialTranslationOperator(shift)
    >>> q = Quantity([0, 0, 0], "kpc")
    >>> op(q)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')

    >>> vec = cx.CartesianPos3D.from_(q).represent_as(cx.SphericalPos)
    >>> op(vec)
    SphericalPos( r=Distance(value=f32[], unit=Unit("kpc")),
                     theta=Quantity[...](value=f32[], unit=Unit("rad")),
                     phi=Quantity[...](value=f32[], unit=Unit("rad")) )

    Many operators are time dependent and require a time argument. This operator
    is time independent and will pass through the time argument:

    >>> t = Quantity(0, "Gyr")
    >>> op(q, t)[1] is t
    True

    """

    translation: AbstractPos = eqx.field(converter=_converter_spatialtranslation)
    """The spatial translation.

    This parameters accepts either a :class:`vector.AbstracVector` instance or
    uses a Cartesian vector from_ to enable a variety of more convenient
    input types to create a Cartesian vector. See
    :class:`vector.CartesianPos3D.from_` for an example when doing a 3D
    translation.
    """

    # -------------------------------------------

    @override
    @classmethod
    @dispatch
    def from_(
        cls: "type[GalileanSpatialTranslationOperator]",
        x: ArrayLike | list[float | int],
        unit: str,  # TODO: support unit object
        /,
    ) -> "GalileanSpatialTranslationOperator":
        """Construct a spatial translation operator.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> op = cx.operators.GalileanSpatialTranslationOperator.from_([1, 1, 1], "kpc")
        >>> op.translation.x
        Quantity['length'](Array(1., dtype=float32), unit='kpc')

        """
        return cls(Quantity(x, unit))

    @override
    @classmethod
    @dispatch
    def from_(
        cls: "type[GalileanSpatialTranslationOperator]",
        x: ArrayLike | list[float | int],
        *,
        unit: Any,
    ) -> "GalileanSpatialTranslationOperator":
        """Construct a spatial translation operator.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> op = cx.operators.GalileanSpatialTranslationOperator.from_([1, 1, 1], "kpc")
        >>> op.translation.x
        Quantity['length'](Array(1., dtype=float32), unit='kpc')

        """
        return cls(Quantity(x, unit))

    # -------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean translation is an inertial frame-preserving transformation.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanSpatialTranslationOperator

        >>> shift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
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

        >>> shift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> op.inverse
        GalileanSpatialTranslationOperator( translation=CartesianPos3D( ... ) )

        >>> op.inverse.translation.x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

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
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanSpatialTranslationOperator

        >>> shift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> t = Quantity(0, "Gyr")
        >>> newq = op(q)
        >>> newq.x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        """
        return q + self.translation

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanSpatialTranslationOperator",
        q: AbstractPos,
        t: Quantity["time"],
        /,
    ) -> tuple[AbstractPos, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanSpatialTranslationOperator

        >>> shift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> op = GalileanSpatialTranslationOperator(shift)

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
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
    ) -> AbstractPos:
        """Apply the translation to the coordinates."""  # TODO: docstring
        return replace(v4, q=v4.q + self.translation)


@simplify_op.register
def _simplify_op_spatialtranslation(
    op: GalileanSpatialTranslationOperator, /, **kwargs: Any
) -> AbstractOperator:
    """Simplify a spatial translation operator."""
    # Check if the translation is zero.
    if jnp.allclose(convert(op.translation, Quantity).value, jnp.zeros((3,)), **kwargs):
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
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import coordinax.operators as co

    We can then create a translation operator:

    >>> op = GalileanTranslationOperator(Quantity([1.0, 2.0, 3.0, 4.0], "kpc"))
    >>> op
    GalileanTranslationOperator(
      translation=FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("kpc s / km")),
        q=CartesianPos3D( ... ) )
    )

    Note that the translation is a :class:`vector.FourVector`, which was
    constructed from a 1D array, using :meth:`vector.FourVector.from_`. We
    can also construct it directly, which allows for other vector types.

    >>> qshift = cx.SphericalPos(r=Quantity(1.0, "kpc"),
    ...                               theta=Quantity(jnp.pi/2, "rad"),
    ...                               phi=Quantity(0, "rad"))
    >>> op = GalileanTranslationOperator(FourVector(t=Quantity(1.0, "Gyr"), q=qshift))
    >>> op
    GalileanTranslationOperator(
      translation=FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("Gyr")),
        q=SphericalPos( ... ) )
    )

    Translation operators can be applied to :class:`vector.FourVector`:

    >>> w = FourVector.from_([0, 0, 0, 0], "kpc")
    >>> op(w)
    FourVector(
      t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("kpc s / km")),
      q=CartesianPos3D( ... )
    )

    Also to :class:`vector.AbstractPos3D` and :class:`unxt.Quantity`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")
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
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanTranslationOperator

        >>> shift = cx.FourVector.from_([0, 1, 1, 1], "kpc")
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

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
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

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> tshift = Quantity(1, "Gyr")
        >>> shift = FourVector(tshift, qshift)
        >>> op = GalileanTranslationOperator(shift)

        Construct a vector to translate, using the convenience from_ (the
        0th component is :math:`c * t`, the rest are spatial components):

        >>> w = cx.FourVector.from_([0, 1, 2, 3], "kpc")
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
        x: AbstractPos3D,
        t: Quantity["time"],
        /,
    ) -> tuple[AbstractPos3D, Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import GalileanTranslationOperator

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "kpc")
        >>> tshift = Quantity(1, "Gyr")
        >>> shift = cx.FourVector(tshift, qshift)
        >>> op = GalileanTranslationOperator(shift)

        Construct a vector to translate

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
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
    if jnp.allclose(convert(op.translation, Quantity).value, jnp.zeros((4,)), **kwargs):
        return IdentityOperator()
    # Check if the translation is purely spatial.
    if op.translation[0] == 0:
        return GalileanSpatialTranslationOperator(op.translation[1:])
    return op
