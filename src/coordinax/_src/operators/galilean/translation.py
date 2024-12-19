# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanTranslation"]


from typing import Any, Literal, final

import equinox as eqx
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractGalileanOperator
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors.d3 import AbstractPos3D
from coordinax._src.vectors.d4 import FourVector


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

    >>> op = cx.ops.GalileanTranslation.from_([1.0, 2.0, 3.0, 4.0], "km")
    >>> op
    GalileanTranslation(FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
        q=CartesianPos3D( ... ) ))

    Note that the translation is a :class:`vector.FourVector`, which was
    constructed from a 1D array, using :meth:`vector.FourVector.from_`. We
    can also construct it directly, which allows for other vector types.

    >>> qshift = cx.SphericalPos(r=u.Quantity(1.0, "km"),
    ...                          theta=u.Quantity(jnp.pi/2, "rad"),
    ...                          phi=u.Quantity(0, "rad"))
    >>> shift = cx.FourVector(u.Quantity(1.0, "Gyr"), qshift)
    >>> op = cx.ops.GalileanTranslation(shift)
    >>> op
    GalileanTranslation(FourVector(
        t=Quantity[PhysicalType('time')](value=...f32[], unit=Unit("Gyr")),
        q=SphericalPos( ... ) ))

    Translation operators can be applied to :class:`vector.FourVector`:

    >>> w = cx.FourVector.from_([0, 0, 0, 0], "km")
    >>> op(w)
    FourVector(
      t=Quantity[PhysicalType('time')](value=...f32[], unit=Unit("s")),
      q=CartesianPos3D( ... )
    )

    Also to :class:`vector.AbstractPos3D` and :class:`unxt.Quantity`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> t = u.Quantity(0, "Gyr")
    >>> newq, newt = op(q, t)
    >>> newq.x
    Quantity['length'](Array(1., dtype=float32, ...), unit='km')
    >>> newt
    Quantity['time'](Array(1., dtype=float32, ...), unit='Gyr')

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

        >>> shift = cx.FourVector.from_([0, 1, 1, 1], "km")
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

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
        >>> shift = FourVector(u.Quantity(1, "Gyr"), qshift)
        >>> op = cx.ops.GalileanTranslation(shift)

        >>> op.inverse
        GalileanTranslation(FourVector( ... ))

        >>> op.inverse.translation.q.x
        Quantity['length'](Array(-1, dtype=int32), unit='km')

        """
        return GalileanTranslation(-self.translation)

    # -------------------------------------------

    @AbstractOperator.__call__.dispatch
    def __call__(
        self: "GalileanTranslation", x: FourVector, /, **__: Any
    ) -> FourVector:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.ops as cxo

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
        >>> shift = FourVector(u.Quantity(1, "Gyr"), qshift)
        >>> op = cx.ops.GalileanTranslation(shift)

        Construct a vector to translate, using the convenience from_ (the
        0th component is :math:`c * t`, the rest are spatial components):

        >>> w = cx.FourVector.from_([0, 1, 2, 3], "km")
        >>> w.t
        Quantity['time'](Array(0., dtype=float32, ...), unit='s')

        Apply the translation operator:

        >>> new = op(w)
        >>> new.x
        Quantity['length'](Array(2, dtype=int32), unit='km')

        >>> new.t.uconvert("Gyr")
        Quantity['time'](Array(1., dtype=float32, ...), unit='Gyr')

        """
        return x + self.translation

    @AbstractOperator.__call__.dispatch
    def __call__(
        self: "GalileanTranslation",
        x: AbstractPos3D,
        t: u.Quantity["time"],
        /,
        **__: Any,
    ) -> tuple[AbstractPos3D, u.Quantity["time"]]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.ops as cxo

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
        >>> tshift = u.Quantity(1, "Gyr")
        >>> shift = cx.FourVector(tshift, qshift)
        >>> op = cx.ops.GalileanTranslation(shift)

        Construct a vector to translate

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> t = u.Quantity(1, "Gyr")
        >>> newq, newt = op(q, t)

        >>> newq.x
        Quantity['length'](Array(2, dtype=int32), unit='km')

        >>> newt
        Quantity['time'](Array(2, dtype=int32, ...), unit='Gyr')

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

    >>> qshift = cx.CartesianPos3D.from_([1, 0, 0], "km")
    >>> tshift = u.Quantity(1, "Gyr")
    >>> op1 = cx.ops.GalileanTranslation(FourVector(tshift, qshift))

    >>> qshift = cx.CartesianPos3D.from_([0, 1, 0], "km")
    >>> tshift = u.Quantity(1, "Gyr")
    >>> op2 = cx.ops.GalileanTranslation(FourVector(tshift, qshift))

    >>> op3 = cx.ops.simplify_op(op1, op2)
    >>> op3
    GalileanTranslation(FourVector( ... ))

    """
    return GalileanTranslation(op1.translation + op2.translation)
