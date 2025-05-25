# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanTranslation"]


from typing import Any, Literal, cast, final

import equinox as eqx
import wadler_lindig as wl
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
    translation
        The translation vector [T, Q].  This parameters uses
        `coordinax.vecs.FourVector.from_` to enable a variety of more convenient
        input types. See `coordinax.vecs.FourVector` for details.

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
      t=Quantity(f32[], unit='s'), q=CartesianPos3D( ... )
    ))

    Note that the translation is a `coordinax.vecs.FourVector`, which was
    constructed from a 1D array, using :meth:`coordinax.vecs.FourVector.from_`.
    We can also construct it directly, which allows for other vector types.

    >>> qshift = cx.SphericalPos(r=u.Quantity(1.0, "km"),
    ...                          theta=u.Quantity(jnp.pi/2, "rad"),
    ...                          phi=u.Quantity(0, "rad"))
    >>> shift = cx.FourVector(u.Quantity(1.0, "Gyr"), qshift)
    >>> op = cx.ops.GalileanTranslation(shift)
    >>> op
    GalileanTranslation(FourVector(
      t=Quantity(weak_f32[], unit='Gyr'), q=SphericalPos( ... )
    ))

    Translation operators can be applied to `coordinax.vecs.FourVector`:

    >>> w = cx.FourVector.from_([0, 0, 0, 0], "km")
    >>> print(op(w))
    <FourVector: (t[s], q=(x, y, z) [km])
        [ 3.156e+16  1.000e+00  0.000e+00 -4.371e-08]>

    Also to `vector.AbstractPos3D` and `unxt.Quantity`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> t = u.Quantity(0, "Gyr")
    >>> newt, newq = op(t, q)
    >>> newq.x
    Quantity(Array(1., dtype=float32, ...), unit='km')
    >>> newt
    Quantity(Array(1., dtype=float32, ...), unit='Gyr')

    """

    translation: FourVector = eqx.field(converter=FourVector.from_)
    """The temporal + spatial translation.

    The translation vector [T, Q].  This parameters uses
    :meth:`coordinax.vecs.FourVector.from_` to enable a variety of more convenient
    input types. See `coordinax.vecs.FourVector` for details.
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
        >>> shift = FourVector (u.Quantity(1, "Gyr"), qshift)
        >>> op = cx.ops.GalileanTranslation(shift)

        >>> op.inverse
        GalileanTranslation(FourVector( ... ))

        >>> op.inverse.translation.q.x
        Quantity(Array(-1, dtype=int32), unit='km')

        """
        return GalileanTranslation(cast(FourVector, -self.translation))

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
        >>> shift = FourVector (u.Quantity(1, "Gyr"), qshift)
        >>> op = cx.ops.GalileanTranslation(shift)

        Construct a vector to translate, using the convenience from_ (the
        0th component is :math:`c * t`, the rest are spatial components):

        >>> w = cx.FourVector.from_([0, 1, 2, 3], "km")
        >>> w.t
        Quantity(Array(0., dtype=float32, ...), unit='s')

        Apply the translation operator:

        >>> new = op(w)
        >>> new.x
        Quantity(Array(2, dtype=int32), unit='km')

        >>> new.t.uconvert("Gyr")
        Quantity(Array(1., dtype=float32, ...), unit='Gyr')

        """
        return x + self.translation

    @AbstractOperator.__call__.dispatch
    def __call__(
        self: "GalileanTranslation",
        t: u.Quantity["time"],
        x: AbstractPos3D,
        /,
        **__: Any,
    ) -> tuple[u.Quantity["time"], AbstractPos3D]:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.ops as cxo

        Explicitly construct the translation operator:

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
        >>> tshift = u.Quantity(1, "Gyr")
        >>> shift = cx.FourVector (tshift, qshift)
        >>> op = cx.ops.GalileanTranslation(shift)

        Construct a vector to translate

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> t = u.Quantity(1, "Gyr")
        >>> newt, newq = op(t, q)

        >>> newq.x
        Quantity(Array(2, dtype=int32), unit='km')

        >>> newt
        Quantity(Array(2, dtype=int32, ...), unit='Gyr')

        """
        return t + self.translation.t, x + self.translation.q

    # -------------------------------------------

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation."""
        return (
            wl.TextDoc(f"{self.__class__.__name__}(")
            + wl.pdoc(self.translation, **kwargs)
            + wl.TextDoc(")")
        )


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
    >>> op1 = cx.ops.GalileanTranslation(FourVector (tshift, qshift))

    >>> qshift = cx.CartesianPos3D.from_([0, 1, 0], "km")
    >>> tshift = u.Quantity(1, "Gyr")
    >>> op2 = cx.ops.GalileanTranslation(FourVector (tshift, qshift))

    >>> op3 = cx.ops.simplify_op(op1, op2)
    >>> op3
    GalileanTranslation(FourVector( ... ))

    """
    return GalileanTranslation(op1.translation + op2.translation)
