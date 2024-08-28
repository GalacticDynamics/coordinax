# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanOperator"]


from typing import TYPE_CHECKING, Any, final, overload

import equinox as eqx

import quaxed.array_api as xp
from dataclassish.converters import Unless
from unxt import Quantity

from .base import AbstractGalileanOperator
from .boost import GalileanBoostOperator
from .rotation import GalileanRotationOperator
from .translation import GalileanTranslationOperator
from coordinax._coordinax.operators.base import AbstractOperator
from coordinax._coordinax.operators.composite import AbstractCompositeOperator
from coordinax._coordinax.operators.funcs import simplify_op
from coordinax._coordinax.operators.identity import IdentityOperator
from coordinax._coordinax.operators.sequential import OperatorSequence

if TYPE_CHECKING:
    from typing_extensions import Self


@final
class GalileanOperator(AbstractCompositeOperator, AbstractGalileanOperator):
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
    translation : `coordinax.operators.GalileanTranslationOperator`
        The spatial translation of the frame. See
        :class:`coordinax.operators.GalileanTranslationOperator` for
        alternative inputs to construct this parameter.
    velocity : :class:`coordinax.operators.GalileanBoostOperator`
        The boost to the frame. See
        :class:`coordinax.operators.GalileanBoostOperator` for
        alternative inputs to construct this parameter.

    Examples
    --------
    We start with the required imports:

    >>> from unxt import Quantity
    >>> import coordinax as cx

    We can then create a Galilean operator:

    >>> op = cx.operators.GalileanOperator(
    ...     translation=Quantity([0., 2., 3., 4.], "kpc"),
    ...     velocity=Quantity([1., 2., 3.], "km/s"))
    >>> op
    GalileanOperator(
      rotation=GalileanRotationOperator(rotation=f32[3,3]),
      translation=GalileanTranslationOperator(
        translation=FourVector(
          t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("kpc s / km")),
          q=CartesianPosition3D( ... ) )
      ),
      velocity=GalileanBoostOperator( velocity=CartesianVelocity3D( ... ) )
    )

    Note that the translation is a
    :class:`coordinax.operators.GalileanTranslationOperator` with a
    :class:`vector.FourVector` translation, and the velocity is a
    :class:`coordinax.operators.GalileanBoostOperator` with a
    :class:`vector.CartesianVelocity3D` velocity. We can also construct them
    directly, which allows for other vector types.

    >>> op = cx.operators.GalileanOperator(
    ...     translation=cx.operators.GalileanTranslationOperator(
    ...         cx.FourVector(t=Quantity(2.5, "Gyr"),
    ...                    q=cx.SphericalPosition(r=Quantity(1, "kpc"),
    ...                                         theta=Quantity(90, "deg"),
    ...                                         phi=Quantity(0, "rad") ) ) ),
    ...     velocity=cx.operators.GalileanBoostOperator(
    ...         cx.CartesianVelocity3D(d_x=Quantity(1, "km/s"),
    ...                                    d_y=Quantity(2, "km/s"),
    ...                                    d_z=Quantity(3, "km/s")))
    ... )
    >>> op
    GalileanOperator(
      rotation=GalileanRotationOperator(rotation=f32[3,3]),
      translation=GalileanTranslationOperator(
        translation=FourVector(
          t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("Gyr")),
          q=SphericalPosition( ... )
        )
      ),
      velocity=GalileanBoostOperator( velocity=CartesianVelocity3D( ... ) )
    )

    Galilean operators can be applied to :class:`vector.FourVector`:

    >>> w = cx.FourVector.constructor([0, 0, 0, 0], "kpc")
    >>> new = op(w)
    >>> new
    FourVector(
      t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("kpc s / km")),
      q=CartesianPosition3D( ... )
    )
    >>> new.t.to_units("Gyr").value.round(2)
    Array(2.5, dtype=float32)
    >>> new.q.x
    Quantity['length'](Array(3.5567803, dtype=float32), unit='kpc')

    Also the Galilean operators can also be applied to
    :class:`vector.AbstractPosition3D` and :class:`unxt.Quantity`:

    >>> q = cx.CartesianPosition3D.constructor([0, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")
    >>> newq, newt = op(q, t)
    >>> newq.x
    Quantity['length'](Array(3.5567803, dtype=float32), unit='kpc')
    >>> newt
    Quantity['time'](Array(2.5, dtype=float32), unit='Gyr')

    """

    rotation: GalileanRotationOperator = eqx.field(
        default=GalileanRotationOperator(xp.eye(3)),
        converter=GalileanRotationOperator,
    )
    """The in-frame spatial rotation."""

    translation: GalileanTranslationOperator = eqx.field(
        default=GalileanTranslationOperator(Quantity([0, 0, 0, 0], "kpc")),
        converter=Unless(
            GalileanTranslationOperator, converter=GalileanTranslationOperator
        ),
    )
    """The temporal + spatial translation.

    The translation vector [T, Q].  This parameters accetps either a
    :class:`coordinax.operators.GalileanTranslationOperator` instance or
    any input that can be used to construct a :meth:`vector.FourVector`, using
    :meth:`vector.FourVector.constructor`. See :class:`vector.FourVector` for
    details.
    """

    velocity: GalileanBoostOperator = eqx.field(
        default=GalileanBoostOperator(Quantity([0, 0, 0], "km/s")),
        converter=Unless(GalileanBoostOperator, converter=GalileanBoostOperator),
    )
    """The boost to the frame.

    This parameters accepts either a
    :class:`coordinax.operators.GalileanBoostOperator` instance or any
    input that can be used to construct a
    :class:`vector.CartesianVelocity3D`, using
    :meth:`vector.CartesianVelocity3D.constructor`. See
    :class:`vector.CartesianVelocity3D` for details.
    """

    @property
    def operators(
        self,
    ) -> tuple[
        GalileanRotationOperator, GalileanTranslationOperator, GalileanBoostOperator
    ]:
        """Rotation -> translateion -> boost."""
        return (self.rotation, self.translation, self.velocity)

    @overload
    def __getitem__(self, key: int) -> AbstractOperator: ...

    @overload
    def __getitem__(self, key: slice) -> "Self": ...

    def __getitem__(self, key: int | slice) -> "AbstractOperator | Self":
        if isinstance(key, int):
            return self.operators[key]
        return OperatorSequence(self.operators[key])


@simplify_op.register
def _simplify_op_galilean(op: GalileanOperator, /, **kwargs: Any) -> AbstractOperator:
    """Simplify a Galilean operator."""
    # Check if all the sub-operators can be simplified to the identity.
    if all(
        isinstance(simplify_op(x, **kwargs), IdentityOperator) for x in op.operators
    ):
        return IdentityOperator()

    return op
