# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanOperator"]


from typing import TYPE_CHECKING, Any, final, overload

import equinox as eqx
from plum import dispatch

import quaxed.numpy as xp
import unxt as u
from dataclassish.converters import Unless

from .base import AbstractGalileanOperator
from .boost import GalileanBoost
from .rotation import GalileanRotation
from .translation import GalileanTranslation
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.composite import AbstractCompositeOperator
from coordinax._src.operators.sequential import Sequence

if TYPE_CHECKING:
    from typing import Self


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
    translation : `coordinax.operators.GalileanTranslation`
        The spatial translation of the frame. See
        :class:`coordinax.operators.GalileanTranslation` for alternative
        inputs to construct this parameter.
    velocity : :class:`coordinax.operators.GalileanBoost`
        The boost to the frame. See
        :class:`coordinax.operators.GalileanBoost` for alternative
        inputs to construct this parameter.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.operators as cxo

    >>> op = cxo.GalileanOperator(
    ...     translation=u.Quantity([0., 2., 3., 4.], "kpc"),
    ...     velocity=u.Quantity([1., 2., 3.], "km/s"))
    >>> op
    GalileanOperator(
      rotation=GalileanRotation(rotation=f32[3,3]),
      translation=GalileanTranslation(
        translation=FourVector(
          t=Quantity[...](value=f32[], unit=Unit("kpc s / km")),
          q=CartesianPos3D( ... ) )
      ),
      velocity=GalileanBoost( velocity=CartesianVel3D( ... ) )
    )

    Note that the translation is a
    :class:`coordinax.operators.GalileanTranslation` with a
    :class:`vector.FourVector` translation, and the velocity is a
    :class:`coordinax.operators.GalileanBoost` with a
    :class:`vector.CartesianVel3D` velocity. We can also construct them
    directly, which allows for other vector types.

    >>> op = cxo.GalileanOperator(
    ...     translation=cxo.GalileanTranslation(
    ...         cx.FourVector(t=u.Quantity(2.5, "Gyr"),
    ...                       q=cx.SphericalPos(r=u.Quantity(1, "kpc"),
    ...                                         theta=u.Quantity(90, "deg"),
    ...                                         phi=u.Quantity(0, "rad") ) ) ),
    ...     velocity=cxo.GalileanBoost(
    ...         cx.CartesianVel3D.from_([1, 2, 3], "km/s") )
    ... )
    >>> op
    GalileanOperator(
      rotation=GalileanRotation(rotation=f32[3,3]),
      translation=GalileanTranslation(
        translation=FourVector(
          t=Quantity[...)](value=f32[], unit=Unit("Gyr")),
          q=SphericalPos( ... )
        )
      ),
      velocity=GalileanBoost( velocity=CartesianVel3D( ... ) )
    )

    Galilean operators can be applied to :class:`vector.FourVector`:

    >>> w = cx.FourVector.from_([0, 0, 0, 0], "kpc")
    >>> new = op(w)
    >>> new
    FourVector(
      t=Quantity[...](value=f32[], unit=Unit("kpc s / km")),
      q=CartesianPos3D( ... )
    )
    >>> new.t.ustrip("Gyr").round(2)
    Array(2.5, dtype=float32)
    >>> new.q.x
    Quantity['length'](Array(3.5567803, dtype=float32), unit='kpc')

    Also the Galilean operators can also be applied to
    :class:`vector.AbstractPos3D` and :class:`unxt.Quantity`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> newq, newt = op(q, t)
    >>> newq.x
    Quantity['length'](Array(3.5567803, dtype=float32), unit='kpc')
    >>> newt
    Quantity['time'](Array(2.5, dtype=float32), unit='Gyr')

    """

    rotation: GalileanRotation = eqx.field(
        default=GalileanRotation(xp.eye(3)),
        converter=GalileanRotation.from_,
    )
    """The in-frame spatial rotation."""

    translation: GalileanTranslation = eqx.field(
        default=GalileanTranslation(u.Quantity([0, 0, 0, 0], "kpc")),
        converter=Unless(GalileanTranslation, converter=GalileanTranslation.from_),
    )
    """The temporal + spatial translation.

    The translation vector [T, Q].  This parameters accetps either a
    :class:`coordinax.operators.GalileanTranslation` instance or
    any input that can be used to construct a :meth:`vector.FourVector`, using
    :meth:`vector.FourVector.from_`. See :class:`vector.FourVector` for
    details.
    """

    velocity: GalileanBoost = eqx.field(
        default=GalileanBoost(u.Quantity([0, 0, 0], "km/s")),
        converter=Unless(GalileanBoost, converter=GalileanBoost.from_),
    )
    """The boost to the frame.

    This parameters accepts either a
    :class:`coordinax.operators.GalileanBoost` instance or any input
    that can be used to construct a :class:`vector.CartesianVel3D`, using
    :meth:`vector.CartesianVel3D.from_`. See :class:`vector.CartesianVel3D` for
    details.
    """

    @property
    def operators(
        self,
    ) -> tuple[GalileanRotation, GalileanTranslation, GalileanBoost]:
        """Rotation -> translation -> boost."""
        return (self.rotation, self.translation, self.velocity)

    @overload
    def __getitem__(self, key: int) -> AbstractOperator: ...

    @overload
    def __getitem__(self, key: slice) -> "Self": ...

    def __getitem__(self, key: int | slice) -> "AbstractOperator | Self":
        if isinstance(key, int):
            return self.operators[key]
        return Sequence(self.operators[key])


@dispatch  # type: ignore[misc]
def simplify_op(op: GalileanOperator, /, **kwargs: Any) -> AbstractOperator:
    """Simplify a Galilean operator.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.operators as cxo

    This Galilean operator cannot be simplified:

    >>> op = cxo.GalileanOperator(
    ...     translation=u.Quantity([0., 2., 3., 4.], "kpc"),
    ...     velocity=u.Quantity([1., 2., 3.], "km/s"),
    ...     rotation=jnp.eye(3).at[0, 2].set(1),
    ... )
    >>> op
    GalileanOperator(
      rotation=GalileanRotation(rotation=f32[3,3]),
      translation=GalileanTranslation(
        translation=FourVector(
          t=Quantity[...](value=f32[], unit=Unit("kpc s / km")),
          q=CartesianPos3D( ... ) )
      ),
      velocity=GalileanBoost( velocity=CartesianVel3D( ... ) )
    )

    >>> cxo.simplify_op(op) is op
    True

    This Galilean operator can be simplified in all its components except the
    translation:

    >>> op = cxo.GalileanOperator(translation=u.Quantity([0., 2., 3., 4.], "kpc"))
    >>> cxo.simplify_op(op)
    Sequence(
      operators=( GalileanTranslation( translation=FourVector( ... ) ), )
    )

    """
    simple_ops = [simplify_op(x, **kwargs) for x in op.operators]
    if any(
        not isinstance(x, type(orig))
        for x, orig in zip(simple_ops, op.operators, strict=True)
    ):
        return simplify_op(Sequence(simple_ops))

    return op
