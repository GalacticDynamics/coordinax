"""Galilean coordinate transformations."""

__all__ = ("GalileanOp",)


from typing import TYPE_CHECKING, Any, TypeAlias, cast, final, overload

import equinox as eqx
import plum

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

from .boost import Boost
from .rotate import Rotate
from .translate import Translate
from coordinax._src import api
from coordinax._src.api import apply_op
from coordinax._src.constants import LENGTH, SPEED
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.composite import AbstractCompositeOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.operators.pipe import Pipe

if TYPE_CHECKING:
    from typing import Self


@final
class GalileanOp(AbstractCompositeOperator):
    r"""Operator for general Galilean transformations.

    In the transformed frame the coordinates are given by:

    $$
    (t, \mathbf{x}) \mapsto (t + s, R \mathbf{x} + \mathbf{v} t + \mathbf{a})
    $$

    where $R$ is the rotation matrix, $\mathbf{v}$ is the boost
    velocity, $\mathbf{a}$ is the spatial translation, and $s$ is
    the time translation. This is equivalent to a sequential operation of:

    1. a rotation
    2. a spatial translation
    3. a velocity boost

    Parameters
    ----------
    rotation
        The in-frame spatial rotation. This is a ``Rotate`` operator.
    translation
        The spatial translation. This is a ``Translate`` operator.
    velocity
        The velocity boost. This is a ``Boost`` operator.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.ops as cxo

    Create a Galilean operator with spatial translation and velocity boost:

    >>> op = cxo.GalileanOp(
    ...     translation=cxo.Translate.from_([2., 3., 4.], "km"),
    ...     velocity=cxo.Boost.from_([1., 2., 3.], "km/s"),
    ... )
    >>> op
    GalileanOp(
      Translate(
          {'x': Q(f64[], 'km'), 'y': Q(f64[], 'km'), 'z': Q(f64[], 'km')}, chart=Cart3D()
      ),
      Boost(
        delta={'x': Q(f64[], 'km / s'), 'y': Q(f64[], 'km / s'), 'z': Q(f64[], 'km / s')},
        chart=Cart3D()
      )
    )

    Apply to a Quantity (tau=0 means no boost displacement):

    >>> q = u.Q([0, 0, 0], "km")
    >>> tau = u.Q(0, "s")
    >>> op(tau, q)
    Quantity(Array([2., 3., 4.], dtype=float64), unit='km')

    """

    rotation: Rotate = eqx.field(
        default=Rotate(jnp.eye(3)),
        converter=Unless(Rotate, Rotate.from_),
    )
    """The in-frame spatial rotation."""

    translation: Translate = eqx.field(
        default=Translate.from_([0, 0, 0], "km"),
        converter=Unless(Translate, converter=Translate.from_),
    )
    """The spatial translation.

    This accepts either a ``Translate`` instance or any input that can be
    used to construct one via ``Translate.from_``.
    """

    velocity: Boost = eqx.field(
        default=Boost.from_([0, 0, 0], "km/s"),
        converter=Unless(Boost, converter=Boost.from_),
    )
    """The velocity boost.

    This accepts either a ``Boost`` instance or any input that can be used
    to construct one via ``Boost.from_``.
    """

    @property  # type: ignore[misc]
    def operators(  # type: ignore[override]
        self,
    ) -> tuple[Rotate, Translate, Boost]:
        """Rotate -> translation -> boost."""
        return (self.rotation, self.translation, self.velocity)

    @overload
    def __getitem__(self, key: int) -> AbstractOperator: ...

    @overload
    def __getitem__(self, key: slice) -> "Self": ...

    def __getitem__(self, key: int | slice) -> AbstractOperator | Pipe:
        """Getitem from the operators.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.ops as cxo

        >>> op = cxo.GalileanOp(
        ...     translation=cxo.Translate.from_([2., 3., 4.], "km"),
        ...     velocity=cxo.Boost.from_([1., 2., 3.], "km/s"),
        ... )

        >>> op[0]
        Rotate(f64[3,3](jax))

        >>> op[1:]
        Pipe(( Translate(...), Boost(...) ))

        """
        if isinstance(key, int):
            return self.operators[key]
        return Pipe(self.operators[key])  # type: ignore[arg-type]


# ===================================================================
# apply_op for GalileanOp


@plum.dispatch
def apply_op(
    op: GalileanOp, tau: Any, x: u.AbstractQuantity, /, **kwargs: Any
) -> u.AbstractQuantity:
    """Apply GalileanOp to a Quantity.

    This iterates through the component operators (rotation, translation,
    velocity boost) and applies each in sequence.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> op = cxo.GalileanOp(
    ...     translation=cxo.Translate.from_([2., 3., 4.], "km"),
    ...     velocity=cxo.Boost.from_([1., 2., 3.], "km/s"),
    ... )
    >>> q = u.Q([0, 0, 0], "km")
    >>> tau = u.Q(0, "s")  # tau=0 means no boost displacement
    >>> cxo.apply_op(op, tau, q)
    Quantity(Array([2., 3., 4.], dtype=float64), unit='km')

    """
    result = x
    for sub_op in op.operators:
        result = apply_op(sub_op, tau, result, **kwargs)
    return result


# ===================================================================
# Simplification

SimplifyOpR: TypeAlias = GalileanOp | Pipe | Translate | Boost | Rotate | Identity


@plum.dispatch
def simplify(op: GalileanOp, /, **kwargs: Any) -> SimplifyOpR:
    """Simplify a Galilean operator.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    This Galilean operator cannot be simplified:

    >>> op = cxo.GalileanOp(
    ...     translation=cxo.Translate.from_([2., 3., 4.], "km"),
    ...     velocity=cxo.Boost.from_([1., 2., 3.], "km/s"),
    ...     rotation=jnp.eye(3).at[0, 2].set(1),
    ... )
    >>> op
    GalileanOp(
      Rotate(R=f64[3,3](jax)),
      Translate( {'x': Q(f64[], 'km'), 'y': Q(f64[], 'km'), 'z': Q(f64[], 'km')}, chart=Cart3D() ),
      Boost( delta={'x': Q(f64[], 'km / s'), 'y': Q(f64[], 'km / s'), 'z': Q(f64[], 'km / s')}, chart=Cart3D() )
    )

    >>> cxo.simplify(op) is op
    True

    This Galilean operator can be simplified to just the translation:

    >>> op = cxo.GalileanOp(
    ...     translation=cxo.Translate.from_([2., 3., 4.], "km"))
    >>> cxo.simplify(op)
    Translate(
        {'x': Q(2., 'km'), 'y': Q(3., 'km'), 'z': Q(4., 'km')}, chart=Cart3D()
    )

    """
    simple_ops = [api.simplify(x, **kwargs) for x in op.operators]
    if any(
        not isinstance(x, type(orig))
        for x, orig in zip(simple_ops, op.operators, strict=True)
    ):
        return cast("SimplifyOpR", Pipe(tuple(simple_ops)).simplify())

    return op


# ===================================================================
# Constructors


@GalileanOp.from_.dispatch
def from_(cls: type[GalileanOp], q: u.AbstractQuantity, /) -> GalileanOp:
    """Construct a GalileanOp from a Quantity.

    Length dimension -> translation, speed dimension -> velocity boost.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    Length Quantity creates a translation:

    >>> cxo.GalileanOp.from_(u.Q([1, 2, 3], "km"))
    GalileanOp(
      Translate( {'x': Q(i64[], 'km'), 'y': Q(i64[], 'km'), 'z': Q(i64[], 'km')},
                chart=Cart3D()
      )
    )

    Speed Quantity creates a velocity boost:

    >>> cxo.GalileanOp.from_(u.Q([100, 0, 0], "km/s"))
    GalileanOp(
      Boost( delta={'x': Q(i64[], 'km / s'), 'y': Q(i64[], 'km / s'),
                    'z': Q(i64[], 'km / s')},
            chart=Cart3D()
      )
    )

    """
    dim = u.dimension_of(q)
    if dim == LENGTH:
        return cls(translation=q)
    if dim == SPEED:
        return cls(velocity=q)
    msg = f"Cannot construct GalileanOp from Quantity with dimension {dim}"
    raise TypeError(msg)
