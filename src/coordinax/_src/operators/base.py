"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractOperator"]

import textwrap
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import equinox as eqx
from jaxtyping import ArrayLike
from plum import dispatch

import unxt as u
from dataclassish import field_items, flags

from .api import simplify_op
from coordinax._src.vectors.base import AbstractPos

if TYPE_CHECKING:
    from coordinax.ops import Pipe


class AbstractOperator(eqx.Module):  # type: ignore[misc]
    """Abstract base class for operators on coordinates and potentials.

    An operator is an object that defines a transformation on coordinates. It
    can be applied to a set of coordinates to produce a new set of coordinates.
    Operators can be composed together to form a sequence of transformations.

    When defining a new operator, it should be able to work on
    `coordinax.AbstractVector` objects. If it is a spatial operator, it should
    also be able to work on (`coordinax.AbstractPos`, `unxt.Quantity['time']`)
    pairs (and then also `coordinax.FourVector` objects). If the vector can be
    created from a `unxt.Quantity` object, then the operator should also be able
    to work on `unxt.Quantity` object.

    """

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch(precedence=-1)
    def from_(
        cls: "type[AbstractOperator]", *args: object, **kwargs: object
    ) -> "AbstractOperator":
        """Construct from a set of arguments.

        This is a low-priority dispatch that will be called if no other
        dispatch is found. It just tries to pass the arguments to the
        constructor.

        """
        return cls(*args, **kwargs)

    @classmethod
    @dispatch
    def from_(
        cls: "type[AbstractOperator]", obj: Mapping[str, Any], /
    ) -> "AbstractOperator":
        """Construct from a mapping.

        Examples
        --------
        >>> import coordinax as cx
        >>> pipe = cx.ops.Identity() | cx.ops.Identity()
        >>> cx.ops.Pipe.from_({"operators": pipe})
        Pipe((Identity(), Identity()))

        """
        return cls(**obj)

    @classmethod
    @dispatch
    def from_(
        cls: "type[AbstractOperator]",
        x: ArrayLike | list[float | int],
        unit: str,  # TODO: support unit object
        /,
    ) -> "AbstractOperator":
        """Construct from a Quantity's value and unit.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanSpatialTranslation.from_([1, 1, 1], "km")
        >>> print(op.translation)
        <CartesianPos3D (x[km], y[km], z[km])
            [1 1 1]>

        >>> op = cx.ops.GalileanTranslation.from_([3e5, 1, 1, 1], "km")
        >>> print(op.translation)
        <FourVector (t[s], q=(x[km], y[km], z[km]))
            [1.001 1.    1.    1.   ]>

        >>> op = cx.ops.GalileanBoost.from_([1, 1, 1], "km/s")
        >>> print(op.velocity)
        <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
            [1 1 1]>

        """
        return cls(u.Quantity(x, unit))

    # ===========================================
    # Operator API

    @dispatch.abstract
    def __call__(
        self: "AbstractOperator",
        x: AbstractPos,  # noqa: ARG002
        /,
        **kwargs: Any,  # noqa: ARG002
    ) -> AbstractPos:
        """Apply the operator to the coordinates `x`."""
        msg = "implement this method in the subclass"
        raise TypeError(msg)

    @dispatch.abstract
    def __call__(
        self: "AbstractOperator",
        x: AbstractPos,  # noqa: ARG002
        t: u.Quantity["time"],  # noqa: ARG002
        /,
        **kwargs: Any,  # noqa: ARG002
    ) -> AbstractPos:
        """Apply the operator to the coordinates `x` at a time `t`."""
        msg = "implement this method in the subclass"
        raise TypeError(msg)

    # -------------------------------------------

    @property
    @abstractmethod
    def is_inertial(self) -> bool:
        """Whether the operation maintains an inertial reference frame."""
        ...

    @property
    @abstractmethod
    def inverse(self) -> "AbstractOperator":
        """The inverse of the operator."""
        ...

    def simplify(self) -> "AbstractOperator":
        """Simplify the operator.

        This method calls `coordinax.ops.simplify_op` to simplify the
        operator.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.ops.Identity()
        >>> op.simplify() is op
        True

        >>> op = cx.ops.Pipe((cx.ops.Identity(), cx.ops.Identity()))
        >>> op.simplify()
        Identity()

        >>> op = cx.ops.GalileanOperator(translation=u.Quantity([0., 2., 3., 4.], "km"))
        >>> op.simplify()
        GalileanTranslation(FourVector( ... ))

        """
        return simplify_op(self)

    # ===========================================
    # Python stuff

    def __str__(self) -> str:
        """Return a string representation of the operator.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanRotation([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> print(op)
        GalileanRotation([[1 0 0]
         [0 1 0]
         [0 0 1]])

        >>> op = cx.ops.GalileanOperator(
        ...     translation=u.Quantity([0., 2, 3, 4], "km"),
        ...     velocity=u.Quantity([1., 2, 3], "km/s"),
        ...     rotation=jnp.eye(3).at[0, 2].set(1),
        ... )
        >>> print(op)
        GalileanOperator(
            rotation=GalileanRotation([[1. 0. 1.]
                [0. 1. 0.]
                [0. 0. 1.]]),
            translation=GalileanTranslation(<FourVector (t[s], q=(x[km], y[km], z[km]))
                    [0. 2. 3. 4.]>),
            velocity=GalileanBoost(<CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
                    [1. 2. 3.]>)
        )

        """  # noqa: E501
        name = self.__class__.__name__
        fitems = field_items(flags.FilterRepr, self)
        if len(fitems) == 1:
            fs = str(fitems[0][1]).replace("\n", "\n    ")
        else:
            fs = ",\n".join(textwrap.indent(f"{k}={v}", "    ") for k, v in fitems)
            fs = "\n" + fs + "\n"

        return f"{name}({fs})"

    # ===========================================
    # Operator Composition

    def __or__(self, other: "AbstractOperator") -> "Pipe":
        """Compose with another operator.

        Examples
        --------
        >>> import coordinax as cx

        >>> op1 = cx.ops.GalileanRotation([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> op2 = cx.ops.Identity()
        >>> op3 = op1 | op2
        >>> op3
        Pipe((GalileanRotation(rotation=i32[3,3]), Identity()))

        >>> op4 = cx.ops.Identity() | op3
        >>> op4
        Pipe((Identity(), GalileanRotation(rotation=i32[3,3]), Identity()))

        """
        from .pipe import Pipe

        if isinstance(other, Pipe):
            return other.__ror__(self)
        return Pipe((self, other))


@AbstractOperator.from_.dispatch  # type: ignore[attr-defined, misc]
def from_(cls: type[AbstractOperator], obj: AbstractOperator, /) -> AbstractOperator:
    """Construct an operator from another operator.

    Raises
    ------
    TypeError
        If the input object is not a subclass of the target class.

    Examples
    --------
    >>> import coordinax as cx

    If the object is the same type, it should return the object itself.

    >>> op = cx.ops.Identity()
    >>> cx.ops.Identity.from_(op) is op
    True

    If the object is a different type, it will error.

    >>> try:
    ...     cx.ops.GalileanBoost.from_(op)
    ... except TypeError as e:
    ...     print(e)
    Cannot construct <class 'coordinax...GalileanBoost'> from <class 'coordinax...Identity'>.

    Unless the object is a subclass of the target class.

    >>> class MyOperator(cx.ops.Identity):
    ...     pass

    >>> op = MyOperator()
    >>> op
    MyOperator()

    >>> newop = cx.ops.Identity.from_(op)
    >>> newop is op, isinstance(newop, cx.ops.Identity)
    (False, True)

    """  # noqa: E501
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls} from {type(obj)}."
        raise TypeError(msg)

    # avoid copying if the types are the same. Isinstance is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(field_items(obj)))
