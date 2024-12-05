"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractOperator"]

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import equinox as eqx
from jaxtyping import ArrayLike
from plum import dispatch

import unxt as u
from dataclassish import field_items

from coordinax._src.vectors.base import AbstractPos

if TYPE_CHECKING:
    from coordinax.operators import Sequence


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
        >>> import coordinax.operators as co
        >>> operators = co.Identity() | co.Identity()
        >>> co.Sequence.from_({"operators": operators})
        Sequence(operators=(Identity(), Identity()))

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
        >>> import coordinax.operators as cxo

        >>> op = cxo.GalileanSpatialTranslation.from_([1, 1, 1], "kpc")
        >>> print(op.translation)
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [1. 1. 1.]>

        >>> op = cxo.GalileanTranslation.from_([3e5, 1, 1, 1], "kpc")
        >>> print(op.translation)
        <FourVector (t[kpc s / km], q=(x[kpc], y[kpc], z[kpc]))
            [1.001 1.    1.    1.   ]>

        >>> op = cxo.GalileanBoost.from_([1, 1, 1], "km/s")
        >>> print(op.velocity)
        <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
            [1. 1. 1.]>

        """
        return cls(u.Quantity(x, unit))

    # -------------------------------------------

    @dispatch.abstract
    def __call__(
        self: "AbstractOperator",
        x: AbstractPos,  # noqa: ARG002
        /,
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

    # ===========================================
    # Sequence

    def __or__(self, other: "AbstractOperator") -> "Sequence":
        """Compose with another operator.

        Examples
        --------
        >>> import coordinax.operators as cxo

        >>> op1 = cxo.Identity()
        >>> op2 = cxo.Identity()
        >>> op1 | op2
        Sequence(operators=(Identity(), Identity()))

        """
        from .sequential import Sequence

        if isinstance(other, Sequence):
            return other.__ror__(self)
        return Sequence((self, other))


# TODO: move to the class in py3.11+
@AbstractOperator.from_.dispatch  # type: ignore[attr-defined, misc]
def from_(cls: type[AbstractOperator], obj: AbstractOperator, /) -> AbstractOperator:
    """Construct an operator from another operator.

    Examples
    --------
    >>> import coordinax.operators as cxo

    If the object is the same type, it should return the object itself.

    >>> op = cxo.Identity()
    >>> cxo.Identity.from_(op) is op
    True

    If the object is a different type, it will error.

    >>> try:
    ...     cxo.GalileanBoost.from_(op)
    ... except TypeError as e:
    ...     print(e)
    Cannot construct <class 'coordinax...GalileanBoost'> from <class 'coordinax...Identity'>.

    Unless the object is a subclass of the target class.

    >>> class MyOperator(cxo.Identity):
    ...     pass

    >>> op = MyOperator()
    >>> op
    MyOperator()

    >>> newop = cxo.Identity.from_(op)
    >>> newop is op, isinstance(newop, cxo.Identity)
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
