"""Base classes for operators on coordinates and potentials."""

__all__ = ["AbstractOperator"]

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import equinox as eqx
from plum import dispatch

from unxt import Quantity

from coordinax._base_pos import AbstractPosition
from coordinax._utils import dataclass_items

if TYPE_CHECKING:
    from ._sequential import OperatorSequence


class AbstractOperator(eqx.Module):  # type: ignore[misc]
    """Abstract base class for operators on coordinates and potentials."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch  # type: ignore[misc]
    def constructor(
        cls: "type[AbstractOperator]", obj: Mapping[str, Any], /
    ) -> "AbstractOperator":
        """Construct from a mapping.

        Parameters
        ----------
        obj : Mapping[str, Any]
            The object to construct from.

        Returns
        -------
        AbstractOperator
            The constructed operator.

        Examples
        --------
        >>> import coordinax.operators as co
        >>> operators = co.IdentityOperator() | co.IdentityOperator()
        >>> co.OperatorSequence.constructor({"operators": operators})
        OperatorSequence(operators=(IdentityOperator(), IdentityOperator()))

        """
        return cls(**obj)

    # -------------------------------------------

    @dispatch.abstract
    def __call__(
        self: "AbstractOperator",
        x: AbstractPosition,  # noqa: ARG002
        /,
    ) -> AbstractPosition:
        """Apply the operator to the coordinates `x`."""
        msg = "implement this method in the subclass"
        raise TypeError(msg)

    @dispatch.abstract
    def __call__(
        self: "AbstractOperator",
        x: AbstractPosition,  # noqa: ARG002
        t: Quantity["time"],  # noqa: ARG002
        /,
    ) -> AbstractPosition:
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

    def __or__(self, other: "AbstractOperator") -> "OperatorSequence":
        """Compose with another operator."""
        from ._sequential import OperatorSequence

        if isinstance(other, OperatorSequence):
            return other.__ror__(self)
        return OperatorSequence((self, other))


op_call_dispatch = AbstractOperator.__call__.dispatch  # type: ignore[attr-defined]


# TODO: move to the class in py3.11+
@AbstractOperator.constructor._f.dispatch  # type: ignore[misc]  # noqa: SLF001
def constructor(
    cls: type[AbstractOperator], obj: AbstractOperator, /
) -> AbstractOperator:
    """Construct an operator from another operator.

    Parameters
    ----------
    cls : type[AbstractOperator]
        The operator class.
    obj : :class:`coordinax.operators.AbstractOperator`
        The object to construct from.

    """  # pylint: disable=R0801
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls} from {type(obj)}."
        raise TypeError(msg)

    # avoid copying if the types are the same. Isinstance is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(dataclass_items(obj)))
