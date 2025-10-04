"""Base classes for operators."""

__all__ = ("operate", "simplify")

from typing import Any

from plum import dispatch


@dispatch.abstract
def operate(op: type, params: dict[str, Any], tau: Any, x: Any, /) -> Any:
    """Apply an operator.

    Examples
    --------
    >>> import coordinax.ops as cxo

    ### `Identity` operator

    ### `Add` operator

    ### `Rotate` operator

    """


@dispatch.abstract
def simplify(op: Any, /) -> Any:
    """Simplify an operator.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.Rotate.from_euler("z", u.Quantity(45, "deg"))
    >>> simplified = cx.ops.simplify(op)
    >>> simplified
    Rotate(rotation=f32[3,3])

    >>> op = (  cx.ops.Rotate.from_euler("z", u.Quantity(45, "deg"))
    ...       @ cx.ops.Rotate.from_euler("z", u.Quantity(-45, "deg")))
    >>> cx.ops.simplify(op)
    Identity()

    """
