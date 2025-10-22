"""Base classes for operators."""

__all__ = ("operate", "simplify_op")

from typing import Any

from plum import dispatch


@dispatch.abstract
def operate(op: type, params: dict[str, Any], tau: Any, x: Any, /) -> Any:
    """Apply an operator."""


@dispatch.abstract
def simplify_op(op: Any, /) -> Any:
    """Simplify an operator.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.Rotate.from_euler("z", 45)
    >>> simplified = cx.ops.simplify_op(op)
    >>> simplified
    Rotate(rotation=f32[3,3])

    >>> op = cx.ops.Rotate.from_euler("z", 45) @ cx.ops.Rotate.from_euler("z", -45)
    >>> cx.ops.simplify_op(op)
    Identity()

    """
