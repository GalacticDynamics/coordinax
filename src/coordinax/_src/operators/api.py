"""Base classes for operators on coordinates and potentials."""

__all__ = ["simplify_op"]

from typing import Any

from plum import dispatch


@dispatch.abstract  # type: ignore[misc]
def simplify_op(op: Any, /) -> Any:
    """Simplify an operator."""
