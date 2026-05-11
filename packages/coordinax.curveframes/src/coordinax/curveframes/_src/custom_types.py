"""Internal custom types."""

__all__ = (
    "CKey",
    "CDict",
)

from typing import Any, TypeAlias

CKey: TypeAlias = str
CDict: TypeAlias = dict[CKey, Any]
