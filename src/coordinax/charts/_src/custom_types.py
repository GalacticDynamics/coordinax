"""Custom types."""

__all__: tuple[str, ...] = ("CKey", "CDict")

from typing import Any, TypeAlias

CKey: TypeAlias = str
CDict: TypeAlias = dict[CKey, Any]
