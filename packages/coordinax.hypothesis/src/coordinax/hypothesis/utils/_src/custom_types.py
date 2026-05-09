"""Internal custom types."""

__all__ = (
    "Shape",
    "CKey",
    "CDict",
)

from typing import Any, TypeAlias

# =========================================================
# Array-related Types

Shape: TypeAlias = tuple[int, ...]

# =========================================================
# Vector-related Types

CKey: TypeAlias = str
CDict: TypeAlias = dict[CKey, Any]
