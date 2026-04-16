__all__ = ("CKey", "CDict")

from typing import Any, TypeAlias

# Component key type: string for all charts (including dot-delimited product keys)
CKey: TypeAlias = str

# Parameter dictionary type alias
CDict: TypeAlias = dict[CKey, Any]
