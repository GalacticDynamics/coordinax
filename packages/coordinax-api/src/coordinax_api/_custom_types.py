__all__ = (
    "ComponentKey",
    "ProductComponentKey",
    "ComponentsKey",
    "CDict",
    "CsDict",
)

from typing import Any, TypeAlias

# Component key type: string for simple charts, tuple for product charts
ComponentKey: TypeAlias = str
ProductComponentKey: TypeAlias = tuple[str, str]
ComponentsKey: TypeAlias = ComponentKey | ProductComponentKey

# Parameter dictionary type alias (supports both flat and product keys)
CDict: TypeAlias = dict[ComponentKey, Any]
CsDict: TypeAlias = dict[ComponentsKey, Any]
