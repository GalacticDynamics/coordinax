__all__ = ("CKey", "CDict")

from typing import TypeAlias

# Component key type: string for all charts (including dot-delimited product keys)
CKey: TypeAlias = str

# Parameter dictionary type alias
# NOTE: deliberately the bare `dict`, not `dict[str, Any]`: a parametric
# annotation makes every plum signature that uses CDict "unfaithful",
# which disables plum's method cache and forces a full (~200x slower)
# resolution on every call of `act`/`pt_map`/`cconvert`/etc.
CDict: TypeAlias = dict
