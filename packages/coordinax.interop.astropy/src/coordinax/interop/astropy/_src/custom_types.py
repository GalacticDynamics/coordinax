"""Internal custom types."""

__all__ = (
    "CKey",
    "CDict",
)

from typing import TypeAlias

CKey: TypeAlias = str
# NOTE: deliberately the bare `dict`, not `dict[str, Any]`: a parametric
# annotation makes every plum signature that uses CDict "unfaithful",
# which disables plum's method cache and forces a full (~200x slower)
# resolution on every call of `act`/`pt_map`/`cconvert`/etc.
CDict: TypeAlias = dict
