"""Null atlas."""

__all__ = ("NoAtlas", "no_atlas")


from typing import Any, NoReturn

import jax.tree_util as jtu

from coordinax._src.base_atlas import AbstractAtlas


@jtu.register_static
class NoAtlas(AbstractAtlas):
    """Trivial atlas that supports no charts."""

    ndim = 0

    def default_chart(self) -> NoReturn:
        raise ValueError("NoAtlas does not support any charts.")

    def has_chart(self, _: Any, /) -> bool:
        return False


no_atlas = NoAtlas()
"""Canonical empty atlas that supports no charts."""
