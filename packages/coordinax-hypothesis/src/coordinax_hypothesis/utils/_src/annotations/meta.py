"""Utilities."""

__all__ = ("Metadata",)


from collections.abc import Callable
from typing import Any, TypedDict, final

import hypothesis.strategies as st


@final
class Metadata(TypedDict, total=False):  # closed=False
    """Holds shape and dtype information for strategy generation."""

    dtype: st.SearchStrategy[Any]
    shape: st.SearchStrategy[tuple[int, ...]]
    validators: list[Callable[[Any], bool]]  # Beartype validator functions
