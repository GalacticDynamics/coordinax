"""Coordinax Hypothesis Utilities.

NOTE: these are internal utilities for the `coordinax.hypothesis` package and are not guaranteed to be stable. They may be changed without warning.

"""

__all__ = (
    "draw_if_strategy",
    "get_all_subclasses",
    "annotations",
    "strip_return_annotation",
    # Custom types
    "Shape",
    "CKey",
    "CDict",
)

from ._src import (
    CDict,
    CKey,
    Shape,
    annotations,
    draw_if_strategy,
    get_all_subclasses,
    strip_return_annotation,
)
