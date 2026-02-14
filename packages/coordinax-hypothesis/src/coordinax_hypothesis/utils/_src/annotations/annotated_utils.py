"""Utilities."""

__all__ = ("AnnotatedNotIntrospectable",)

from dataclasses import dataclass

from typing import TypeVar, final

import beartype
import hypothesis.strategies as st
import plum
from is_annotated import isannotated

from .meta import Metadata
from .wrap import (
    RECOGNIZE_NONINTROSPECTABLE,
    AbstractNotIntrospectable,
    wrap_if_not_inspectable,
)

T = TypeVar("T")


BeartypeValidator = beartype.vale.Is[lambda x: x].__class__


@final
@dataclass(frozen=True, slots=True)
class AnnotatedNotIntrospectable(AbstractNotIntrospectable[T]):
    """Wrapper for Annotated types that are not introspectable."""

    ann: T


RECOGNIZE_NONINTROSPECTABLE.append((isannotated, AnnotatedNotIntrospectable))


@plum.dispatch
def strategy_for_annotation(
    ann: AnnotatedNotIntrospectable, /, *, meta: Metadata
) -> st.SearchStrategy:  # type: ignore[type-arg]
    """Unwrap and parse."""
    # Unpack Annotated type
    typ = ann.ann.__origin__
    # Extract metadata from annotations
    for md in ann.ann.__metadata__:
        if isinstance(md, dict):
            meta |= md
        elif isinstance(md, BeartypeValidator):
            # Initialize validators list if not present
            if "validators" not in meta:
                meta["validators"] = []
            meta["validators"].append(md.is_valid)

    # Re-dispatch with unwrapped type and parsed metadata
    return strategy_for_annotation(wrap_if_not_inspectable(typ), meta=meta)
