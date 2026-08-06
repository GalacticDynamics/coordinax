"""Shared strategy helpers for representation-kind class samplers."""

__all__: tuple[str, ...] = ()

from typing import TypeVar

import hypothesis.strategies as st

T = TypeVar("T")


def draw_subclass(
    draw: st.DrawFn,
    all_classes: tuple[type[T], ...],
    /,
    *,
    include: tuple[type[T], ...] | None,
    exclude: tuple[type[T], ...],
    kind: str,
) -> type[T]:
    """Draw a concrete subclass, honouring ``include`` / ``exclude``.

    Shared body for the ``*_classes`` strategies (bases, geometries, semantic
    kinds): candidates default to ``all_classes`` unless ``include`` is given,
    then ``exclude`` is removed and one is sampled. Raises `ValueError` (naming
    ``kind``) if nothing remains.

    Callers pass ``all_classes`` freshly from `get_all_subclasses` rather than
    from a module-level constant. A constant is resolved at *import* time, which
    is before any module imported later can register its subclasses -- and,
    unlike `get_all_subclasses`, it cannot be refreshed, so
    ``get_all_subclasses.cache_clear()`` had no effect on these three strategies.
    """
    candidates = all_classes if include is None else include
    exclude_set = set(exclude)
    candidates = tuple(c for c in candidates if c not in exclude_set)
    if not candidates:
        msg = f"No {kind} classes left after exclusions"
        raise ValueError(msg)
    return draw(st.sampled_from(candidates))
