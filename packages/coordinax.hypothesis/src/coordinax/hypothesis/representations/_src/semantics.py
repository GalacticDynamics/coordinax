"""Hypothesis strategies for Coordinax vectors."""

__all__ = ("semantic_classes", "semantics")

from typing import Final

import hypothesis.strategies as st

import coordinax.representations as cxr

from coordinax.hypothesis.utils import get_all_subclasses

SEMANTICS: Final = get_all_subclasses(cxr.AbstractSemanticKind, exclude_abstract=True)


@st.composite
def semantic_classes(
    draw: st.DrawFn,
    *,
    include: tuple[type[cxr.AbstractSemanticKind], ...] | None = None,
    exclude: tuple[type[cxr.AbstractSemanticKind], ...] = (),
) -> type[cxr.AbstractSemanticKind]:
    """Generate random semantic classes (not instances).

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    include
        If provided, only generate semantic classes from this tuple. Otherwise,
        all concrete semantic classes are considered.
    exclude
        Semantic classes to exclude from generation. Default is empty.

    Returns
    -------
    type[cxr.AbstractSemanticKind]
        A concrete semantic class such as ``cxr.PointSemanticKind``.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.representations as cxr
    >>> import coordinax.hypothesis.main as cxst

    >>> @given(sem_cls=cxst.semantic_classes())
    ... def test_any_semantic_class(sem_cls):
    ...     assert issubclass(sem_cls, cxr.AbstractSemanticKind)

    >>> @given(sem_cls=cxst.semantic_classes(include=(cxr.Location,)))
    ... def test_subset(sem_cls):
    ...     assert issubclass(sem_cls, cxr.Location)

    """
    # Determine candidate semantic classes
    candidates = SEMANTICS if include is None else include

    # Filter out excluded semantic classes
    candidates = tuple(r for r in candidates if r not in exclude)

    if not candidates:
        msg = "No semantic classes left after exclusions"
        raise ValueError(msg)

    return draw(st.sampled_from(candidates))  # ty: ignore[invalid-return-type]


@st.composite
def semantics(
    draw: st.DrawFn,
    *,
    include: tuple[type[cxr.AbstractSemanticKind], ...] | None = None,
    exclude: tuple[type[cxr.AbstractSemanticKind], ...] = (),
) -> cxr.AbstractSemanticKind:
    """Generate random `coordinax` semantic kinds.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    include
        If provided, only generate semantic kinds from this tuple. Otherwise, all
        semantic kinds are considered.
    exclude
        Semantic kinds to exclude from generation. Default is empty (no exclusions).

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.representations as cxr
    >>> import coordinax.hypothesis.main as cxst

    >>> @given(sem=cxst.semantics())
    ... def test_any_semantic(sem):
    ...     assert isinstance(sem, cxr.AbstractSemanticKind)

    >>> @given(sem=cxst.semantics(include=(cxr.Location,)))
    ... def test_subset(sem):
    ...     assert isinstance(sem, cxr.Location)

    """
    sem_cls = draw(semantic_classes(include=include, exclude=exclude))
    return sem_cls()
