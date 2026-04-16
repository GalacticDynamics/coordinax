"""Hypothesis strategies for Coordinax vectors."""

__all__ = ("basis_classes", "bases")

from typing import Final

import hypothesis.strategies as st

import coordinax.representations as cxr

from coordinax.hypothesis.utils import get_all_subclasses

BASES: Final = get_all_subclasses(cxr.AbstractBasis, exclude_abstract=True)


@st.composite
def basis_classes(
    draw: st.DrawFn,
    *,
    include: tuple[type[cxr.AbstractBasis], ...] | None = None,
    exclude: tuple[type[cxr.AbstractBasis], ...] = (),
) -> type[cxr.AbstractBasis]:
    """Generate random basis classes (not instances).

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    include
        If provided, only generate basis classes from this tuple. Otherwise,
        all concrete basis classes are considered.
    exclude
        Basis classes to exclude from generation. Default is empty.

    Returns
    -------
    type[cxr.AbstractBasis]
        A concrete basis class such as ``cxr.CartesianBasis``.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.representations as cxr
    >>> import coordinax.hypothesis.main as cxst

    >>> @given(basis_cls=cxst.basis_classes())
    ... def test_any_basis_class(basis_cls):
    ...     assert issubclass(basis_cls, cxr.AbstractBasis)

    >>> @given(basis_cls=cxst.basis_classes(include=(cxr.NoBasis,)))
    ... def test_subset(basis_cls):
    ...     assert issubclass(basis_cls, cxr.NoBasis)

    """
    # Determine candidate basis classes
    candidates = BASES if include is None else include

    # Filter out excluded basis classes
    candidates = tuple(r for r in candidates if r not in exclude)

    if not candidates:
        msg = "No basis classes left after exclusions"
        raise ValueError(msg)

    return draw(st.sampled_from(candidates))


@st.composite
def bases(
    draw: st.DrawFn,
    *,
    include: tuple[type[cxr.AbstractBasis], ...] | None = None,
    exclude: tuple[type[cxr.AbstractBasis], ...] = (),
) -> cxr.AbstractBasis:
    """Generate random `coordinax` bases.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    include
        If provided, only generate bases from this tuple. Otherwise, all
        bases are considered (NoBasis, etc.).
    exclude
        Bases to exclude from generation. Default is empty (no exclusions).

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.representations as cxr
    >>> import coordinax.hypothesis.representations as cxst

    >>> @given(role=cxst.bases())
    ... def test_any_basis(role):
    ...     assert isinstance(role, cxr.AbstractBasis)

    >>> @given(role=cxst.bases(include=(cxr.NoBasis,)))
    ... def test_position_like_bases(role):
    ...     assert isinstance(role, cxr.NoBasis)

    """
    basis_cls = draw(basis_classes(include=include, exclude=exclude))  # ty: ignore[missing-argument]
    return basis_cls()
