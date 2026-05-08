"""Hypothesis strategies for Coordinax vectors."""

__all__ = (
    "valid_basis_classes_for_geometry",
    "valid_semantic_classes_for_geometry",
    "representations",
)


from typing import cast

import hypothesis.strategies as st
import plum

import coordinax.representations as cxr
from coordinax.representations._src.semantics import _TANGENT_TIME_ORDER_LADDER

from .bases import bases
from .geoms import geometries
from .semantics import semantics
from coordinax.hypothesis.utils import draw_if_strategy, get_all_subclasses


@plum.dispatch
def valid_basis_classes_for_geometry(  # noqa: F811
    geom_kind: cxr.AbstractGeometry, /
) -> tuple[type[cxr.AbstractBasis], ...]:
    """Return valid basis classes for a geometry kind."""
    del geom_kind
    return cast(
        "tuple[type[cxr.AbstractBasis], ...]",
        get_all_subclasses(cxr.AbstractBasis, exclude_abstract=True),
    )


@plum.dispatch
def valid_basis_classes_for_geometry(  # noqa: F811
    geom_kind: cxr.PointGeometry, /
) -> tuple[type[cxr.AbstractBasis], ...]:
    """Return valid basis classes for point geometry."""
    del geom_kind
    return (cxr.NoBasis,)


@plum.dispatch
def valid_basis_classes_for_geometry(  # noqa: F811
    geom_kind: cxr.TangentGeometry, /
) -> tuple[type[cxr.AbstractBasis], ...]:
    """Return valid basis classes for tangent geometry.

    TangentGeometry requires a linear basis (coordinate or physical).
    """
    del geom_kind
    return cast(
        "tuple[type[cxr.AbstractBasis], ...]",
        get_all_subclasses(cxr.AbstractLinearBasis, exclude_abstract=True),
    )


@plum.dispatch
def valid_semantic_classes_for_geometry(
    geom_kind: cxr.AbstractGeometry, /
) -> tuple[type[cxr.AbstractSemanticKind], ...]:
    """Return valid semantic classes for a geometry kind."""
    del geom_kind
    return cast(
        "tuple[type[cxr.AbstractSemanticKind], ...]",
        get_all_subclasses(cxr.AbstractSemanticKind, exclude_abstract=True),
    )


@plum.dispatch
def valid_semantic_classes_for_geometry(  # noqa: F811
    geom_kind: cxr.PointGeometry, /
) -> tuple[type[cxr.AbstractSemanticKind], ...]:
    """Return valid semantic classes for point geometry."""
    del geom_kind
    return (cxr.Location,)


@plum.dispatch
def valid_semantic_classes_for_geometry(  # noqa: F811
    geom_kind: cxr.TangentGeometry, /
) -> tuple[type[cxr.AbstractSemanticKind], ...]:
    """Return valid semantic classes for tangent geometry.

    Returns all registered tangent semantic kinds (e.g. Displacement, Velocity,
    Acceleration), sorted by their ``order`` attribute for deterministic output.
    """
    del geom_kind
    return cast(
        "tuple[type[cxr.AbstractSemanticKind], ...]",
        tuple(cls for _, cls in sorted(_TANGENT_TIME_ORDER_LADDER.items())),
    )


@st.composite
def representations(
    draw: st.DrawFn,
    *,
    geom_kind: cxr.AbstractGeometry
    | None
    | st.SearchStrategy[cxr.AbstractGeometry | None] = None,
    basis_kind: cxr.AbstractBasis
    | None
    | st.SearchStrategy[cxr.AbstractBasis | None] = None,
    semantic_kind: cxr.AbstractSemanticKind
    | None
    | st.SearchStrategy[cxr.AbstractSemanticKind | None] = None,
    check_valid: bool = True,
) -> cxr.Representation:
    """Generate random `coordinax` representations.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    geom_kind
        Geometry kind (or strategy) to use. If `None`, draw from all
        available geometry kinds.
    basis_kind
        Basis kind (or strategy) to use. If `None`, draw from all bases or,
        when `check_valid=True`, only those compatible with `geom_kind`.
    semantic_kind
        Semantic kind (or strategy) to use. If `None`, draw from all semantics
        or, when `check_valid=True`, only those compatible with `geom_kind`.
    check_valid
        If `True`, enforce geometry-conditioned basis and semantic constraints.
        If `False`, allow explicit incompatible pairings.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.representations as cxr
    >>> import coordinax.hypothesis.main as cxst

    >>> @given(role=cxst.representations())
    ... def test_any_representation(role):
    ...     assert isinstance(role, cxr.Representation)

    >>> @given(rep=cxst.representations(geom_kind=cxr.PointGeometry()))
    ... def test_point_representations(rep):
    ...     assert isinstance(rep.geom_kind, cxr.PointGeometry)
    ...     assert isinstance(rep.basis, cxr.NoBasis)
    ...     assert isinstance(rep.semantic_kind, cxr.Location)

    """
    # Draw the geometry kind
    geom_kind = draw_if_strategy(draw, geom_kind)
    if geom_kind is None:
        geom_kind = draw(geometries())
    assert isinstance(geom_kind, cxr.AbstractGeometry)

    # Draw the basis kind
    basis_kind = draw_if_strategy(draw, basis_kind)
    if basis_kind is None:
        include_bases = (
            valid_basis_classes_for_geometry(geom_kind) if check_valid else None
        )
        basis_kind = draw(bases(include=include_bases))
    elif check_valid and not isinstance(
        basis_kind,
        (valid_basis_classes := valid_basis_classes_for_geometry(geom_kind)),
    ):
        valid_basis = ", ".join(c.__name__ for c in valid_basis_classes)
        msg = (
            "Invalid basis_kind for geom_kind. "
            f"Got {type(basis_kind).__name__} for {type(geom_kind).__name__}; "
            f"expected one of: {valid_basis}."
        )
        raise ValueError(msg)

    # Draw the semantic kind
    semantic_kind = draw_if_strategy(draw, semantic_kind)
    if semantic_kind is None:
        include_sems = (
            valid_semantic_classes_for_geometry(geom_kind) if check_valid else None
        )
        semantic_kind = draw(semantics(include=include_sems))

    elif check_valid and not isinstance(
        semantic_kind,
        (valid_semantic_classes := valid_semantic_classes_for_geometry(geom_kind)),
    ):
        valid_semantics = ", ".join(c.__name__ for c in valid_semantic_classes)
        msg = (
            "Invalid semantic_kind for geom_kind. "
            f"Got {type(semantic_kind).__name__} for {type(geom_kind).__name__}; "
            f"expected one of: {valid_semantics}."
        )
        raise ValueError(msg)

    # Construct the Representation
    return cxr.Representation(
        geom_kind=geom_kind, basis=basis_kind, semantic_kind=semantic_kind
    )
