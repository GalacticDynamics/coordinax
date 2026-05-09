"""Hypothesis strategies for Coordinax vectors."""

__all__ = ("geometry_classes", "geometries")

from typing import Final

import hypothesis.strategies as st

import coordinax.representations as cxr

from coordinax.hypothesis.utils import get_all_subclasses

GEOMETRIES: Final = get_all_subclasses(cxr.AbstractGeometry, exclude_abstract=True)


@st.composite
def geometry_classes(
    draw: st.DrawFn,
    *,
    include: tuple[type[cxr.AbstractGeometry], ...] | None = None,
    exclude: tuple[type[cxr.AbstractGeometry], ...] = (),
) -> type[cxr.AbstractGeometry]:
    """Generate random geometry classes (not instances).

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    include
        If provided, only generate geometry classes from this tuple. Otherwise,
        all concrete geometry classes are considered.
    exclude
        Geometry classes to exclude from generation. Default is empty.

    Returns
    -------
    type[cxr.AbstractGeometry]
        A concrete geometry class such as ``cxr.PointGeometry``.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.representations as cxr
    >>> import coordinax.hypothesis.main as cxst

    >>> @given(geom_cls=cxst.geometry_classes())
    ... def test_any_geometry_class(geom_cls):
    ...     assert issubclass(geom_cls, cxr.AbstractGeometry)

    >>> @given(geom_cls=cxst.geometry_classes(include=(cxr.PointGeometry,)))
    ... def test_subset(geom_cls):
    ...     assert issubclass(geom_cls, cxr.PointGeometry)

    """
    # Determine candidate geometry classes
    candidates = GEOMETRIES if include is None else include

    # Filter out excluded geometry classes
    candidates = tuple(r for r in candidates if r not in exclude)

    if not candidates:
        msg = "No role classes left after exclusions"
        raise ValueError(msg)

    return draw(st.sampled_from(candidates))  # ty: ignore[invalid-return-type]


@st.composite
def geometries(
    draw: st.DrawFn,
    *,
    include: tuple[type[cxr.AbstractGeometry], ...] | None = None,
    exclude: tuple[type[cxr.AbstractGeometry], ...] = (),
) -> cxr.AbstractGeometry:
    """Generate random `coordinax` geometries.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    include
        If provided, only generate geometries from this tuple. Otherwise, all
        geometries are considered (PointGeometry, etc.).
    exclude
        Geometries to exclude from generation. Default is empty (no exclusions).

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.representations as cxr
    >>> import coordinax.hypothesis.main as cxst

    >>> @given(role=cxst.geometries())
    ... def test_any_geometry(role):
    ...     assert isinstance(role, cxr.AbstractGeometry)

    >>> @given(role=cxst.geometries(include=(cxr.PointGeometry,)))
    ... def test_position_like_geometries(role):
    ...     assert isinstance(role, cxr.PointGeometry)

    """
    geom_cls = draw(geometry_classes(include=include, exclude=exclude))
    return geom_cls()
