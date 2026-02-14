"""Hypothesis strategies for Coordinax vectors."""

__all__ = ("role_classes", "roles", "point_role", "physical_roles", "coord_roles")

from typing import Final, TypeVar

import hypothesis.strategies as st

import coordinax.roles as cxr
from coordinax_hypothesis.utils import get_all_subclasses

Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])

ROLES: Final = get_all_subclasses(cxr.AbstractPhysRole, exclude_abstract=True)
PHYS_ROLES: Final = tuple(r for r in ROLES if issubclass(r, cxr.AbstractPhysRole))
COORD_ROLES: Final = tuple(r for r in ROLES if issubclass(r, cxr.AbstractCoordRole))


@st.composite
def role_classes(
    draw: st.DrawFn,
    *,
    include: tuple[type[cxr.AbstractRole], ...] | None = None,
    exclude: tuple[type[cxr.AbstractRole], ...] = (),
) -> type[cxr.AbstractRole]:
    """Generate random Coordinax role classes (not instances).

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    include
        If provided, only generate role classes from this tuple. Otherwise, all
        concrete role classes are considered.
    exclude
        Role classes to exclude from generation. Default is empty.

    Returns
    -------
    type[cxr.AbstractRole]
        A concrete role class such as ``cxr.Point``, ``cxr.PhysDisp``, etc.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.roles as cxr
    >>> import coordinax_hypothesis.core as cxst

    >>> @given(role_cls=cxst.role_classes())
    ... def test_any_role_class(role_cls):
    ...     assert issubclass(role_cls, cxr.AbstractRole)

    >>> @given(role_cls=cxst.role_classes(include=(cxr.PhysDisp, cxr.PhysVel)))
    ... def test_subset(role_cls):
    ...     assert issubclass(role_cls, cxr.AbstractPhysRole)

    """
    # Determine candidate role classes
    candidates = ROLES if include is None else include

    # Filter out excluded role classes
    candidates = tuple(r for r in candidates if r not in exclude)

    if not candidates:
        msg = "No role classes left after exclusions"
        raise ValueError(msg)

    return draw(st.sampled_from(candidates))


@st.composite
def roles(
    draw: st.DrawFn,
    *,
    include: tuple[type[cxr.AbstractRole], ...] | None = None,
    exclude: tuple[type[cxr.AbstractRole], ...] = (),
) -> cxr.AbstractRole:
    """Generate random Coordinax role flags.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    include
        If provided, only generate roles from this tuple. Otherwise, all roles
        are considered (Point, PhysDisp, etc.).
    exclude
        Roles to exclude from generation. Default is empty (no exclusions).

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax as cx
    >>> import coordinax_hypothesis.core as cxst

    >>> @given(role=cxst.roles())
    ... def test_any_role(role):
    ...     assert isinstance(role, cxr.AbstractRole)

    >>> @given(role=cxst.roles(include=(cxr.PhysDisp, cxr.PhysDisp)))
    ... def test_position_like_roles(role):
    ...     assert isinstance(role, (cxr.PhysDisp, cxr.PhysDisp))

    """
    role_cls = draw(role_classes(include=include, exclude=exclude))
    return role_cls()


@st.composite
def point_role(draw: st.DrawFn) -> cxr.Point:
    """Generate the Point role.

    Point represents an affine point on the manifold (not a tangent vector).
    This strategy always returns the Point role instance.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.roles as cxr
    >>> import coordinax_hypothesis.core as cxst

    >>> @given(role=cxst.point_role())
    ... def test_point_only(role):
    ...     assert isinstance(role, cxr.Point)

    """
    # Return Point role (use draw with st.just for consistency with Hypothesis patterns)
    return draw(st.just(cxr.point))


@st.composite
def physical_roles(draw: st.DrawFn) -> cxr.AbstractRole:
    """Generate physical tangent role flags (PhysDisp, PhysVel, PhysAcc).

    These are roles representing physical tangent vectors that require uniform
    physical dimension across components.

    Returns
    -------
    cxr.AbstractRole
        A physical tangent role instance: `coordinax.roles.phys_disp`,
        `coordinax.roles.phys_vel`, or `coordinax.roles.phys_acc`.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.roles as cxr
    >>> import coordinax_hypothesis.core as cxst

    >>> @given(role=cxst.physical_roles())
    ... def test_tangent_role(role):
    ...     # Only PhysDisp, Vel, PhysAcc (not Point)
    ...     assert isinstance(role, cxr.AbstractPhysRole)

    """
    role_cls = draw(st.sampled_from(PHYS_ROLES))
    return role_cls()


@st.composite
def coord_roles(draw: st.DrawFn) -> cxr.AbstractCoordRole:
    """Generate coordinate role flags (CoordDisp, CoordVel, CoordAcc).

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.roles as cxr
    >>> import coordinax_hypothesis.core as cxst

    >>> @given(role=cxst.coord_roles())
    ... def test_tangent_role(role):
    ...     # Only CoordDisp, CoordVel, CoordAcc (not Point)
    ...     assert isinstance(role, cxr.AbstractCoordRole)

    """
    role_cls = draw(st.sampled_from(COORD_ROLES))
    return role_cls()
