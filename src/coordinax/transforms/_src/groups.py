"""Frames sub-package.

This is the private implementation of the frames sub-package.

"""

__all__ = (
    "AbstractTransformGroup",
    "IdentityGroup",
    "DiffeomorphismGroup",
    "AffineGroup",
    "EuclideanGroup",
    "OrthogonalGroup",
    "SpecialOrthogonalGroup",
    "PoincareGroup",
    "LorentzGroup",
    "ProperOrthochronousLorentzGroup",
)

import abc
from functools import cache

from typing import ClassVar, NoReturn

import jax


@jax.tree_util.register_static
class AbstractTransformGroup(metaclass=abc.ABCMeta):
    """Abstract base class for transformation-group kinds.

    A transformation group is a group whose elements are maps acting on a space,
    typically preserving some chosen structure such as smooth, affine, or metric
    structure. Concrete subclasses represent named group kinds used for
    classification and dispatch.

    This class is not instantiated directly.
    """

    __declared_supergroups__: ClassVar[
        tuple[type["AbstractTransformGroup"], ...] | None
    ] = None
    __direct_supergroups__: ClassVar[tuple[type["AbstractTransformGroup"], ...]] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        direct_supergroups = cls.__dict__.get("__declared_supergroups__")
        if direct_supergroups is None:
            direct_supergroups = tuple(
                base
                for base in cls.__bases__
                if issubclass(base, AbstractTransformGroup)
                and base is not AbstractTransformGroup
            )
        cls.__direct_supergroups__ = direct_supergroups

    def __new__(cls, *_: object, **__: object) -> NoReturn:
        raise TypeError("AbstractTransformGroup cannot be instantiated directly")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class DiffeomorphismGroup(AbstractTransformGroup):
    """The group of smooth invertible self-maps of a manifold.

    Its elements are diffeomorphisms: smooth maps with smooth inverses.  This is
    the largest natural transformation group associated with a smooth manifold.
    """


class AffineGroup(DiffeomorphismGroup):
    r"""The group of affine transformations of an affine space.

    An affine transformation preserves affine combinations and may be written in
    coordinates as ``x \mapsto A x + b`` with ``A`` invertible.
    """


class EuclideanGroup(AffineGroup):
    """The group of Euclidean isometries of Euclidean space.

    Its elements preserve the Euclidean metric, and equivalently preserve
    distances. In coordinates these are rigid motions: rotations, reflections,
    and translations.
    """


class OrthogonalGroup(EuclideanGroup):
    """The group of orthogonal linear transformations.

    Its elements preserve a positive-definite inner product and fix the origin.
    In matrix form they satisfy ``Q^T Q = I``.
    """


class SpecialOrthogonalGroup(OrthogonalGroup):
    """The group of orientation-preserving orthogonal transformations.

    This is the subgroup of the orthogonal group with determinant ``+1``.  In
    Euclidean space its elements are rotations.
    """


# TODO: is this a special case of AffineGroup(4) ?
class PoincareGroup(DiffeomorphismGroup):
    """The group of isometries of Minkowski spacetime.

    It is the semidirect product of spacetime translations with the Lorentz
    group. Its elements preserve the Minkowski metric.
    """


class LorentzGroup(OrthogonalGroup):
    """The group of linear isometries of Minkowski spacetime.

    Its elements preserve the Minkowski bilinear form and fix the origin.  It is
    the indefinite-orthogonal group associated with spacetime signature.
    """

    __declared_supergroups__ = (PoincareGroup,)


class ProperOrthochronousLorentzGroup(LorentzGroup):
    """The identity component of the Lorentz group.

    Its elements preserve both spatial orientation and time orientation.  This
    is the subgroup of Lorentz transformations continuously connected to the
    identity.
    """


class IdentityGroup(AbstractTransformGroup):
    """The trivial group containing only the identity map.

    This is the one-element group whose sole transformation leaves every point
    fixed.
    """

    __declared_supergroups__ = (
        SpecialOrthogonalGroup,
        ProperOrthochronousLorentzGroup,
    )


@cache
def _supergroups_of(
    group: type[AbstractTransformGroup], /
) -> frozenset[type[AbstractTransformGroup]]:
    return frozenset({group}).union(
        *(_supergroups_of(parent) for parent in group.__direct_supergroups__)
    )


def _specificity(group: type[AbstractTransformGroup], /) -> int:
    return len(_supergroups_of(group))


def most_specific_group(
    groups: frozenset[type] | set[type] | tuple[type, ...], /
) -> type[AbstractTransformGroup]:
    """Return the most specific group from a transform's declared groups."""
    typed_groups = tuple(g for g in groups if issubclass(g, AbstractTransformGroup))
    if not typed_groups:
        msg = "Expected at least one transform group."
        raise ValueError(msg)
    return max(typed_groups, key=_specificity)


def least_common_supergroup(
    groups: tuple[type[AbstractTransformGroup], ...], /
) -> type[AbstractTransformGroup]:
    """Return the least common supergroup in the spec hierarchy."""
    if not groups:
        return IdentityGroup

    common = set.intersection(*(set(_supergroups_of(group)) for group in groups))
    return max(common, key=_specificity, default=DiffeomorphismGroup)
