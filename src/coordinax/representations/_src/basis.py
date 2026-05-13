"""Vector Basis."""

__all__ = (
    "AbstractBasis",
    # No basis
    "NoBasis",
    "no_basis",
    # Linear bases
    "AbstractLinearBasis",
    "CoordinateBasis",
    "coord_basis",
    "PhysicalBasis",
    "phys_basis",
)


import abc
import dataclasses

from typing import Any, ClassVar, Final, final

import jax.tree_util as jtu
import wadler_lindig as wl

from dataclassish import field_items


@jtu.register_static
class AbstractBasis(metaclass=abc.ABCMeta):
    r"""Abstract base class for basis kind.

    A basis kind specifies the **component basis** in which data is expressed,
    when such a choice is meaningful, independent of the underlying geometric
    type and independent of which coordinate chart is used.

    In the representation model used by `coordinax`, the full representation is
    determined by three orthogonal pieces of information:

    1. the geometric kind,
    2. the basis, and
    3. the semantic kind.

    This class provides the second of these pieces: it answers the question "in
    what basis are the components written?" Examples include no basis for affine
    point data, and coordinate and physical bases for tangent or cotangent
    objects.

    Mathematical Role:

    The basis determines how component values are interpreted within a fixed
    geometric type.

    - For a **point**, there is typically no basis choice: points are affine
      objects rather than vectors in a linear space.
    - For a **tangent vector**, one may distinguish between components written
      in a coordinate basis and components written in an orthonormal physical
      basis.
    - For a **cotangent vector**, an analogous distinction may be made between
      the dual coordinate basis and other dual frames.

    Thus basis is distinct from:

    - a **chart**, which specifies how local coordinates are assigned,
    - a **geometry kind**, which specifies what sort of geometric object the
      data represents, and
    - a **semantic kind**, which specifies the interpretation of an object of a
      given geometric type (for example, location, displacement, velocity, or
      acceleration).

    Examples
    --------
    >>> import coordinax.main as cx

    Construct the no-basis object directly:

    >>> basis = cx.NoBasis()

    With the basis object, we construct a full representation for point data:

    >>> rep = cx.Representation(cx.point_geom, basis, cx.loc)

    The representation can then be used with
    `coordinax.representations.cconvert` to convert point data between charts
    while preserving both the represented geometric object and the fact that
    point data has no basis-dependent components:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> q = cx.cconvert(p, cx.cart3d, rep, cx.sph3d)

    The output `q` is still point data, but expressed in the target chart.

    Notes
    -----
    This is a static dispatch object and carries no runtime numerical data.
    Concrete subclasses should represent immutable basis categories.

    """

    canonical_name: ClassVar[str | None] = None
    """Canonical name for the basis kind."""

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, canonical: bool = True, **kw: Any) -> wl.AbstractDoc:
        """Generate a Wadler-Lindig docstring for this Basis.

        Parameters
        ----------
        canonical
            Whether to use the canonical forms of the representation in the
            docstring. E.g. `PointGeometry()` -> `point_geom`.
        **kw
            Additional keyword arguments to pass to the Wadler-Lindig docstring
            formatter.

        Examples
        --------
        >>> import wadler_lindig as wl
        >>> import coordinax.representations as cxr

        >>> basis = cxr.NoBasis()
        >>> wl.pprint(basis, canonical=False)
        NoBasis()

        >>> wl.pprint(basis, canonical=True)
        no_basis

        """
        if canonical and self.canonical_name is not None:
            return wl.TextDoc(self.canonical_name)

        items = field_items(self) if dataclasses.is_dataclass(self) else ()
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=wl.named_objs(items, **kw),
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kw.get("indent", 4),
        )

    def __repr__(self) -> str:
        """Return the canonical string representation.

        >>> import coordinax.representations as cxr
        >>> repr(cxr.coord_basis)
        'coord_basis'
        >>> repr(cxr.CoordinateBasis())
        'coord_basis'

        """
        return wl.pformat(self, canonical=True)

    def __str__(self) -> str:
        """Return the verbose string representation.

        >>> import coordinax.representations as cxr
        >>> str(cxr.coord_basis)
        'CoordinateBasis()'

        """
        return wl.pformat(self, canonical=False)


@final
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class NoBasis(AbstractBasis):
    r"""No-basis kind.

    A no-basis kind indicates that the represented data does **not** live in a
    basis-dependent linear space, so there is no meaningful choice of components
    with respect to a basis.

    Mathematical Definition:

    `NoBasis` is used for geometric objects whose representation is not given by
    expansion in a vector-space basis. The canonical example is a point on a
    manifold.

    Let $M$ be a smooth manifold. A point is an element $p \in M$, not a vector
    in a tangent space $T_p M$. Accordingly, point data has no associated basis:
    it is represented by chart coordinates, and under a change of chart those
    coordinates transform by the ordinary chart transition map, not by a change
    of basis.

    Examples
    --------
    Construct the no-basis object directly:

    >>> import coordinax.representations as cxr
    >>> basis = cxr.NoBasis()

    Use it inside a full representation for point data:

    >>> rep = cxr.Representation(cxr.point_geom, basis, cxr.loc)

    The representation can then be used with
    `coordinax.representations.cconvert` to convert point data between charts
    while preserving the fact that point data has no basis-dependent components:

    >>> import coordinax.charts as cxc
    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> q = cxr.cconvert(p, cxc.cart3d, rep, cxc.sph3d)

    The output `q` is still point data, but expressed in the target chart.

    >>> import wadler_lindig as wl
    >>> wl.pprint(basis, canonical=False)
    NoBasis()

    >>> wl.pprint(basis, canonical=True)
    no_basis

    Notes
    -----
    `NoBasis` does not mean "no coordinates". It means that the represented
    object is not described by components in a basis-dependent linear space.

    """

    canonical_name: ClassVar = "no_basis"
    """Canonical name for the no-basis kind."""


no_basis: Final = NoBasis()
"""Instance of `NoBasis`."""


@jtu.register_static
class AbstractLinearBasis(AbstractBasis, metaclass=abc.ABCMeta):
    r"""Abstract base class for linear (basis-dependent) basis kinds.

    A linear basis kind indicates that component data is expressed as
    components with respect to a specific choice of basis vectors in a
    tangent space or other linear space.

    In the representation model used by `coordinax`, `AbstractLinearBasis`
    refines `AbstractBasis` by indicating that the object lives in a
    basis-dependent linear space, in contrast to `NoBasis` which is used for
    affine point data.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> isinstance(cxr.CoordinateBasis(), cxr.AbstractLinearBasis)
    True

    """


@final
@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class CoordinateBasis(AbstractLinearBasis):
    r"""Coordinate basis kind.

    A coordinate basis kind indicates that tangent-vector components are
    expressed in the **coordinate basis** induced by the current chart. In a
    coordinate basis, the basis vectors are the partial derivative operators
    $\partial/\partial x^i$ associated with the chart coordinates $x^i$.

    Mathematical Definition:

    Let $(U, \varphi)$ be a chart on a smooth manifold $M$. The coordinate
    basis at a point $p \in U$ consists of the tangent vectors
    $\partial_i = \partial/\partial x^i|_p$. A tangent vector $v \in T_p M$
    expressed in the coordinate basis has components $v^i$ such that
    $v = v^i \partial_i$.

    Under a change of chart, coordinate-basis components transform by the
    Jacobian matrix of the chart transition map.

    Examples
    --------
    Construct the coordinate basis object directly:

    >>> import coordinax.representations as cxr
    >>> basis = cxr.CoordinateBasis()

    Use it inside a full representation for tangent data:

    >>> rep = cxr.Representation(cxr.tangent_geom, basis, cxr.dpl)

    Notes
    -----
    `CoordinateBasis` does not carry unit-length information. Components in a
    coordinate basis are not necessarily dimensionless: for example, the
    $\partial_r$ component of a vector in spherical coordinates carries units
    of inverse length relative to the physical basis.

    """

    canonical_name: ClassVar = "coord_basis"
    """Canonical name for the coordinate basis kind."""


coord_basis: Final = CoordinateBasis()
"""Instance of `CoordinateBasis`."""


@final
@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class PhysicalBasis(AbstractLinearBasis):
    r"""Physical (orthonormal) basis kind.

    A physical basis kind indicates that tangent-vector components are expressed
    in an **orthonormal physical basis**. In a physical basis, the basis vectors
    are unit-length and mutually orthogonal with respect to the metric.

    Mathematical Definition:

    Let $M$ carry a Riemannian metric $g$ and let $(U, \varphi)$ be a chart on
    $M$. The physical basis at a point $p \in U$ consists of unit vectors
    $\hat{e}_i = \partial_i / \sqrt{g_{ii}}$ (for orthogonal charts). A tangent
    vector $v \in T_p M$ expressed in the physical basis has components
    $\hat{v}^i$ such that $v = \hat{v}^i \hat{e}_i$.

    Under a change of chart, physical-basis components transform by both the
    Jacobian and the normalization factors of the chart transition.

    Examples
    --------
    Construct the physical basis object directly:

    >>> import coordinax.representations as cxr
    >>> basis = cxr.PhysicalBasis()

    Use it inside a full representation for tangent data:

    >>> rep = cxr.Representation(cxr.tangent_geom, basis, cxr.vel)

    Notes
    -----
    `PhysicalBasis` components have consistent physical dimensions across all
    charts, unlike coordinate-basis components.

    """

    canonical_name: ClassVar = "phys_basis"
    """Canonical name for the physical basis kind."""


phys_basis: Final = PhysicalBasis()
"""Instance of `PhysicalBasis`."""
