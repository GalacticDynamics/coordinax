"""Vector representation."""

__all__ = (
    "Representation",
    "point",
    # Tangent singletons
    "coord_disp",
    "coord_vel",
    "coord_acc",
    "phys_disp",
    "phys_vel",
    "phys_acc",
)


import dataclasses

from typing import TYPE_CHECKING, Any, Final, Generic, cast, final
from typing_extensions import TypeVar

import jax.tree_util as jtu
import wadler_lindig as wl

from dataclassish import field_items, field_values

from .basis import AbstractBasis, coord_basis, no_basis, phys_basis
from .geom import AbstractGeometry, point_geom, tangent_geom
from .semantics import AbstractSemanticKind, acc, dpl, loc, vel

if TYPE_CHECKING:
    from collections.abc import Iterable

GeomT = TypeVar("GeomT", bound=AbstractGeometry, default=AbstractGeometry)
BasisT = TypeVar("BasisT", bound=AbstractBasis, default=AbstractBasis)
SemanticT = TypeVar(
    "SemanticT", bound=AbstractSemanticKind, default=AbstractSemanticKind
)


@jtu.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class Representation(Generic[GeomT, BasisT, SemanticT]):
    r"""Representation of geometric component data.

    A representation specifies **what kind of geometric object** component data
    is meant to represent, independently of the chart used to write down the
    coordinates or components.

    In `coordinax`, a representation is the ordered triple

    $$ R = (K, B, S), $$

    where:

    - $K$ is the geometric kind (`coordinax.representations.AbstractGeometry`),
    - $B$ is the basis kind (`coordinax.representations.AbstractBasis`), and
    - $S$ is the semantic kind (`coordinax.representations.AbstractSemanticKind`).

    Thus a `Representation` answers three distinct questions:

    1. **What sort of geometric object is this?**
       For example, a point.
    2. **In what basis are its components written?**
       For example, no basis for affine point data.
    3. **What does the object mean?**
       For example, a location on a manifold.

    A representation is therefore **not** the same thing as a chart.

    - A **chart** specifies how a manifold is coordinatized locally: component
      names, ordering, dimensionalities, and the coordinate map into
      $\mathbb{R}^n$.
    - A **representation** specifies how the data should be interpreted
      geometrically.

    Equivalently: the chart determines the coordinate system, while the
    representation determines the geometric role of the data written in that
    coordinate system.

    For the current point-focused design, the canonical representation is

    $$ (\mathrm{PointGeometry},\, \mathrm{NoBasis},\, \mathrm{Location}). $$

    This indicates that the data represents a point on a manifold, that it does
    not live in a basis-dependent linear space, and that its semantic meaning is
    a location.

    Parameters
    ----------
    geom_kind : AbstractGeometry
        The geometric kind of the represented object.
    basis : AbstractBasis
        The basis kind in which components are expressed.
    semantic_kind : AbstractSemanticKind
        The semantic interpretation attached to the represented object.

    Examples
    --------
    Construct the canonical point representation directly:

    >>> import coordinax.representations as cxr
    >>> rep = cxr.Representation(cxr.PointGeometry(), cxr.NoBasis(), cxr.Location())

    `coordinax` also provides the predefined point representation:

    >>> rep == cxr.point
    True

    Use the representation with `cconvert` to convert point data between charts
    while preserving its geometric interpretation:

    >>> import coordinax.charts as cxc
    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> cxr.cconvert(p, cxc.cart3d, cxr.point, cxc.sph3d, cxr.point)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output represents the same point, but in the target chart.

    Notes
    -----
    `Representation` is a static, immutable descriptor object. It carries no
    runtime numerical data itself; it only describes how such data should be
    interpreted.

    """

    geom_kind: GeomT
    """Geometric kind of the represented object."""

    basis: BasisT
    """Basis kind in which components are expressed."""

    semantic_kind: SemanticT
    """Semantic interpretation attached to the represented object."""

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(
        self, *, field_names: bool = True, canonical: bool = False, **kw: Any
    ) -> wl.AbstractDoc:
        """Generate a Wadler-Lindig docstring for this representation.

        Parameters
        ----------
        field_names
            Whether to include field names in the docstring.
        canonical
            Whether to use the canonical forms of the representation in the
            docstring.
        **kw
            Additional keyword arguments to pass to the Wadler-Lindig docstring
            formatter.

        Examples
        --------
        >>> import wadler_lindig as wl
        >>> import coordinax.representations as cxr

        >>> rep = cxr.Representation(cxr.PointGeometry(), cxr.NoBasis(), cxr.Location())
        >>> wl.pprint(rep)
        Representation(geom_kind=PointGeometry(), basis=NoBasis(),
                       semantic_kind=Location())

        >>> wl.pprint(rep, field_names=False)
        Representation(PointGeometry(), NoBasis(), Location())

        >>> wl.pprint(rep, canonical=True)
        point

        """
        if canonical and self in CANONICAL_REPRESENTATIONS:
            return wl.TextDoc(CANONICAL_REPRESENTATIONS[self])

        if field_names:
            docs = wl.named_objs(field_items(self), canonical=canonical, **kw)
        else:
            fvals = cast("Iterable[Any]", field_values(self))
            docs = [wl.pdoc(v, canonical=canonical, **kw) for v in fvals]

        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kw.get("indent", 4),
        )


point = Representation(point_geom, no_basis, loc)
r"""Predefined point representation.

$(\mathrm{PointGeometry}, \mathrm{NoBasis}, \mathrm{Location})$.
"""

coord_disp = Representation(tangent_geom, coord_basis, dpl)
r"""Coordinate-basis displacement representation.

$(\mathrm{TangentGeometry}, \mathrm{CoordinateBasis}, \mathrm{Displacement})$.
"""

coord_vel = Representation(tangent_geom, coord_basis, vel)
r"""Coordinate-basis velocity representation.

$(\mathrm{TangentGeometry}, \mathrm{CoordinateBasis}, \mathrm{Velocity})$.
"""

coord_acc = Representation(tangent_geom, coord_basis, acc)
r"""Coordinate-basis acceleration representation.

$(\mathrm{TangentGeometry}, \mathrm{CoordinateBasis}, \mathrm{Acceleration})$.
"""

phys_disp = Representation(tangent_geom, phys_basis, dpl)
r"""Physical-basis displacement representation.

$(\mathrm{TangentGeometry}, \mathrm{PhysicalBasis}, \mathrm{Displacement})$.
"""

phys_vel = Representation(tangent_geom, phys_basis, vel)
r"""Physical-basis velocity representation.

$(\mathrm{TangentGeometry}, \mathrm{PhysicalBasis}, \mathrm{Velocity})$.
"""

phys_acc = Representation(tangent_geom, phys_basis, acc)
r"""Physical-basis acceleration representation.

$(\mathrm{TangentGeometry}, \mathrm{PhysicalBasis}, \mathrm{Acceleration})$.
"""


CANONICAL_REPRESENTATIONS: Final[dict[Representation[Any, Any, Any], str]] = {
    point: "point",
    coord_disp: "coord_disp",
    coord_vel: "coord_vel",
    coord_acc: "coord_acc",
    phys_disp: "phys_disp",
    phys_vel: "phys_vel",
    phys_acc: "phys_acc",
}
