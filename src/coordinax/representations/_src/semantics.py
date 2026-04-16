"""Vector semantic kind."""

__all__ = (
    "AbstractSemanticKind",
    # Location
    "Location",
    "loc",
)

import abc
import dataclasses
import functools as ft

from typing import Any, ClassVar, final

import jax.tree_util as jtu
import wadler_lindig as wl

import unxt as u
from dataclassish import field_items

import coordinax.charts as cxc


@jtu.register_static
class AbstractSemanticKind(metaclass=abc.ABCMeta):
    r"""Abstract base class for semantic kind.

    A semantic kind specifies the **meaning** attached to a represented
    geometric object, independent of the underlying geometric type, basis, and
    chart used to express its components.

    In the representation model used by `coordinax`, the full representation is
    determined by three orthogonal pieces of information:

    1. the geometric kind,
    2. the basis, and
    3. the semantic kind.

    This class provides the third of these pieces: it answers the question "what
    does this geometric object represent?" Examples include location for points,
    and later may include displacement, velocity, or acceleration for
    tangent-like objects.

    Mathematical Role:

    The semantic kind refines the interpretation of data within a fixed
    geometric type and basis.

    - For a **point** with `NoBasis`, the semantic kind `Location` indicates
      that the data represents where a point lies on a manifold.
    - For a **tangent vector**, different semantic kinds may distinguish
      displacement from velocity or acceleration, even when the underlying
      transformation law is the same.
    - For a **cotangent object**, semantic kinds may distinguish different dual
      interpretations without changing the underlying covector character.

    Thus semantic kind is distinct from:

    - a **chart**, which specifies how local coordinates are assigned,
    - a **geometry kind**, which specifies what sort of geometric object the
      data represents, and
    - a **basis**, which specifies in what basis the components are written when
      such a choice is meaningful.

    Examples
    --------
    >>> import coordinax.main as cx

    Construct the location semantic object directly:

    >>> semantic = cx.Location()

    With the semantic object, we construct a full representation for point data:

    >>> rep = cx.Representation(cx.point_geom, cx.no_basis, semantic)

    The representation can then be used with `cconvert` to convert point data
    between charts while preserving the fact that the data represents a
    location:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> cx.cconvert(p, cx.cart3d, rep, cx.sph3d, rep)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output is still point data representing the same location, but
    expressed in the target chart.

    Notes
    -----
    This is a static dispatch object and carries no runtime numerical data.
    Concrete subclasses should represent immutable semantic categories.

    """

    canonical_name: ClassVar[str | None] = None
    """Canonical name for the geometric kind."""

    # ===============================================================
    # Dimension API

    @classmethod
    @abc.abstractmethod
    def coord_dimensions(
        cls, chart: cxc.AbstractChart[Any, Any], /
    ) -> tuple[u.AbstractDimension | None, ...]:
        """Return the physical dimensions of the components for this semantic kind.

        Parameters
        ----------
        chart
            The chart defining component names and base coordinate dimensions.

        Returns
        -------
        tuple[u.AbstractDimension | None, ...]
            One entry per component: the physical dimension of that component
            under this semantic kind, or ``None`` if the component is
            dimensionless.

        """
        ...

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, canonical: bool = False, **kw: Any) -> wl.AbstractDoc:
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

        >>> semantic = cxr.Location()
        >>> wl.pprint(semantic)
        Location()

        >>> wl.pprint(semantic, canonical=True)
        loc

        """
        if canonical and self.canonical_name is not None:
            return wl.TextDoc(self.canonical_name)

        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=wl.named_objs(field_items(self), **kw),
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kw.get("indent", 4),
        )


# ===================================================================


@final
@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class Location(AbstractSemanticKind):
    r"""Location semantic kind.

    A location semantic kind indicates that the represented data specifies
    **where** a geometric object is, rather than how it is displaced, how fast
    it moves, or how it accelerates.

    Mathematical Definition:

    `Location` is the canonical semantic kind for point data. Let $M$ be a
    smooth manifold. A point is an element $p \in M$, and `Location` indicates
    that the represented data should be interpreted as the coordinates of that
    point in some chart.

    The semantic kind `Location` therefore does not change the underlying
    geometric transformation law: for point data, coordinates still transform by
    the ordinary chart transition map. Instead, it records the interpretation of
    the point-like data as an actual position on the manifold.

    Examples
    --------
    Construct the location semantic object directly:

    >>> import coordinax.representations as cxr
    >>> semantic = cxr.Location()

    Use it inside a full representation for point data:

    >>> rep = cxr.Representation(cxr.point_geom, cxr.no_basis, semantic)

    The representation can then be used with `cconvert` to convert point data
    between charts while preserving the fact that the data represents a
    location:

    >>> import coordinax.charts as cxc
    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> cxr.cconvert(p, cxc.cart3d, rep, cxc.sph3d, rep)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output is still point data representing the same location, but
    expressed in the target chart.

    Notes
    -----
    `Location` does not by itself imply that the represented object is a point,
    but in the current `coordinax` design it is primarily used as the semantic
    kind paired with `PointGeometry`.

    """

    canonical_name: ClassVar = "loc"
    """Canonical name for the location semantic kind."""

    @classmethod
    @ft.cache
    def coord_dimensions(
        cls, chart: cxc.AbstractChart[Any, Any], /
    ) -> tuple[u.AbstractDimension | None, ...]:
        """Return the physical dimensions of each component for a location.

        For a location semantic kind, the dimensions are determined solely by
        the chart: each component's dimension is taken directly from
        ``chart.coord_dimensions``.

        Examples
        --------
        >>> import coordinax.charts as cxc
        >>> import coordinax.representations as cxr
        >>> cxr.Location.coord_dimensions(cxc.cart3d)
        (PhysicalType('length'), PhysicalType('length'), PhysicalType('length'))

        """
        return tuple(  # ty: ignore[invalid-return-type]
            u.dimension(d) if d is not None else None for d in chart.coord_dimensions
        )


loc = Location()
"""Instance of `Location`."""
