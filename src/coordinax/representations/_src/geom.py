"""Vector Geometry."""

__all__ = (
    "AbstractGeometry",
    # Point
    "PointGeometry",
    "point_geom",
    # Tangent
    "TangentGeometry",
    "tangent_geom",
)


import abc
import dataclasses

from typing import Any, ClassVar, final

import jax.tree_util as jtu
import wadler_lindig as wl

from dataclassish import field_items


@jtu.register_static
class AbstractGeometry(metaclass=abc.ABCMeta):
    r"""Abstract base class for geometric kind.

    A geometric kind specifies the underlying **geometric type** of the data,
    independent of any coordinate chart or basis in which components may be
    written.

    In the representation model used by `coordinax`, the full representation is
    determined by three orthogonal pieces of information:

    1. the geometric kind,
    2. the basis, and
    3. the semantic kind.

    This class provides the first of these pieces: it answers the question "what
    sort of geometric object is this?" Examples include a point.

    Mathematical Role:

    The geometric kind determines the abstract transformation behavior of the
    object under coordinate changes.

    - For a **point**, coordinates transform by the chart transition map.
    - For a **tangent vector**, components transform by the pushforward
      (Jacobian) of the coordinate map.
    - For a **cotangent vector**, components transform by the pullback.

    Thus geometric kind is distinct from:

    - a **chart**, which specifies how local coordinates are assigned, and
    - a **semantic kind**, which specifies the interpretation of an object of a
      given geometric type (for example, location, displacement, velocity, or
      acceleration).

    Examples
    --------
    >>> import coordinax.main as cx

    Construct the point geometry object directly:

    >>> geom = cx.PointGeometry()

    With the geometry object, we construct a full representation for point data:

    >>> rep = cx.Representation(geom, cx.no_basis, cx.loc)

    The representation can then be used with `cconvert` to convert point data
    between charts while preserving the represented geometric object:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> cx.cconvert(p, cx.cart3d, rep, cx.sph3d, rep)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output is still point data, but expressed in the target chart.

    Notes
    -----
    This is a static dispatch object and carries no runtime numerical data.
    Concrete subclasses should represent immutable geometric categories.

    """

    canonical_name: ClassVar[str | None] = None
    """Canonical name for the geometric kind."""

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

        >>> geom = cxr.PointGeometry()
        >>> wl.pprint(geom)
        PointGeometry()

        >>> wl.pprint(geom, canonical=True)
        point_geom

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


@final
@dataclasses.dataclass(frozen=True, slots=True)
class PointGeometry(AbstractGeometry):
    r"""Point geometric kind.

    A point geometry indicates that component data should be interpreted as the
    coordinates of a **point** on a manifold, rather than as components of a
    vector- or covector-like object.

    Mathematical Definition:

    Let $M$ be a smooth manifold. A point is an element $p \in M$.

    Point data is therefore **affine**, not linear: points do not in general
    form a vector space, so operations such as adding two points are not
    geometrically defined. Under a change of chart, point coordinates transform
    by the ordinary chart transition map.

    Examples
    --------
    Construct the point geometry object directly:

    >>> import coordinax.representations as cxr
    >>> geom = cxr.PointGeometry()

    Use it inside a full representation for point data:

    >>> rep = cxr.Representation(geom, cxr.no_basis, cxr.loc)

    The representation can then be used with `cconvert` to convert point data
    between charts while preserving the represented geometric object:

    >>> import coordinax.charts as cxc
    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> cxr.cconvert(p, cxc.cart3d, rep, cxc.sph3d, rep)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    The output is still point data, but expressed in the target chart.

    Notes
    -----
    `PointGeometry` describes only the geometric kind. The chart still
    determines the coordinate system, component names, and coordinate domains.

    """

    canonical_name: ClassVar = "point_geom"
    """Canonical name for the point geometry kind."""


point_geom = PointGeometry()
"""Point geometric kind instance."""


@final
@jtu.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class TangentGeometry(AbstractGeometry):
    r"""Tangent-vector geometric kind.

    A tangent geometry indicates that component data should be interpreted as
    the components of a **tangent vector** at a point on a manifold, rather than
    as coordinates of a point.

    Mathematical Definition:

    Let $M$ be a smooth manifold and $p \in M$ a point. A tangent vector is an
    element $v \in T_p M$ of the tangent space at $p$.

    Tangent data is **linear**: tangent vectors form a vector space, so adding
    two tangent vectors at the same point is geometrically well-defined. Under a
    change of chart, tangent vector components transform by the Jacobian
    (pushforward) of the chart transition map.

    Examples
    --------
    Construct the tangent geometry object directly:

    >>> import coordinax.representations as cxr
    >>> geom = cxr.TangentGeometry()

    Use it inside a full representation for tangent data:

    >>> rep = cxr.Representation(geom, cxr.coord_basis, cxr.dpl)

    Notes
    -----
    This is a static dispatch object and carries no runtime numerical data.
    `TangentGeometry` describes only the geometric kind. The chart still
    determines the coordinate system, component names, and coordinate domains.

    """

    canonical_name: ClassVar = "tangent_geom"
    """Canonical name for the tangent geometry kind."""


tangent_geom = TangentGeometry()
"""Tangent geometric kind instance."""
