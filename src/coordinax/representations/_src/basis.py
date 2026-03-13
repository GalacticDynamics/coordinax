"""Vector Basis."""

__all__ = ("AbstractBasis", "NoBasis", "nobasis")


import abc
import dataclasses

from typing import Any, ClassVar, Final, final

import jax.tree_util as jtu
import wadler_lindig as wl  # type: ignore[import-untyped]

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

    >>> rep = cx.Representation(cx.point_geom, basis, cx.location)

    The representation can then be used with
    `coordinax.representations.vconvert` to convert point data between charts
    while preserving both the represented geometric object and the fact that
    point data has no basis-dependent components:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> q = cx.vconvert(cx.sph3d, rep, cx.cart3d, rep, p)

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

        >>> basis = cxr.NoBasis()
        >>> wl.pprint(basis)
        NoBasis()

        >>> wl.pprint(basis, canonical=True)
        nobasis

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

    >>> rep = cxr.Representation(cxr.point_geom, basis, cxr.location)

    The representation can then be used with
    `coordinax.representations.vconvert` to convert point data between charts
    while preserving the fact that point data has no basis-dependent components:

    >>> import coordinax.charts as cxc
    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> q = cxr.vconvert(cxc.sph3d, rep, cxc.cart3d, rep, p)

    The output `q` is still point data, but expressed in the target chart.

    >>> import wadler_lindig as wl
    >>> wl.pprint(basis)
    NoBasis()

    >>> wl.pprint(basis, canonical=True)
    nobasis

    Notes
    -----
    `NoBasis` does not mean "no coordinates". It means that the represented
    object is not described by components in a basis-dependent linear space.

    """

    canonical_name: ClassVar = "nobasis"
    """Canonical name for the no-basis kind."""


nobasis: Final = NoBasis()
"""Instance of `NoBasis`."""
