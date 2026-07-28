"""Shared canonical-name Wadler-Lindig repr for static dispatch kinds."""

__all__: tuple[str, ...] = ()

import dataclasses

from typing import Any, ClassVar

import wadler_lindig as wl

from dataclassish import field_items


class CanonicalStaticReprMixin:
    """Mixin providing the canonical/verbose repr shared by static kinds.

    `AbstractGeometry`, `AbstractBasis`, and `AbstractSemanticKind` are all
    static dispatch objects that carry no runtime data. Each renders as a short
    canonical name (e.g. ``point_geom``) when ``canonical_name`` is set, or as
    the verbose dataclass form otherwise. This mixin holds that single shared
    implementation.
    """

    canonical_name: ClassVar[str | None] = None
    """Canonical name for the kind, or ``None`` to always render verbosely."""

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, canonical: bool = True, **kw: Any) -> wl.AbstractDoc:
        """Generate a Wadler-Lindig docstring for this kind.

        Parameters
        ----------
        canonical
            Whether to use the canonical form of the kind in the docstring.
            E.g. `PointGeometry()` -> `point_geom`.
        **kw
            Additional keyword arguments to pass to the Wadler-Lindig docstring
            formatter.

        Examples
        --------
        >>> import wadler_lindig as wl
        >>> import coordinax.representations as cxr

        >>> geom = cxr.PointGeometry()
        >>> wl.pprint(geom, canonical=False)
        PointGeometry()

        >>> wl.pprint(geom, canonical=True)
        point_geom

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
        >>> repr(cxr.point_geom)
        'point_geom'
        >>> repr(cxr.coord_basis)
        'coord_basis'
        >>> repr(cxr.vel)
        'vel'

        """
        return wl.pformat(self, canonical=True)

    def __str__(self) -> str:
        """Return the verbose string representation.

        >>> import coordinax.representations as cxr
        >>> str(cxr.point_geom)
        'PointGeometry()'
        >>> str(cxr.coord_basis)
        'CoordinateBasis()'

        """
        return wl.pformat(self, canonical=False)
