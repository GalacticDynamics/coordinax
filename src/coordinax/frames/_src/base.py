"""Base implementation of coordinate frames."""

__all__ = ("AbstractReferenceFrame",)

from collections.abc import Mapping
from typing import Any, cast

import equinox as eqx
import plum
import wadler_lindig as wl

from dataclassish import field_items

import coordinax.api.frames as cxfmapi
from coordinax.transforms import AbstractTransform


class AbstractReferenceFrame(eqx.Module):
    """Base class for all reference frames."""

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @plum.dispatch.abstract
    def from_(
        cls: "type[AbstractReferenceFrame]", obj: Any, /
    ) -> "AbstractReferenceFrame":
        """Construct a reference frame."""
        raise NotImplementedError  # pragma: no cover

    # ---------------------------------------------------------------
    # Transformations

    def transform_to(self, to_frame: "AbstractReferenceFrame", /) -> AbstractTransform:
        """Make a frame transform operator.

        Parameters
        ----------
        to_frame : AbstractReferenceFrame
            The reference frame to transform to.

        Returns
        -------
        AbstractTransform
            The operator that transforms coordinates from this frame to
            `to_frame`.

        Examples
        --------
        >>> import coordinax.frames as cxf

        >>> op = cxf.alice.transform_to(cxf.alex)
        >>> op
        Composed(( ... ))

        >>> op = cxf.alex.transform_to(cxf.alice)
        >>> op
        Composed(( ... ))

        """
        out = cxfmapi.frame_transition(self, to_frame)
        return cast("AbstractTransform", out)

    def frame_transition(
        self, to_frame: "AbstractReferenceFrame", /
    ) -> AbstractTransform:
        """Backward-compatible alias for {meth}`transform_to`."""
        return self.transform_to(to_frame)

    # ---------------------------------------------------------------
    # Wadler-Lindig API

    def __pdoc__(self, **kw: object) -> wl.AbstractDoc:
        """Wadler-Lindig documentation for reference frames."""
        # Set defaults for pdoc kwargs
        kw.setdefault("include_params", False)
        kw.setdefault("short_arrays", "compact")
        kw.setdefault("use_short_names", True)
        kw.setdefault("named_unit", False)

        # Include only fields that differ from their default values.
        fitems = cast("list[tuple[str, Any]]", field_items(self))
        docs = wl.named_objs(
            [
                (k, v)
                for k, v in fitems
                if v is not self.__dataclass_fields__[k].default
            ],
            **kw,
        )

        # Format as ClassName(field1=value1, field2=value2, ...)
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kw.get("indent", 4),
        )


# =============================================================================
# Constructors


@AbstractReferenceFrame.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[AbstractReferenceFrame], obj: Mapping[str, Any], /
) -> AbstractReferenceFrame:
    """Construct a reference frame from a mapping.

    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice.from_({})
    >>> alice
    Alice()

    >>> alex = cxf.Alex.from_({})
    >>> print(alex)
    Alex()

    """
    return cls(**obj)


@AbstractReferenceFrame.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[AbstractReferenceFrame], obj: AbstractReferenceFrame, /
) -> AbstractReferenceFrame:
    """Construct a reference frame from another reference frame.

    Raises
    ------
    TypeError
        If the input object is not a subclass of the target class.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> cxf.AbstractReferenceFrame.from_(cxf.alice) is cxf.alice
    True

    >>> import coordinax.astro as cxastro
    >>> try:
    ...     cxastro.Galactocentric.from_(cxf.alice)
    ... except TypeError as e:
    ...     print(e)
    Cannot construct 'Galactocentric' from Alice()

    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls.__qualname__!r} from {obj}"
        raise TypeError(msg)

    return obj
