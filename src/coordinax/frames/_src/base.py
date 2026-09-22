"""Base implementation of coordinate frames."""

__all__ = ("AbstractReferenceFrame",)

from collections.abc import Mapping
from typing import Any, cast

import equinox as eqx
import plum
import wadler_lindig as wl

from dataclassish import field_items

import coordinaxs.api.frames as cxfmapi
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
        kw.setdefault("use_short_name", True)
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

    >>> import coordinaxs.astro as cxastro
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


#: Cap on how much of the offending value the error quotes.
_MAX_SHOWN_CHARS = 120


@AbstractReferenceFrame.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[AbstractReferenceFrame], obj: Any, /) -> AbstractReferenceFrame:
    """Reject an unsupported input, naming the class and the argument.

    This catch-all is deliberately the least specific method, so any concrete
    registration wins over it. It exists so that an unsupported argument gets a
    domain error that says *what* was unsupported, instead of plum's generic
    ``NotFoundLookupError`` -- which, for this function, cannot even be built:
    the first implementation registered here is a jaxtyping wrapper when
    ``COORDINAX_ENABLE_RUNTIME_TYPECHECKING`` is set, so
    ``plum.Function.owner`` looks the owning class up in jaxtyping's module
    namespace and dies with ``KeyError('AbstractReferenceFrame')``.

    Examples
    --------
    >>> import coordinax.frames as cxf

    The message is a single line -- elided here only because it is longer
    than this file's line limit:

    >>> try:
    ...     cxf.Alice.from_(1)
    ... except TypeError as e:
    ...     print(e)
    Cannot construct 'Alice' from 1, of type 'int'. Supported input types ...

    The value is shown by `repr`, so a string is not mistaken for a name:

    >>> try:
    ...     cxf.Alice.from_("alice")
    ... except TypeError as e:
    ...     print(e)
    Cannot construct 'Alice' from 'alice', of type 'str'. Supported input ...

    """
    # `repr` over `str`: unambiguous (quotes survive) and stable across types.
    # But either can run to thousands of characters over dozens of lines for an
    # array or a coordinate dict, so flatten and cap it -- a one-line error is
    # worth more here than the tail of a 1000-element array.
    try:
        shown = " ".join(repr(obj).split())
    except Exception:  # noqa: BLE001
        # A broken `__repr__` must not replace the error we are here to raise.
        # Blanket, deliberately: a third-party `__repr__` may raise anything,
        # and whatever it is, the caller still needs to be told their input was
        # unsupported. `object.__repr__` cannot reach user code.
        shown = object.__repr__(obj)
    if len(shown) > _MAX_SHOWN_CHARS:
        shown = shown[: _MAX_SHOWN_CHARS - 3] + "..."
    msg = (
        f"Cannot construct {cls.__qualname__!r} from {shown}, of type "
        f"{type(obj).__qualname__!r}. Supported input types are listed by "
        f"`{cls.__qualname__}.from_.methods`."
    )
    raise TypeError(msg)
