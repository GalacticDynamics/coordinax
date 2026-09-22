"""Base implementation of coordinate frames."""

__all__ = ("AbstractReferenceFrame", "is_same_frame")

from collections.abc import Mapping
from typing import Any, cast

import equinox as eqx
import jax
import plum
import wadler_lindig as wl

from dataclassish import field_items

import coordinaxs.api.frames as cxfmapi
from coordinax.transforms import AbstractTransform


def is_same_frame(
    from_frame: "AbstractReferenceFrame", to_frame: "AbstractReferenceFrame", /
) -> bool:
    """Whether two frames are *statically known* to be the same frame.

    Frames are `equinox.Module` pytrees, so ``from_frame == to_frame`` over
    array-valued fields yields a 0-d `jax.Array`, not a `bool`. Putting that in
    an ``if`` works eagerly but raises ``TracerBoolConversionError`` under
    `jax.jit`; plain ``from_frame is to_frame`` is trace-safe but misses two
    equal-but-distinct frames. This predicate is both: structural when the
    leaves are concrete, and `False` -- fall through to the general, always
    correct transform -- when any leaf is traced.

    Examples
    --------
    >>> import jax
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.astro as cxastro
    >>> import coordinax.frames as cxf

    Equal-but-distinct frames are the same frame:

    >>> cxf.is_same_frame(cxastro.Galactocentric(), cxastro.Galactocentric())
    True

    >>> cxf.is_same_frame(
    ...     cxastro.Galactocentric(), cxastro.Galactocentric(roll=u.Q(10, "deg"))
    ... )
    False

    Frames of different types never are, even when their fields agree:

    >>> cxf.is_same_frame(cxastro.ICRS(), cxastro.Galactic())
    False

    Under tracing sameness is not statically knowable, so the answer is `False`
    and the caller builds the general transform:

    >>> gc = cxastro.Galactocentric()
    >>> jax.jit(lambda a, b: jnp.asarray(cxf.is_same_frame(a, b)))(gc, gc)
    Array(False, dtype=bool)

    """
    if from_frame is to_frame:
        return True
    if type(from_frame) is not type(to_frame):
        return False
    eq = eqx.tree_equal(from_frame, to_frame)
    # `tree_equal` is `True`/`False` only when every leaf is a non-array; with
    # array leaves it is a 0-d array, concrete or traced. A traced one cannot be
    # decided at trace time, so report "not the same" and let the caller build
    # the general transform.
    if isinstance(eq, jax.core.Tracer):  # ty: ignore[possibly-missing-submodule]
        return False
    return bool(eq)


class AbstractReferenceFrame(eqx.Module, is_abstract=True):
    """Base class for all reference frames.

    Abstract: a category for typing and dispatch, not a frame. Concrete
    frames subclass it.

    >>> import coordinax.frames as cxf
    >>> try:
    ...     cxf.AbstractReferenceFrame()
    ... except TypeError as e:
    ...     print(e)
    Cannot instantiate abstract `equinox.Module`.

    """

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
