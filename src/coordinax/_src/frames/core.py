"""Coordinates."""

__all__ = ["AbstractCoordinate", "Coordinate"]


from collections.abc import Mapping
from textwrap import indent
from typing import Any, NoReturn

import equinox as eqx
import jax
from plum import dispatch
from quax import register

from dataclassish import field_items, replace

from .base import AbstractReferenceFrame
from coordinax._src.operators.base import AbstractOperator, op_call_dispatch
from coordinax._src.operators.identity import IdentityOperator
from coordinax._src.vectors.base import AbstractPos, AbstractVector
from coordinax._src.vectors.space.core import Space


# TODO: parametrize by the vector type(s), when Space is parametrized,
# and the frame type(s)
class AbstractCoordinate(AbstractVector):
    """Coordinates are vectors in a reference frame.

    See Also
    --------
    `coordinax.Coordinate` for a concrete implementation.

    """

    #: The data of the coordinate.
    data: Space = eqx.field(converter=Space.from_)

    #: The base reference frame of the coordinate.  Data is in the reference
    # frame defined by the transformations (`coordinax.AbstractCoordinate.ops`)
    # from the base frame (`coordinax.AbstractCoordinate.base_frame`).
    base_frame: AbstractReferenceFrame = eqx.field(
        converter=AbstractReferenceFrame.from_
    )

    #: Additional transformations from the `frame`. The daata
    ops: AbstractOperator = eqx.field(
        converter=AbstractOperator.from_, default=IdentityOperator()
    )

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch  # type: ignore[misc]
    def from_(
        cls: "type[AbstractCoordinate]", obj: Mapping[str, Any], /
    ) -> "AbstractCoordinate":
        """Construct a coordinate from a Mapping.

        Examples
        --------
        >>> import coordinax as cx

        >>> frame = cx.frames.ICRS()
        >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> cx.Coordinate.from_({"data": data, "frame": frame})
        Coordinate(
            data=Space({ 'length': CartesianPos3D( ... ) }),
            frame=ICRS()
        )

        """
        return cls(**obj)

    # ===============================================================
    # Coordinate API

    def to_frame(self, to_frame: AbstractReferenceFrame, /) -> "AbstractCoordinate":
        """Transform the coordinate to a specified frame.

        Examples
        --------
        >>> import coordinax as cx

        >>> frame = cx.frames.ICRS()
        >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> coord = cx.Coordinate(data, frame)

        >>> coord.to_frame(cx.frames.Galactocentric())

        """
        op = self.transform_op(to_frame)
        return replace(self, data=op(self.data), frame=to_frame)

    # ===============================================================
    # Frame API

    def transform_op(self, to_frame: AbstractReferenceFrame, /) -> AbstractOperator:
        """Make a frame transform operator."""
        return self.ops.inverse | self.base_frame.transform_op(to_frame)

    # ===============================================================
    # Vector API

    def __neg__(self) -> "AbstractCoordinate":
        """Negate the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> coord = cx.Coordinate(data, cx.frames.ICRS())

        >>> -coord.data["length"].x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return replace(self, data=-self.data)

    def represent_as(self, *args: Any, **kwargs: Any) -> "AbstractCoordinate":
        """Change the representation of the data.

        Examples
        --------
        >>> import coordinax as cx

        >>> frame = cx.frames.ICRS()
        >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> coord = cx.Coordinate(data, frame)

        >>> coord.represent_as(cx.SphericalPos)
        Coordinate(
            data=Space({ 'length': SphericalPos( ... ) }),
            frame=ICRS()
        )

        """
        return represent_as(self, *args, **kwargs)

    # ===============================================================
    # Quax API

    def aval(self) -> NoReturn:
        """Return the vector as a JAX array."""
        raise NotImplementedError  # TODO: implement this

    # ===============================================================
    # Plum API

    __faithful__ = True

    # ===============================================================
    # Python API

    def __repr__(self) -> str:
        """Return string representation.

        Examples
        --------
        >>> import coordinax as cx

        >>> frame = cx.frames.ICRS()
        >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> print(repr(cx.Coordinate(data, frame)))
        Coordinate(
            data=Space({ 'length': CartesianPos3D( ... ) }),
            frame=ICRS()
        )

        """
        # NOTE: this is necessary because equinox __repr__ isn't great
        str_fs = ",\n".join(indent(f"{k}={v}", "    ") for k, v in field_items(self))
        return f"{type(self).__name__}(\n{str_fs}\n)"


##############################################################################


class Coordinate(AbstractCoordinate):
    """Coordinates are vectors in a reference frame.

    Examples
    --------
    >>> import coordinax as cx

    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> coord = cx.Coordinate(data, cx.frames.ICRS())
    >>> coord
    Coordinate(
        data=Space({ 'length': CartesianPos3D( ... ) }),
        frame=ICRS()
    )

    Coordinates support arithmetic operations.

    >>> coord + coord

    If the coords are in different frames, they will be transformed.

    >>> ocoord = cx.Coordinate(data, cx.frames.Galactocentric())

    """

    @classmethod
    @AbstractCoordinate.from_._f.dispatch  # type: ignore[misc]  # noqa: SLF001
    def from_(
        cls: "type[Coordinate]",
        data: Space | AbstractPos,
        base_frame: AbstractReferenceFrame,
        ops: AbstractOperator = IdentityOperator(),
        /,
    ) -> "Coordinate":
        """Construct a coordinate from data and a frame.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> cx.Coordinate.from_(data, cx.frames.ICRS())
        Coordinate(
            data=Space({ 'length': CartesianPos3D( ... ) }),
            frame=ICRS()
        )

        """
        return cls(data=data, base_frame=base_frame, ops=ops)


##############################################################################


@dispatch  # type: ignore[misc]
def represent_as(w: Coordinate, target: type[AbstractPos], /) -> Coordinate:
    """Transform the representation of a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> frame = cx.frames.ICRS()
    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> w = cx.Coordinate(data, frame)

    >>> cx.represent_as(w, cx.SphericalPos)
    Coordinate(
        data=Space({ 'length': SphericalPos( ... ) }),
        frame=ICRS()
    )

    """
    return replace(w, data=w.data.represent_as(target))


##############################################################################
# Transform operations


@op_call_dispatch  # type: ignore[misc]
def call(self: AbstractOperator, x: Coordinate, /) -> Coordinate:
    """Dispatch to the operator's `__call__` method."""
    return replace(x, data=self(x.data))


##############################################################################


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_crd_crd(w1: Coordinate, w2: Coordinate, /) -> Coordinate:
    """Add two coordinates."""
    if w1.frame != w2.frame:
        new2 = w2.transform_op(w1.frame)(w2)

    return replace(w1, data=w1.data + new2.data)
