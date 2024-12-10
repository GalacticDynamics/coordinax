"""Coordinates."""

__all__ = ["AbstractCoordinate", "Coordinate"]


from textwrap import indent
from typing import NoReturn

import equinox as eqx
from plum import dispatch

from dataclassish import field_items, replace
from dataclassish.converters import Unless

from .base import AbstractReferenceFrame
from .xfm import TransformedReferenceFrame
from coordinax._src.operators import AbstractOperator
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

    #: The data of the coordinate. This is a `coordinax.Space` object, which is
    #: a collection of vectors. This can be constructed from a space object, or
    #: any input that can construct a `coordinax.Space` via
    #: `coordinax.Space.from_`.
    data: Space = eqx.field(converter=Space.from_)

    #: The reference frame of the coordinate as a
    #: `coordinax.frames.AbstractReferenceFrame` object. This can be
    #: from a reference frame object, or any input that can construct a
    #: `coordinax.frames.TransformedReferenceFrame` via
    #: `coordinax.frames.AbstractReferenceFrame.from_`.
    frame: AbstractReferenceFrame = eqx.field(
        converter=Unless(AbstractReferenceFrame, TransformedReferenceFrame.from_)
    )

    # ===============================================================
    # Coordinate API

    def to_frame(self, to_frame: AbstractReferenceFrame, /) -> "AbstractCoordinate":
        """Transform the coordinate to a specified frame.

        Examples
        --------
        >>> import coordinax as cx

        >>> cicrs = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.ICRS())

        >>> cgcf = cicrs.to_frame(cx.frames.Galactocentric())
        >>> cgcf
        Coordinate(
            data=Space({ 'length': CartesianPos3D( ... ) }),
            frame=Galactocentric( ... )
        )

        """
        op = self.transform_op(to_frame)
        new_data = op(self.data)
        return replace(self, data=new_data, frame=to_frame)

    # ===============================================================
    # Frame API

    def transform_op(self, to_frame: AbstractReferenceFrame, /) -> AbstractOperator:
        """Make a frame transform operator.

        Examples
        --------
        >>> import coordinax as cx

        >>> cicrs = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.ICRS())
        >>> cicrs.transform_op(cx.frames.Galactocentric())
        Pipe((
            GalileanRotation(rotation=f32[3,3]),
            GalileanSpatialTranslation(CartesianPos3D( ... )),
            GalileanRotation(rotation=f32[3,3]),
            VelocityBoost(CartesianVel3D( ... ))
        ))

        """
        return self.frame.transform_op(to_frame)

    # ===============================================================
    # Vector API

    def __neg__(self) -> "AbstractCoordinate":
        """Negate the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> coord = cx.Coordinate(data, cx.frames.ICRS())

        >>> (-coord).data["length"].x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return replace(self, data=-self.data)

    # ===============================================================
    # Quax API

    # TODO: is there a way to make this work?
    def aval(self) -> NoReturn:
        """Return the vector as a JAX array."""
        raise NotImplementedError  # pragma: no cover

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

    _repr_latex_ = __repr__  # TODO: implement this


##############################################################################


class Coordinate(AbstractCoordinate):
    """Coordinates are vectors in a reference frame.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
    ...                       cx.frames.ICRS())
    >>> coord
    Coordinate(
        data=Space({ 'length': CartesianPos3D( ... ) }),
        frame=ICRS()
    )

    Alternative Construction:

    >>> frame = cx.frames.ICRS()
    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> cx.Coordinate.from_({"data": data, "frame": frame})
    Coordinate(
        data=Space({ 'length': CartesianPos3D( ... ) }),
        frame=ICRS()
    )

    Changing Representation:

    >>> frame = cx.frames.ICRS()
    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> coord = cx.Coordinate(data, frame)

    >>> coord.vconvert(cx.SphericalPos)
    Coordinate(
        data=Space({ 'length': SphericalPos( ... ) }),
        frame=ICRS()
    )

    """

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @AbstractCoordinate.from_.dispatch
    def from_(
        cls: "type[Coordinate]",
        data: Space | AbstractPos,
        frame: AbstractReferenceFrame,
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
        return cls(data=data, frame=frame)

    @classmethod
    @AbstractCoordinate.from_.dispatch
    def from_(
        cls: "type[Coordinate]",
        data: Space | AbstractPos,
        base_frame: AbstractReferenceFrame,
        ops: AbstractOperator,
        /,
    ) -> "Coordinate":
        """Construct a coordinate from data and a frame.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> cx.Coordinate.from_(data, cx.frames.ICRS(), cx.ops.Identity())
        Coordinate(
            data=Space({ 'length': CartesianPos3D( ... ) }),
            frame=TransformedReferenceFrame(base_frame=ICRS(), xop=Identity())
        )

        """
        return cls(data=data, frame=TransformedReferenceFrame(base_frame, ops))


##############################################################################


@dispatch  # type: ignore[misc]
def vconvert(target: type[AbstractPos], w: Coordinate, /) -> Coordinate:
    """Transform the vector representation of a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> frame = cx.frames.ICRS()
    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> w = cx.Coordinate(data, frame)

    >>> cx.vconvert(cx.SphericalPos, w)
    Coordinate(
        data=Space({ 'length': SphericalPos( ... ) }),
        frame=ICRS()
    )

    """
    return replace(w, data=w.data.vconvert(target))


##############################################################################
# Transform operations


@AbstractOperator.__call__.dispatch  # type: ignore[attr-defined, misc]
def call(self: AbstractOperator, x: Coordinate, /) -> Coordinate:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
    ...                       cx.frames.ICRS())
    >>> coord
    Coordinate(
        data=Space({ 'length': CartesianPos3D( ... ) }),
        frame=ICRS()
    )

    >>> op = cx.ops.GalileanSpatialTranslation.from_([-1, -1, -1], "kpc")

    >>> new_coord = op(coord)
    >>> print(new_coord.data["length"])
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [0. 1. 2.]>

    """
    return replace(x, data=self(x.data))


##############################################################################
# Math operations


# @register(jax.lax.add_p)  # type: ignore[misc]
# def _add_crd_crd(w1: Coordinate, w2: Coordinate, /) -> Coordinate:
#     """Add two coordinates."""
#     # Transform w2 to w1's frame if necessary
#     new2 = w2 if w1.frame == w2.frame else w2.to_frame(w1.frame)
#     # Add the data
#     data = w1.data + new2.data

#     return replace(w1, data=data)
