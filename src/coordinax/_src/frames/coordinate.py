"""Coordinates."""

__all__ = ["AbstractCoordinate", "Coordinate"]


from textwrap import indent
from typing import Any, ClassVar, NoReturn, cast
from typing_extensions import override

import equinox as eqx
from plum import dispatch

from dataclassish import field_items, replace
from dataclassish.converters import Unless

from .base import AbstractReferenceFrame
from .xfm import TransformedReferenceFrame
from coordinax._src.operators import Identity
from coordinax._src.vectors.base import AbstractVector
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
    #: a collection of vectors.
    data: eqx.AbstractVar[Space]

    #: The reference frame of the coordinate as a
    #: `coordinax.frames.AbstractReferenceFrame` object.
    frame: eqx.AbstractVar[AbstractReferenceFrame]

    # ===============================================================
    # Coordinate API

    def to_frame(self, to_frame: AbstractReferenceFrame, /) -> "AbstractCoordinate":
        """Transform the coordinate to a specified frame.

        Examples
        --------
        >>> import coordinax as cx

        >>> cicrs = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.ICRS())

        >>> cicrs.to_frame(cx.frames.ICRS()) is cicrs
        True

        >>> cgcf = cicrs.to_frame(cx.frames.Galactocentric())
        >>> cgcf
        Coordinate(
            data=Space({ 'length': CartesianPos3D( ... ) }),
            frame=Galactocentric( ... )
        )

        """
        op = self.frame.transform_op(to_frame)

        # Special case for identity operations
        if isinstance(op, Identity):
            return self

        # Otherwise, apply the transformation and return a new coordinate
        new_data = op(self.data)
        return cast(AbstractCoordinate, type(self).from_(new_data, to_frame))

    # ===============================================================
    # Quax API

    # TODO: is there a way to make this work?
    def aval(self) -> NoReturn:
        """Return the vector as a JAX array."""
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Plum API

    __faithful__: ClassVar = True

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
        cls_name = type(self).__name__
        str_fs = ",\n".join(indent(f"{k}={v!r}", "    ") for k, v in field_items(self))
        return f"{cls_name}(\n{str_fs}\n)"

    _repr_latex_ = __repr__  # TODO: implement this

    def __str__(self) -> str:
        """Return string representation.

        Examples
        --------
        >>> coord = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.ICRS())
        >>> print(coord)
        Coordinate(
            data=Space({
            'length': <CartesianPos3D (x[kpc], y[kpc], z[kpc])
                [1 2 3]>
            }),
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

    Showing Frame Transformation:

    >>> space = cx.Space(length=cx.CartesianPos3D.from_([1.0, 0, 0], "pc"),
    ...                  speed=cx.CartesianVel3D.from_([1.0, 0, 0], "km/s"))

    >>> w=cx.Coordinate(
    ...     data=space,
    ...     frame=cx.frames.TransformedReferenceFrame(
    ...         cx.frames.Galactocentric(),
    ...         cx.ops.GalileanSpatialTranslation.from_([20, 0, 0], "kpc"),
    ...     ),
    ... )

    >>> w.to_frame(cx.frames.ICRS())
    Coordinate(
        data=Space({ 'length': CartesianPos3D( ... ), 'speed': CartesianVel3D( ... ) }),
        frame=ICRS()
    )

    >>> w.to_frame(cx.frames.ICRS()).data["length"]
    CartesianPos3D(
      x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("pc")),
      y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("pc")),
      z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("pc"))
    )

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
    # Vector API

    @override
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                   cx.frames.ICRS())
        >>> try: w._dimensionality()
        ... except NotImplementedError as e: print("not implemented")
        not implemented

        """
        # TODO: Space is currently not implemented.
        return self.data._dimensionality()  # noqa: SLF001

    @dispatch
    def __getitem__(self: "Coordinate", index: Any) -> "Coordinate":
        """Return Coordinate, with indexing applied to the data.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
        >>> w = cx.Coordinate.from_(data, cx.frames.ICRS())

        >>> print(w[0].data["length"])
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [1 2 3]>

        """
        return replace(self, data=self.data[index])

    @dispatch
    def __getitem__(self: "Coordinate", index: str) -> AbstractVector:
        """Return the data of the coordinate.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
        >>> w = cx.Coordinate.from_(data, cx.frames.ICRS())

        >>> print(w["length"])
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[1 2 3]
             [4 5 6]]>

        """
        return self.data[index]
