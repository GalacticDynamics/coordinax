"""Coordinates."""

__all__ = ["AbstractCoordinate", "Coordinate"]


from typing import Any, ClassVar, cast
from typing_extensions import override

import equinox as eqx
import jax
import wadler_lindig as wl
from plum import dispatch

from dataclassish import field_items, replace
from dataclassish.converters import Unless

from .base import AbstractReferenceFrame
from .xfm import TransformedReferenceFrame
from coordinax._src.operators import Identity
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.collection.core import Space


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
        out = self.__class__.from_(new_data, to_frame)
        return cast(AbstractCoordinate, out)

    # ===============================================================
    # Quax API

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        return self.data.aval()

    # ===============================================================
    # Plum API

    __faithful__: ClassVar = True

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation.

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
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=wl.named_objs(tuple(field_items(self)), **kwargs),
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kwargs.get("indent", 4),
        )

    # ===============================================================
    # Python API

    def __str__(self) -> str:
        """Return string representation.

        Examples
        --------
        >>> coord = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.ICRS())
        >>> print(coord)
        Coordinate(
            data=Space({
            'length': <CartesianPos3D: (x, y, z) [kpc]
                [1 2 3]>
            }),
            frame=ICRS()
        )

        """
        return wl.pformat(self, width=88, vector_form=True)

    # ===============================================================
    # IPython API

    _repr_latex_ = lambda self: wl.pformat(self)  # noqa: E731  # TODO: implement this


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
      x=Quantity(-1587.6683, unit='pc'),
      y=Quantity(-24573.762, unit='pc'),
      z=Quantity(-13583.504, unit='pc')
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
        return self.data._dimensionality()

    @dispatch
    def __getitem__(self: "Coordinate", index: Any) -> "Coordinate":
        """Return Coordinate, with indexing applied to the data.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
        >>> w = cx.Coordinate.from_(data, cx.frames.ICRS())

        >>> print(w[0].data["length"])
        <CartesianPos3D: (x, y, z) [kpc]
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
        <CartesianPos3D: (x, y, z) [kpc]
            [[1 2 3]
             [4 5 6]]>

        """
        return self.data[index]
