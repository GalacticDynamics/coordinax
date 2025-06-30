"""Coordinates."""

__all__ = ["AbstractCoordinate", "Coordinate"]


from typing import Any, ClassVar, Literal, cast
from typing_extensions import override

import equinox as eqx
import jax
import wadler_lindig as wl
from plum import dispatch

import unxt as u
from dataclassish import field_items, replace
from dataclassish.converters import Unless

from .base import AbstractReferenceFrame
from .xfm import TransformedReferenceFrame
from coordinax._src.operators import Identity
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.collection.core import KinematicSpace


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
    data: eqx.AbstractVar[KinematicSpace]

    #: The reference frame of the coordinate as a
    #: `coordinax.frames.AbstractReferenceFrame` object.
    frame: eqx.AbstractVar[AbstractReferenceFrame]

    # ===============================================================
    # Coordinate API

    def to_frame(
        self, toframe: AbstractReferenceFrame, /, t: u.Quantity | None = None
    ) -> "AbstractCoordinate":
        """Transform the coordinate to a specified frame.

        Examples
        --------
        >>> import coordinax as cx

        >>> calice = cx.Coordinate(cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.Alice())

        >>> calice.to_frame(cx.frames.Alice()) is calice
        True

        >>> cfriend = calice.to_frame(cx.frames.FriendOfAlice())
        >>> cfriend
        Coordinate(
            KinematicSpace({ 'length': CartesianPos3D( ... ) }),
            frame=FriendOfAlice()
        )

        """
        op = self.frame.transform_op(toframe)

        # Special case for identity operations
        if isinstance(op, Identity):
            return self

        # Otherwise, apply the transformation and return a new coordinate
        new_data = op(self.data) if t is None else op(t, self.data)[1]
        out = self.__class__.from_(new_data, toframe)
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

    def __pdoc__(
        self,
        *,
        include_data_name: Literal["named", "vector", "map"] = "named",
        **kwargs: Any,
    ) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation.

        Parameters
        ----------
        include_data_name : {'named', ''}, optional
            If `named`, include the name of the data field.
            If `vector`, only include the data itself as a `KinematicSpace`.
            If `map`, only include the data as dict-like representation.
        **kwargs : Any, optional
            Additional keyword arguments to pass to `wl.pdoc`.

        Examples
        --------
        >>> import coordinax as cx

        >>> frame = cx.frames.Alice()
        >>> data = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> coord = cx.Coordinate(data, frame)
        >>> print(coord)
        Coordinate(
            { 'length': <CartesianPos3D: (x, y, z) [kpc]
                            [1 2 3]> },
            frame=Alice()
        )

        """
        if include_data_name == "named":
            docs = wl.named_objs(tuple(field_items(self)), **kwargs)
        else:
            docs = [
                wl.pdoc(
                    self.data if include_data_name == "vector" else self.data._data,
                    **kwargs,
                ),
                *wl.named_objs(tuple(field_items(self))[1:], **kwargs),
            ]

        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kwargs.get("indent", 4),
        )

    # ===============================================================
    # Python API

    def __repr__(self) -> str:
        """Return string representation.

        Examples
        --------
        >>> import coordinax as cx
        >>> coord = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.Alice())
        >>> print(repr(coord))
        Coordinate(
            KinematicSpace({ 'length': CartesianPos3D( ... ) }),
            frame=Alice()
        )

        """
        return wl.pformat(self, width=88, include_data_name="vector", vector_form=False)

    def __str__(self) -> str:
        """Return string representation.

        Examples
        --------
        >>> coord = cx.Coordinate(cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.Alice())
        >>> print(coord)
        Coordinate(
            { 'length': <CartesianPos3D: (x, y, z) [kpc]
                            [1 2 3]> },
            frame=Alice()
        )

        """
        return wl.pformat(self, width=88, include_data_name="map", vector_form=True)

    # ===============================================================
    # IPython API

    _repr_latex_ = lambda self: wl.pformat(self)  # noqa: E731  # TODO: implement this


##############################################################################


class Coordinate(AbstractCoordinate):
    """Coordinates are vectors in a reference frame.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc"),
    ...                       cx.frames.Alice())
    >>> coord
    Coordinate(
        KinematicSpace({ 'length': CartesianPos3D( ... ) }),
        frame=Alice()
    )

    Alternative Construction:

    >>> frame = cx.frames.Alice()
    >>> data = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> cx.Coordinate.from_({"data": data, "frame": frame})
    Coordinate(
        KinematicSpace({ 'length': CartesianPos3D( ... ) }),
        frame=Alice()
    )

    Changing Representation:

    >>> frame = cx.frames.Alice()
    >>> data = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> coord = cx.Coordinate(data, frame)

    >>> coord.vconvert(cx.SphericalPos)
    Coordinate(
        KinematicSpace({ 'length': SphericalPos( ... ) }),
        frame=Alice()
    )

    Showing Frame Transformation:

    >>> space = cx.KinematicSpace(
    ...     length=cx.vecs.CartesianPos3D.from_([1.0, 0, 0], "pc"),
    ...     speed=cx.CartesianVel3D.from_([1.0, 0, 0], "km/s"))

    >>> w=cx.Coordinate(
    ...     data=space,
    ...     frame=cx.frames.TransformedReferenceFrame(
    ...         cx.frames.FriendOfAlice(),
    ...         cx.ops.GalileanSpatialTranslation.from_([20, 0, 0], "kpc"),
    ...     ),
    ... )

    >>> w.to_frame(cx.frames.Alice())
    Coordinate(
        KinematicSpace({
            'length': CartesianPos3D(...), 'speed': CartesianVel3D(...) }),
        frame=Alice()
    )

    >>> w.to_frame(cx.frames.Alice()).data["length"]
    CartesianPos3D(
      x=Quantity(-3.2407793e-16, unit='pc'),
      y=Quantity(-20000.998, unit='pc'),
      z=Quantity(0., unit='pc')
    )

    """

    #: The data of the coordinate. This is a `coordinax.Space` object, which is
    #: a collection of vectors. This can be constructed from a space object, or
    #: any input that can construct a `coordinax.Space` via
    #: `coordinax.Space.from_`.
    data: KinematicSpace = eqx.field(
        converter=Unless(KinematicSpace, KinematicSpace.from_)
    )

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

        >>> w = cx.Coordinate(cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc"),
        ...                   cx.frames.Alice())
        >>> try: w._dimensionality()
        ... except NotImplementedError as e: print("not implemented")
        not implemented

        """
        # TODO: KinematicSpace is currently not implemented.
        return self.data._dimensionality()

    @dispatch
    def __getitem__(self: "Coordinate", index: Any) -> "Coordinate":
        """Return Coordinate, with indexing applied to the data.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.vecs.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
        >>> w = cx.Coordinate.from_(data, cx.frames.Alice())

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

        >>> data = cx.vecs.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
        >>> w = cx.Coordinate.from_(data, cx.frames.Alice())

        >>> print(w["length"])
        <CartesianPos3D: (x, y, z) [kpc]
            [[1 2 3]
             [4 5 6]]>

        """
        return self.data[index]
