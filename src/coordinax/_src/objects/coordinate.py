"""Coordinates."""

__all__ = ("AbstractCoordinate", "Coordinate")


from typing import Any, ClassVar, Literal, assert_never, cast, final

import equinox as eqx
import jax
import plum
import wadler_lindig as wl
from quax import register

import unxt as u
from dataclassish import field_items, replace
from dataclassish.converters import Unless

import coordinax._src.charts as cxc
import coordinax._src.frames as cxf
import coordinax._src.operators as cxo
from .base import AbstractVectorLike
from .bundle import FiberPoint
from .vector import Vector


# TODO: parametrize by the vector type(s), when Space is parametrized,
# and the frame type(s)
class AbstractCoordinate(AbstractVectorLike):
    """Coordinates are vectors in a reference frame.

    See Also
    --------
    `coordinax.Coordinate` for a concrete implementation.

    """

    #: The data of the coordinate. This is a `coordinax.FiberPoint` object,
    #: which is a collection of vectors.
    data: eqx.AbstractVar[FiberPoint]  # TODO: FiberPoint[PosT] -- plum#212

    #: The reference frame of the coordinate as a
    #: `coordinax.frames.AbstractReferenceFrame` object.
    frame: eqx.AbstractVar[cxf.AbstractReferenceFrame]

    # ===============================================================
    # Coordinate API

    def to_frame(
        self, toframe: cxf.AbstractReferenceFrame, /, t: u.Q | None = None
    ) -> "AbstractCoordinate":
        """Transform the coordinate to a specified frame.

        Examples
        --------
        >>> import coordinax as cx

        >>> cicrs = cx.Coordinate(cx.Vector.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.ICRS())

        >>> cicrs.to_frame(cx.frames.ICRS()) is cicrs
        True

        >>> cgcf = cicrs.to_frame(cx.frames.Galactocentric())
        >>> cgcf
        Coordinate(
            FiberPoint({ 'base': Cart3D(...) }),
            frame=Galactocentric( ... )
        )

        """
        op = self.frame.transform_op(toframe)

        # Special case for identity operations
        if isinstance(op, Identity):
            return self

        # Otherwise, apply the transformation and return a new coordinate
        new_data = op(self.data) if t is None else op(t, self.data)[1]
        out = self.__class__.from_(new_data, toframe)
        return cast("AbstractCoordinate", out)

    # ===============================================================
    # Quax API

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        return self.data.aval()

    # ===============================================================
    # Plum API

    __faithful__: ClassVar[bool] = True

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
        include_data_name : {'named', 'vector', 'map'}, optional
            If `named`, include the name of the data field.
            If `vector`, only include the data itself as an `FiberPoint`.
            If `map`, only include the data as dict-like representation.
        **kwargs : Any, optional
            Additional keyword arguments to pass to `wl.pdoc`.
            'vector_form' is one option.

        Examples
        --------
        >>> import coordinax as cx
        >>> import wadler_lindig as wl

        >>> frame = cx.frames.ICRS()
        >>> data = cx.Vector.from_([1, 2, 3], "kpc")
        >>> coord = cx.Coordinate(data, frame)

        >>> wl.pprint(coord, include_data_name="named")
        Coordinate(
            data=FiberPoint({ 'base': Cart3D(...) }),
            frame=ICRS()
        )

        >>> wl.pprint(coord, include_data_name="vector")
        Coordinate(
            FiberPoint({ 'base': Cart3D(...) }),
            frame=ICRS()
        )

        >>> wl.pprint(coord, include_data_name="map")
        Coordinate( {'base': Cart3D(...)}, frame=ICRS() )

        >>> print(repr(coord))
        Coordinate(
            FiberPoint({ 'base': Cart3D(...) }),
            frame=ICRS()
        )

        >>> print(str(coord))
        Coordinate(
            {
                'base':
                <Cart3D: (x, y, z) [kpc]
                    [1 2 3]>
            },
            frame=ICRS()
        )

        """
        # Prefer to use short names (e.g. Quantity -> Q) and compact unit forms
        kwargs.setdefault("use_short_name", True)
        kwargs.setdefault("named_unit", False)

        # Different ways to show the data
        match include_data_name:
            case "named":
                docs = wl.named_objs(tuple(field_items(self)), **kwargs)
            case "vector":
                docs = [
                    wl.pdoc(self.data, **kwargs),
                    *wl.named_objs(tuple(field_items(self))[1:], **kwargs),
                ]
            case "map":
                docs = [
                    wl.pdoc(self.data._data, **kwargs),
                    *wl.named_objs(tuple(field_items(self))[1:], **kwargs),
                ]
            case _:
                assert_never()

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
        >>> coord = cx.Coordinate(cx.Vector.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.ICRS())
        >>> print(repr(coord))
        Coordinate(
            FiberPoint({ 'length': Cart3D(...) }),
            frame=ICRS()
        )

        """
        return wl.pformat(self, width=88, include_data_name="vector", vector_form=False)

    def __str__(self) -> str:
        """Return string representation.

        Examples
        --------
        >>> import coordinax as cx
        >>> coord = cx.Coordinate(cx.Vector.from_([1, 2, 3], "kpc"),
        ...                       cx.frames.ICRS())
        >>> print(coord)
        Coordinate(
            {
            'length': <Cart3D: (x, y, z) [kpc]
                [1 2 3]>
            },
            frame=ICRS()
        )

        """
        return wl.pformat(self, width=88, include_data_name="map", vector_form=True)

    # ===============================================================
    # IPython API

    _repr_latex_ = lambda self: wl.pformat(self)  # noqa: E731  # TODO: implement this


@plum.dispatch
def frame_of(obj: AbstractCoordinate) -> cxf.AbstractReferenceFrame:
    """Return the frame of the coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.Vector.from_([1, 2, 3], "kpc"),
    ...                       cx.frames.ICRS())
    >>> cx.frames.frame_of(coord)
    ICRS()

    """
    return obj.frame


##############################################################################


@final
class Coordinate(AbstractCoordinate):
    """Coordinates are vectors in a reference frame.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.Vector.from_([1, 2, 3], "kpc"),
    ...                       cx.frames.ICRS())
    >>> coord
    Coordinate( FiberPoint({ 'base': Cart3D(...) }),
                frame=ICRS() )

    Alternative Construction:

    >>> frame = cx.frames.ICRS()
    >>> data = cx.Vector.from_([1, 2, 3], "kpc")
    >>> cx.Coordinate.from_({"data": data, "frame": frame})
    Coordinate( FiberPoint({ 'base': Cart3D(...) }),
                frame=ICRS() )

    Changing Representation:

    >>> frame = cx.frames.ICRS()
    >>> data = cx.Vector.from_([1, 2, 3], "kpc")
    >>> coord = cx.Coordinate(data, frame)

    >>> coord.vconvert(cx.charts.sph3d)
    Coordinate( FiberPoint({ 'base': Spherical3D( ... ) }),
                frame=ICRS() )

    Showing Frame Transformation:

    >>> space = cx.FiberPoint(
    ...     base=cx.Vector.from_([1.0, 0, 0], "pc"),
    ...     speed=cx.Vector.from_([1.0, 0, 0], "km/s", cx.charts.CartVel3D))

    >>> w=cx.Coordinate(
    ...     data=space,
    ...     frame=cx.frames.TransformedReferenceFrame(
    ...         cx.frames.Galactocentric(),
    ...         cx.ops.GalileanOp.from_([20, 0, 0], "kpc"),
    ...     ),
    ... )

    >>> w.to_frame(cx.frames.ICRS())
    Coordinate(
        FiberPoint({
            'base': Cart3D(...), 'speed': CartVel3D(...) }),
        frame=ICRS()
    )

    >>> w.to_frame(cx.frames.ICRS()).data["base"]
    Cart3D(x=Q(-1587.6683, 'pc'), y=Q(-24573.762, 'pc'), z=Q(-13583.504, 'pc'))

    """

    # The data of the coordinate. This is a `coordinax.FiberPoint` object,
    # which is a collection of vectors. This can be constructed from a space
    # object, or any input that can construct a `coordinax.FiberPoint` via
    # `coordinax.FiberPoint.from_`.
    data: FiberPoint = eqx.field(converter=Unless(FiberPoint, FiberPoint.from_))

    #: The reference frame of the coordinate as a :
    # `coordinax.frames.AbstractReferenceFrame` object. This can be : from a
    # reference frame object, or any input that can construct a :
    # `coordinax.frames.TransformedReferenceFrame` via :
    # `coordinax.frames.AbstractReferenceFrame.from_`.
    frame: cxf.AbstractReferenceFrame = eqx.field(
        converter=Unless(
            cxf.AbstractReferenceFrame, cxf.TransformedReferenceFrame.from_
        )
    )

    # ===============================================================
    # Vector API

    @plum.dispatch
    def __getitem__(self: "Coordinate", index: Any) -> "Coordinate":
        """Return Coordinate, with indexing applied to the data.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.Vector.from_([[1, 2, 3], [4, 5, 6]], "kpc")
        >>> w = cx.Coordinate.from_(data, cx.frames.ICRS())

        >>> print(w[0].data["length"])
        <Cart3D: (x, y, z) [kpc]
            [1 2 3]>

        """
        return replace(self, data=self.data[index])

    @plum.dispatch
    def __getitem__(self: "Coordinate", index: str) -> Vector:
        """Return the data of the coordinate.

        Examples
        --------
        >>> import coordinax as cx

        >>> data = cx.Vector.from_([[1, 2, 3], [4, 5, 6]], "kpc")
        >>> w = cx.Coordinate.from_(data, cx.frames.ICRS())

        >>> print(w["length"])
        <Cart3D: (x, y, z) [kpc]
            [[1 2 3]
             [4 5 6]]>

        """
        return self.data[index]


@plum.dispatch
def vconvert(target: cxc.AbstractChart, w: Coordinate, /) -> Coordinate:  # type: ignore[type-arg]
    """Transform the vector representation of a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> frame = cx.frames.NoFrame()
    >>> data = cx.Vector.from_([1, 2, 3], "kpc")
    >>> w = cx.Coordinate(data, frame)

    >>> cx.vconvert(cx.charts.sph3d, w)
    Coordinate(
        FiberPoint({ 'base': Spherical3D( ... ) }),
        frame=NoFrame()
    )

    """
    return replace(w, data=w.data.vconvert(target))


@register(jax.lax.neg_p)
def neg_p_coord(x: Coordinate, /) -> Coordinate:
    """Negate a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> data = cx.Vector.from_([1, 2, 3], "kpc")
    >>> coord = cx.Coordinate(data, cx.frames.ICRS())

    >>> print(-coord)
    Coordinate(
        {
           'length': <Cart3D: (x, y, z) [kpc]
               [-1 -2 -3]>
        },
        frame=ICRS()
    )

    """
    return replace(x, data=-x.data)


@register(jax.lax.add_p)
def add_p_coord_pos(x: Coordinate, y: Vector, /) -> Coordinate:
    r"""Add a position vector to a coordinate.

    We assume that the position vector is in the same frame as the coordinate.

    To understand this operation, let's consider a phase-space point $(x, v) \in
    \mathbb{R}^3\times\mathbb{R}^3$ consisting of a position and a velocity. A
    pure spatial translation is the map $T_{\Delta x} : (x,v) \mapsto (x+\Delta
    x,\ v)$, i.e. only the position is shifted; velocity is unchanged.

    """
    # Get the Cartesian class for the coordinate's position
    cartrep = y.chart.cartesian
    # Convert the coordinate to that class. This changes the position, but also
    # the other components, e.g. the velocity.
    data = x.data.vconvert(cartrep)
    # Now add the position vector to the position component only
    data = replace(data, length=data["length"] + y)
    # Transform back to the original vector types
    # data.vconvert()  # TODO: all original types
    # Reconstruct the Coordinate
    return Coordinate(data, frame=x.frame)


@plum.dispatch
def operate(self: cxo.AbstractOperator, obj: Coordinate, /) -> Coordinate:
    """Apply the operator to a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.Vector.from_([1, 2, 3], "kpc"),
    ...                       cx.frames.ICRS())
    >>> coord
    Coordinate( FiberPoint({ 'base': Cart3D(...) }),
                frame=ICRS() )

    >>> op = cx.ops.GalileanOp.from_([-1, -1, -1], "kpc")

    >>> new_coord = op(coord)
    >>> print(new_coord.data["base"])
    <Cart3D: (x, y, z) [kpc]
        [0 1 2]>

    """
    # TODO: take the frame into account
    return replace(obj, data=self(obj.data))
