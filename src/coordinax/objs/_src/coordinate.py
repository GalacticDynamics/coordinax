"""Coordinates."""

__all__ = ("AbstractCoordinate", "Coordinate")


from collections.abc import Mapping
from typing import Any, ClassVar, Literal, assert_never, final

import equinox as eqx
import jax
import plum
import wadler_lindig as wl  # type: ignore[import-untyped]
from quax import register

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items, replace
from dataclassish.converters import Unless

import coordinax.api as cxapi
import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.ops as cxop
from .base import AbstractVectorLike
from .bundle import PointedVector
from .vector import Vector


# TODO: parametrize by the vector type(s), when Space is parametrized,
# and the frame type(s)
class AbstractCoordinate(AbstractVectorLike):
    """Coordinates are vectors in a reference frame.

    See Also
    --------
    `coordinax.Coordinate` for a concrete implementation.

    """

    #: The data of the coordinate. This is a `coordinax.PointedVector` object,
    #: which is a collection of vectors.
    data: eqx.AbstractVar[PointedVector]  # TODO: PointedVector[PosT] -- plum#212

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
        >>> import coordinax.frames as cxf

        >>> cicrs = cx.Coordinate(cx.Vector.from_([1, 2, 3], "kpc"),
        ...                       cxf.ICRS())

        >>> cicrs.to_frame(cxf.ICRS()) is cicrs
        True

        >>> cgcf = cicrs.to_frame(cxf.Galactocentric())
        >>> cgcf
        Coordinate(
            PointedVector( base=Vector( ... ) ),
            frame=Galactocentric( ... )
        )

        """
        op = self.frame.transform_op(toframe)

        # Special case for identity operations
        if isinstance(op, cxop.Identity):
            return self

        # Otherwise, apply the transformation and return a new coordinate
        tau = u.Q(jnp.array(0.0), "s") if t is None else t
        new_data = op(tau, self.data)
        out: AbstractCoordinate = self.__class__(new_data, toframe)
        return out

    def vconvert(
        self,
        to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
        /,
        *,
        field_charts: Mapping[str, cxc.AbstractChart] | None = None,  # type: ignore[type-arg]
    ) -> "AbstractCoordinate":
        r"""Convert the coordinate's data to a new chart.

        This method converts the underlying `PointedVector` data to a new chart
        while preserving the reference frame. The conversion follows the same
        semantics as `PointedVector.vconvert`:

        1. The base point converts as a `Point` (position map)
        2. Fibre vectors convert according to their roles using tangent
           transformations with the base as the reference point

        Parameters
        ----------
        to_chart : AbstractChart
            Target chart instance for the base and (by default) all fields.
        field_charts : Mapping[str, AbstractChart], optional
            Override target chart for specific fields. Keys are field names,
            values are target charts for those fields. If not provided, all
            fields use ``to_chart``.

        Returns
        -------
        AbstractCoordinate
            New coordinate with data in the target chart(s), same frame.

        Examples
        --------
        >>> import coordinax as cx

        >>> coord = cx.Coordinate(
        ...     cx.Vector.from_([1, 2, 3], "kpc"),
        ...     cxf.ICRS()
        ... )
        >>> coord.data.base.chart
        Cart3D()

        Convert to spherical coordinates:

        >>> sph_coord = coord.vconvert(cxc.sph3d)
        >>> sph_coord.data.base.chart
        Spherical3D()
        >>> sph_coord.frame
        ICRS()

        With velocity data:

        >>> space = cx.PointedVector(
        ...     base=cx.Vector.from_([1, 0, 0], "kpc"),
        ...     velocity=cx.Vector.from_([10, 20, 30], "km/s"),
        ... )
        >>> coord = cx.Coordinate(space, cxf.ICRS())

        >>> sph_coord = coord.vconvert(cxc.sph3d)
        >>> sph_coord.data.base.chart
        Spherical3D()
        >>> sph_coord.data["velocity"].chart
        Spherical3D()

        Specify different charts for fields:

        >>> mixed = coord.vconvert(
        ...     cxc.sph3d,
        ...     field_charts={"velocity": cxc.cyl3d}
        ... )
        >>> mixed.data.base.chart
        Spherical3D()
        >>> mixed.data["velocity"].chart
        Cylindrical3D()

        """
        new_data = self.data.vconvert(to_chart, field_charts=field_charts)
        return self.__class__(new_data, self.frame)

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
            If `vector`, only include the data itself as an `PointedVector`.
            If `map`, only include the data as dict-like representation.
        **kwargs : Any, optional
            Additional keyword arguments to pass to `wl.pdoc`.
            'vector_form' is one option.

        Examples
        --------
        >>> import coordinax as cx
        >>> import wadler_lindig as wl

        >>> frame = cxf.ICRS()
        >>> data = cx.Vector.from_([1, 2, 3], "kpc")
        >>> coord = cx.Coordinate(data, frame)

        >>> wl.pprint(coord, include_data_name="named")
        Coordinate(
            data=PointedVector( base=Vector( ... ) ),
            frame=ICRS()
        )

        >>> wl.pprint(coord, include_data_name="vector")
        Coordinate(
            PointedVector( base=Vector( ... ) ),
            frame=ICRS()
        )

        >>> wl.pprint(coord, include_data_name="map")
        Coordinate({}, frame=ICRS())

        >>> print(repr(coord))
        Coordinate(
            PointedVector( base=Vector( ... ) ),
            frame=ICRS()
        )

        >>> print(str(coord))
        Coordinate({}, frame=ICRS())

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
        ...                       cxf.ICRS())
        >>> print(repr(coord))
        Coordinate(
            PointedVector( base=Vector( ... ) ),
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
        ...                       cxf.ICRS())
        >>> print(coord)
        Coordinate({}, frame=ICRS())

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
    ...                       cxf.ICRS())
    >>> cxf.frame_of(coord)
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
    ...                       cxf.ICRS())
    >>> coord
    Coordinate( PointedVector( base=Vector( ... ) ), frame=ICRS() )

    Alternative Construction:

    >>> frame = cxf.ICRS()
    >>> data = cx.Vector.from_([1, 2, 3], "kpc")
    >>> cx.Coordinate(data, frame)
    Coordinate( PointedVector( base=Vector( ... ) ), frame=ICRS() )

    Changing Representation:

    >>> frame = cxf.ICRS()
    >>> data = cx.Vector.from_([1, 2, 3], "kpc")
    >>> coord = cx.Coordinate(data, frame)

    >>> cx.vconvert(cxc.sph3d, coord)
    Coordinate( PointedVector( base=Vector( ... ) ), frame=ICRS() )

    Showing Frame Transformation:

    >>> space = cx.PointedVector(
    ...     base=cx.Vector.from_([1.0, 0, 0], "pc"),
    ...     speed=cx.Vector.from_([1.0, 0, 0], "km/s"))

    >>> w=cx.Coordinate(
    ...     data=space,
    ...     frame=cxf.TransformedReferenceFrame(
    ...         cxf.ICRS(),
    ...         cxop.Translate.from_([20, 0, 0], "pc"),
    ...     ),
    ... )

    >>> w.to_frame(cxf.ICRS())
    Coordinate(..., frame=ICRS()...)

    >>> w.to_frame(cxf.ICRS()).data.base
    Vector(
      data={'x': Q(21., 'pc'), 'y': Q(0., 'pc'), 'z': Q(0., 'pc')},
      chart=Cart3D...,
      role=Point()
    )

    """

    # The data of the coordinate. This is a `coordinax.PointedVector` object,
    # which is a collection of vectors. This can be constructed from a space
    # object, or any input that can construct a `coordinax.PointedVector` via
    # `coordinax.PointedVector.from_`.
    data: PointedVector = eqx.field(
        converter=Unless(PointedVector, PointedVector.from_)
    )

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

        >>> space = cx.PointedVector(
        ...     base=cx.Vector.from_([[1, 2, 3], [4, 5, 6]], "kpc"),
        ...     speed=cx.Vector.from_([[1, 0, 0], [0, 1, 0]], "km/s"),
        ... )
        >>> w = cx.Coordinate(space, cxf.ICRS())

        >>> print(w[0].data.base)
        <Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
            [1 2 3]>

        """
        return replace(self, data=self.data[index])

    @plum.dispatch
    def __getitem__(self: "Coordinate", index: str) -> Vector:
        """Return the data of the coordinate.

        Examples
        --------
        >>> import coordinax as cx

        >>> space = cx.PointedVector(
        ...     base=cx.Vector.from_([[1, 2, 3], [4, 5, 6]], "kpc"),
        ...     speed=cx.Vector.from_([[1, 0, 0], [0, 1, 0]], "km/s"),
        ... )
        >>> w = cx.Coordinate(space, cxf.ICRS())

        >>> print(w["speed"])
        <Vector: chart=Cart3D, role=PhysVel (x, y, z) [km / s]
            [[1 0 0]
             [0 1 0]]>

        """
        return self.data[index]


@plum.dispatch
def vconvert(target: cxc.AbstractChart, w: Coordinate, /) -> Coordinate:  # type: ignore[type-arg]
    """Transform the vector representation of a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> frame = cxf.NoFrame()
    >>> data = cx.Vector.from_([1, 2, 3], "kpc")
    >>> w = cx.Coordinate(data, frame)

    >>> cx.vconvert(cxc.sph3d, w)
    Coordinate(
      PointedVector(
        base=Vector(... chart=Spherical3D...)
      ),
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
    >>> coord = cx.Coordinate(data, cxf.ICRS())

    >>> print(-coord)
    Coordinate({}, frame=ICRS())

    >>> print((-coord).data.base)
    <Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
        [-1 -2 -3]>

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
def apply_op(
    op: cxop.AbstractOperator,
    tau: Any,
    obj: Coordinate,
    /,
    **kwargs: Any,
) -> Coordinate:
    """Apply the operator to a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.Vector.from_([1, 2, 3], "kpc"),
    ...                       cxf.ICRS())
    >>> op = cxop.Translate.from_([-1, -1, -1], "kpc")

    >>> new_coord = op(coord)
    >>> print(new_coord.data.base)
    <Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
        [0 1 2]>

    """
    return replace(obj, data=cxapi.apply_op(op, tau, obj.data, **kwargs))
