"""Representation of coordinates in different systems."""

__all__ = ["Space"]

import math
from collections.abc import Callable, Iterable, Mapping
from textwrap import indent
from types import MappingProxyType
from typing import Any, final
from typing_extensions import override

import equinox as eqx
import jax
from astropy.units import PhysicalType as Dimension
from jax import Device
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

from .utils import DimensionLike, _can_broadcast_shapes, _get_dimension_name
from coordinax._src.typing import Unit
from coordinax._src.utils import classproperty
from coordinax._src.vectors.api import vector
from coordinax._src.vectors.base import AbstractVector


# TODO: figure out how to make the keys into Dimension objects, not str. This is
#       running afoul of Jax's tree flattening, where ImmutableMap and
#       eqx.Module differ.
@final
class Space(AbstractVector, ImmutableMap[Dimension, AbstractVector]):  # type: ignore[misc]
    """A collection of vectors that acts like the primary vector.

    Parameters
    ----------
    *args : Any
        See input to `dict` for the input data.

    **kwargs : AbstractVector
        See input to `dict` for the input data.

    Examples
    --------
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> v = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "km/s^2")

    All the vectors can be brought together into a space:

    >>> space = cx.Space(length=x, speed=v, acceleration=a)
    >>> space
    Space({
        'length': CartesianPos3D( ... ),
        'speed': CartesianVel3D( ... ),
        'acceleration': CartesianAcc3D( ... )
    })

    The vectors can then be accessed by key:

    >>> space["length"]
    CartesianPos3D( ... )

    The space can also be converted to different representations. If the
    conversion is on the primary vector, the other vectors will be
    correspondingly converted.

    >>> space.vconvert(cx.SphericalPos)
    Space({
        'length': SphericalPos( ... ),
        'speed': SphericalVel( ... ),
        'acceleration': SphericalAcc( ... )
    })

    >>> cx.vconvert(cx.SphericalPos, space)
    Space({
        'length': SphericalPos( ... ),
        'speed': SphericalVel( ... ),
        'acceleration': SphericalAcc( ... )
    })

    Actions on the space are done on the contained vectors.

    >>> w = cx.Space(
    ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
    ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
    ... )

    >>> w.ndim
    2

    >>> w.shape
    (1, 2)

    >>> w.shapes
    mappingproxy({'length': (1, 2), 'speed': (1, 2)})

    >>> w.mT.shapes
    mappingproxy({'length': (2, 1), 'speed': (2, 1)})

    There are convenience ways to initialize the vectors in the space:

    >>> space = cx.Space.from_({"length": u.Quantity([1, 2, 3], "km"),
    ...                         "speed": u.Quantity([4, 5, 6], "km/s")})
    >>> print(space)
    Space({
       'length': <CartesianPos3D (x[km], y[km], z[km])
           [1 2 3]>,
       'speed': <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
           [4 5 6]>
    })

    """

    _data: dict[str, AbstractVector] = eqx.field(init=False)

    def __init__(  # pylint: disable=super-init-not-called  # TODO: resolve this
        self,
        /,
        *args: Mapping[DimensionLike, Any]
        | tuple[DimensionLike, Any]
        | Iterable[tuple[DimensionLike, Any]],
        **kwargs: Any,
    ) -> None:
        # Consolidate the inputs into a single dict, then process keys & values.
        raw = dict(*args, **kwargs)  # process the input data

        # Process the keys
        dims = tuple(u.dimension(k) for k in raw)
        keys = tuple(_get_dimension_name(dim) for dim in dims)
        # Convert the values to vectors
        values = tuple(vector(v) for v in raw.values())

        # TODO: check the dimension makes sense for the value

        # Check that the shapes are broadcastable
        values = eqx.error_if(
            values,
            not _can_broadcast_shapes(*map(jnp.shape, values)),
            "vector shapes are not broadcastable.",
        )

        ImmutableMap.__init__(self, dict(zip(keys, values, strict=True)))

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the Space.

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Space.from_(cx.CartesianPos3D.from_([1, 2, 3], "kpc"))
        >>> try: w._dimensionality()
        ... except NotImplementedError as e: print("not implemented")
        not implemented

        """
        raise NotImplementedError  # TODO: implement this

    # ===============================================================
    # Mapping API

    @dispatch
    def __getitem__(self, key: Any) -> Any:
        """Get the vector by key.

        Examples
        --------
        >>> import coordinax as cx
        >>> w = cx.Space(length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...              speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s"))

        By number:

        >>> w[0]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=i32[2], unit=Unit("m")),
                y=Quantity[...](value=i32[2], unit=Unit("m")),
                z=Quantity[...](value=i32[2], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=i32[2], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=i32[2], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=i32[2], unit=Unit("m / s") ) )
        })

        By slice:

        >>> w[1:]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=i32[0,2], unit=Unit("m")),
                y=Quantity[...](value=i32[0,2], unit=Unit("m")),
                z=Quantity[...](value=i32[0,2], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=i32[0,2], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=i32[0,2], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=i32[0,2], unit=Unit("m / s") ) )
        })

        By Ellipsis:

        >>> w[...]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=i32[1,2], unit=Unit("m")),
                y=Quantity[...](value=i32[1,2], unit=Unit("m")),
                z=Quantity[...](value=i32[1,2], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ) )
        })

        By tuple[int, ...]:

        >>> w[(0, 1)]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=i32[], unit=Unit("m")),
                y=Quantity[...](value=i32[], unit=Unit("m")),
                z=Quantity[...](value=i32[], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=i32[], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=i32[], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=i32[], unit=Unit("m / s") ) )
        })

        This also supports numpy index arrays. But this example section
        highlights core python indexing.

        """
        return Space(**{k: v[key] for k, v in self.items()})

    @dispatch
    def __getitem__(self, key: str | Dimension) -> Any:
        """Get the vector by key.

        Examples
        --------
        >>> import coordinax as cx
        >>> w = cx.Space(length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...              speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s"))

        By string key (which is the dimension name):

        >>> print(w["length"])
        <CartesianPos3D (x[m], y[m], z[m])
            [[[1 2 3]
            [4 5 6]]]>

        By the actual dimension object:

        >>> import unxt as u
        >>> print(w[u.dimension("length")])
        <CartesianPos3D (x[m], y[m], z[m])
            [[[1 2 3]
            [4 5 6]]]>

        """
        if isinstance(key, Dimension):
            key = _get_dimension_name(key)

        return ImmutableMap.__getitem__(self, key)

    # ===============================================================
    # Quax API

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        raise NotImplementedError  # TODO: implement this

    # ===============================================================
    # Array API

    # ---------------------------------------------------------------
    # Attributes

    @property
    def size(self) -> int:
        """Total number of elements in the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s") )

        >>> w.size
        2

        """
        return math.prod(self.shape)

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose each vector in the space.

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> w.T.shapes
        mappingproxy({'length': (2, 1), 'speed': (2, 1)})

        """
        return super().T

    # ---------------------------------------------------------------
    # Methods

    def __repr__(self) -> str:
        """Return the string representation.

        Examples
        --------
        >>> import coordinax as cx

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> p = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
        >>> w = cx.Space(length=q, speed=p)
        >>> w
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=i32[], unit=Unit("m")),
                y=Quantity[...](value=i32[], unit=Unit("m")),
                z=Quantity[...](value=i32[], unit=Unit("m"))
            ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=i32[], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=i32[], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=i32[], unit=Unit("m / s") )
            ) })

        """
        cls_name = self.__class__.__name__
        data = "{\n" + indent(repr(self._data)[1:-1], "    ") + "\n}"
        return cls_name + "(" + data + ")"

    def __str__(self) -> str:
        """Return the string representation.

        Examples
        --------
        >>> import coordinax as cx

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> p = cx.CartesianVel3D.from_([4, 5, 6], "m/s")
        >>> w = cx.Space(length=q, speed=p)
        >>> print(w)
        Space({
            'length': <CartesianPos3D (x[m], y[m], z[m])
                [1 2 3]>,
            'speed': <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
                [4 5 6]>
        })

        """
        cls_name = self.__class__.__name__
        kv = (f"{k!r}: {v!s}" for k, v in self._data.items())
        data = "{\n" + indent(",\n".join(kv), "   ") + "\n}"
        return cls_name + "(" + data + ")"

    # ===============================================================
    # Collection

    def asdict(
        self,
        *,
        dict_factory: Callable[[Any], Mapping[str, u.Quantity]] = dict,
    ) -> Mapping[str, u.Quantity]:
        """Return the vector as a Mapping.

        Parameters
        ----------
        dict_factory : type[Mapping]
            The type of the mapping to return.

        Returns
        -------
        Mapping[str, Quantity]
            The vector as a mapping.

        See Also
        --------
        `dataclasses.asdict`
            This applies recursively to the components of the vector.

        """
        return dict_factory(self._data)

    @override
    @classproperty
    @classmethod
    def components(cls) -> tuple[str, ...]:  # type: ignore[override]
        """Vector component names."""
        raise NotImplementedError  # TODO: implement this

    @property
    def units(self) -> MappingProxyType[str, Unit]:
        """Get the units of the vector's components."""
        raise NotImplementedError  # TODO: implement this

    @override
    @property
    def dtypes(self) -> MappingProxyType[str, MappingProxyType[str, jnp.dtype[Any]]]:  # type: ignore[override]
        """Get the dtypes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> w.dtypes
        mappingproxy({'length': mappingproxy({'x': dtype('int32'), 'y': dtype('int32'), 'z': dtype('int32')}),
                      'speed': mappingproxy({'d_x': dtype('int32'), 'd_y': dtype('int32'), 'd_z': dtype('int32')})})

        """  # noqa: E501
        return MappingProxyType({k: v.dtypes for k, v in self.items()})

    @override
    @property
    def devices(self) -> MappingProxyType[str, MappingProxyType[str, Device]]:  # type: ignore[override]
        """Get the devices of the vector's components.

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> w.devices
        mappingproxy({'length': mappingproxy({'x': CpuDevice(id=0), 'y': CpuDevice(id=0), 'z': CpuDevice(id=0)}),
                      'speed': mappingproxy({'d_x': CpuDevice(id=0), 'd_y': CpuDevice(id=0), 'd_z': CpuDevice(id=0)})})

        """  # noqa: E501
        return MappingProxyType({k: v.devices for k, v in self.items()})

    @override
    @property
    def shapes(self) -> MappingProxyType[str, MappingProxyType[str, tuple[int, ...]]]:  # type: ignore[override]
        """Get the shapes of the spaces's fields.

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> w.shapes
        mappingproxy({'length': (1, 2), 'speed': (1, 2)})

        """
        return MappingProxyType({k: v.shape for k, v in self.items()})

    @property
    def sizes(self) -> MappingProxyType[str, int]:
        """Get the sizes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> w.sizes
        mappingproxy({'length': 6, 'speed': 6})

        """
        return MappingProxyType({k: v.size for k, v in self.items()})
