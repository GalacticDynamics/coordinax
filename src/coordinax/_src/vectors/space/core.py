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

from .utils import DimensionLike, _get_dimension_name, can_broadcast_shapes
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
    >>> print(space)
    Space({
       'length': <CartesianPos3D (x[km], y[km], z[km])
           [1 2 3]>,
       'speed': <CartesianVel3D (x[km / s], y[km / s], z[km / s])
           [4 5 6]>,
       'acceleration': <CartesianAcc3D (x[km / s2], y[km / s2], z[km / s2])
           [7 8 9]>
    })

    The vectors can initialized from `unxt.Quantity` objects and can have
    (brodcastable) batch shapes:

    >>> w = cx.Space(
    ...     length=u.Quantity([[8.5, 0, 0], [10, 0, 0]], "kpc"),
    ...     speed=u.Quantity([0, 200, 0], "km/s"))
    >>> print(w)
    Space({
       'length': <CartesianPos3D (x[kpc], y[kpc], z[kpc])
           [[ 8.5  0.   0. ]
            [10.   0.   0. ]]>,
       'speed': <CartesianVel3D (x[km / s], y[km / s], z[km / s])
           [  0 200   0]>
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
       'speed': <CartesianVel3D (x[km / s], y[km / s], z[km / s])
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
            not can_broadcast_shapes([v.shape for v in values]),
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
                x=Quantity[...]( value=i32[2], unit=Unit("m / s") ),
                y=Quantity[...]( value=i32[2], unit=Unit("m / s") ),
                z=Quantity[...]( value=i32[2], unit=Unit("m / s") ) )
        })

        By slice:

        >>> w[1:]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=i32[0,2], unit=Unit("m")),
                y=Quantity[...](value=i32[0,2], unit=Unit("m")),
                z=Quantity[...](value=i32[0,2], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                x=Quantity[...]( value=i32[0,2], unit=Unit("m / s") ),
                y=Quantity[...]( value=i32[0,2], unit=Unit("m / s") ),
                z=Quantity[...]( value=i32[0,2], unit=Unit("m / s") ) )
        })

        By Ellipsis:

        >>> w[...]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=i32[1,2], unit=Unit("m")),
                y=Quantity[...](value=i32[1,2], unit=Unit("m")),
                z=Quantity[...](value=i32[1,2], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                x=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ),
                y=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ),
                z=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ) )
        })

        By tuple[int, ...]:

        >>> w[(0, 1)]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=i32[], unit=Unit("m")),
                y=Quantity[...](value=i32[], unit=Unit("m")),
                z=Quantity[...](value=i32[], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                x=Quantity[...]( value=i32[], unit=Unit("m / s") ),
                y=Quantity[...]( value=i32[], unit=Unit("m / s") ),
                z=Quantity[...]( value=i32[], unit=Unit("m / s") ) )
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

    # TODO: unify this with AvalMixin
    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array.

        Examples
        --------
        >>> import coordinax as cx
        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([7, 8, 9], "m/s")
        ... )
        >>> w.aval()
        ShapedArray(int32[1,2,6])

        """
        avals = tuple(v.aval() for v in self.values())
        shapes = [a.shape for a in avals]
        shape = (
            *jnp.broadcast_shapes(*[s[:-1] for s in shapes]),
            sum(s[-1] for s in shapes),
        )
        dtype = jnp.result_type(*map(jnp.dtype, avals))
        return jax.core.ShapedArray(shape, dtype)  # type: ignore[no-untyped-call]

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
                x=Quantity[...]( value=i32[], unit=Unit("m / s") ),
                y=Quantity[...]( value=i32[], unit=Unit("m / s") ),
                z=Quantity[...]( value=i32[], unit=Unit("m / s") )
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
            'speed': <CartesianVel3D (x[m / s], y[m / s], z[m / s])
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
                      'speed': mappingproxy({'x': dtype('int32'), 'y': dtype('int32'), 'z': dtype('int32')})})

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
                      'speed': mappingproxy({'x': CpuDevice(id=0), 'y': CpuDevice(id=0), 'z': CpuDevice(id=0)})})

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
