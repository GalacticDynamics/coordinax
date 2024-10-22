"""Representation of coordinates in different systems."""

__all__ = ["Space"]

from collections.abc import Callable, ItemsView, Iterable, Mapping
from textwrap import indent
from types import MappingProxyType
from typing import Any, TypeAlias, final
from typing_extensions import override

import astropy.units as u
import equinox as eqx
import jax
from astropy.units import PhysicalType as Dimension
from jax import Device
from plum import dispatch

import quaxed.numpy as jnp
from unxt import Quantity, dimensions
from xmmutablemap import ImmutableMap

from .base import AbstractAcc, AbstractPos, AbstractVector, AbstractVel
from .typing import Unit
from .utils import classproperty
from coordinax._src.funcs import represent_as

DimensionLike: TypeAlias = Dimension | str


def _get_dimension_name(dim: DimensionLike, /) -> str:
    return dimensions(dim)._physical_type_list[0]  # noqa: SLF001


def _can_broadcast_shapes(*shapes: tuple[int, ...]) -> bool:
    """Check if the shapes can be broadcasted together."""
    try:
        jnp.broadcast_shapes(*shapes)
    except ValueError:
        return False
    return True


# TODO: figure out how to make the keys into Dimension objects, not str. This is
#       running afoul of Jax's tree flattening, where ImmutableMap and
#       eqx.Module differ.
@final
class Space(AbstractVector, ImmutableMap[Dimension, AbstractVector]):  # type: ignore[misc]
    """A collection of vectors that acts like the primary vector.

    Parameters
    ----------
    *args : Mapping[DimensionLike, AbstractVector] | tuple[DimensionLike, AbstractVector] | Iterable[tuple[DimensionLike, AbstractVector]]
        See input to `dict` for the input data.

    primary_key : DimensionLike, optional
        The key for the primary vector of the space.
        Default is "length" for position vectors.

    **kwargs : AbstractVector
        See input to `dict` for the input data.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    >>> x = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> v = cx.CartesianVel3D.from_([4, 5, 6], "km/s")
    >>> a = cx.CartesianAcc3D.from_([7, 8, 9], "km/s^2")

    >>> space = cx.Space(length=x, speed=v, acceleration=a)
    >>> space
    Space({
        'length': CartesianPos3D( ... ),
        'speed': CartesianVel3D( ... ),
        'acceleration': CartesianAcc3D( ... )}
    )

    >>> space["length"]
    CartesianPos3D( ... )

    >>> space.represent_as(cx.SphericalPos)
    Space({
        'length': SphericalPos( ... ),
        'speed': SphericalVel( ... ),
        'acceleration': SphericalAcc( ... )}
    )

    >>> cx.represent_as(space, cx.SphericalPos)
    Space({
        'length': SphericalPos( ... ),
        'speed': SphericalVel( ... ),
        'acceleration': SphericalAcc( ... )}
    )

    """  # noqa: E501

    _data: dict[Dimension, AbstractVector] = eqx.field(init=False)

    def __init__(  # pylint: disable=super-init-not-called  # TODO: resolve this
        self,
        /,
        *args: Mapping[DimensionLike, AbstractVector]
        | tuple[DimensionLike, AbstractVector]
        | Iterable[tuple[DimensionLike, AbstractVector]],
        **kwargs: AbstractVector,
    ) -> None:
        # Process the input data
        raw = dict(*args, **kwargs)  # process the input data
        keys = [_get_dimension_name(k) for k in raw]
        keys = eqx.error_if(
            keys,
            len(keys) < len(raw),
            f"Space(**input) contained duplicate keys {set(raw) - set(keys)}.",
        )
        # TODO: check the key dimension makes sense for the value

        # Check that the shapes are broadcastable
        keys = eqx.error_if(
            keys,
            not _can_broadcast_shapes(*(v.shape for v in raw.values())),
            "vector shapes are not broadcastable.",
        )

        ImmutableMap.__init__(self, dict(zip(keys, raw.values(), strict=True)))

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
                x=Quantity[...](value=f32[2], unit=Unit("m")),
                y=Quantity[...](value=f32[2], unit=Unit("m")),
                z=Quantity[...](value=f32[2], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=f32[2], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=f32[2], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=f32[2], unit=Unit("m / s") ) )}
        )

        By slice:

        >>> w[1:]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=f32[0,2], unit=Unit("m")),
                y=Quantity[...](value=f32[0,2], unit=Unit("m")),
                z=Quantity[...](value=f32[0,2], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=f32[0,2], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=f32[0,2], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=f32[0,2], unit=Unit("m / s") ) )}
        )

        By Ellipsis:

        >>> w[...]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=f32[1,2], unit=Unit("m")),
                y=Quantity[...](value=f32[1,2], unit=Unit("m")),
                z=Quantity[...](value=f32[1,2], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=f32[1,2], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=f32[1,2], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=f32[1,2], unit=Unit("m / s") ) )}
        )

        By tuple[int, ...]:

        >>> w[(0, 1)]
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=f32[], unit=Unit("m")),
                y=Quantity[...](value=f32[], unit=Unit("m")),
                z=Quantity[...](value=f32[], unit=Unit("m")) ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=f32[], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=f32[], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=f32[], unit=Unit("m / s") ) )}
        )

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

        By string key:

        >>> w["length"]
        CartesianPos3D(
            x=Quantity[...](value=f32[1,2], unit=Unit("m")),
            y=Quantity[...](value=f32[1,2], unit=Unit("m")),
            z=Quantity[...](value=f32[1,2], unit=Unit("m"))
        )

        By dimension:

        >>> import astropy.units as u
        >>> w[u.get_physical_type("length")]
        CartesianPos3D(
            x=Quantity[...](value=f32[1,2], unit=Unit("m")),
            y=Quantity[...](value=f32[1,2], unit=Unit("m")),
            z=Quantity[...](value=f32[1,2], unit=Unit("m"))
        )

        """
        if isinstance(key, Dimension):
            key = _get_dimension_name(key)

        return ImmutableMap.__getitem__(self, key)

    # ===============================================================
    # Quax

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        raise NotImplementedError  # TODO: implement this

    # ===============================================================
    # Array API

    # ---------------------------------------------------------------
    # Attributes

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose each vector in the space.

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )
        >>> w.mT
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=f32[2,1], unit=Unit("m")),
                y=Quantity[...](value=f32[2,1], unit=Unit("m")),
                z=Quantity[...](value=f32[2,1], unit=Unit("m"))
            ), 'speed': CartesianVel3D(
                d_x=Quantity[...]( value=f32[2,1], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=f32[2,1], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=f32[2,1], unit=Unit("m / s") )
            )} )

        """
        return super().mT

    @property
    def ndim(self) -> int:
        """Number of array dimensions (axes).

        Examples
        --------
        >>> import coordinax as cx

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s") )

        >>> w.ndim
        2

        """
        return super().ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> from unxt import Quantity

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> w.shape
        (1, 2)

        """
        return super().shape

    @property
    def size(self) -> int:
        """Total number of elements in the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> from unxt import Quantity

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s") )

        >>> w.size
        2

        """
        return super().size

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose each vector in the space.

        Examples
        --------
        >>> import coordinax as cx
        >>> from unxt import Quantity

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> w.T
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=f32[2,1], unit=Unit("m")),
                y=Quantity[...](value=f32[2,1], unit=Unit("m")),
                z=Quantity[...](value=f32[2,1], unit=Unit("m"))
            ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=f32[2,1], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=f32[2,1], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=f32[2,1], unit=Unit("m / s") )
            )} )

        """
        return super().T

    # ---------------------------------------------------------------
    # Methods

    def __neg__(self) -> "Self":
        """Negative of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> from unxt import Quantity

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> (-w)["length"].x
        Quantity['length'](Array([[-1., -4.]], dtype=float32), unit='m')

        """
        return type(self)(**{k: -v for k, v in self.items()})

    def __repr__(self) -> str:
        """Return the string representation.

        Examples
        --------
        >>> import coordinax as cx
        >>> from unxt import Quantity

        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> p = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
        >>> w = cx.Space(length=q, speed=p)
        >>> w
        Space({
            'length': CartesianPos3D(
                x=Quantity[...](value=f32[], unit=Unit("m")),
                y=Quantity[...](value=f32[], unit=Unit("m")),
                z=Quantity[...](value=f32[], unit=Unit("m"))
            ),
            'speed': CartesianVel3D(
                d_x=Quantity[...]( value=f32[], unit=Unit("m / s") ),
                d_y=Quantity[...]( value=f32[], unit=Unit("m / s") ),
                d_z=Quantity[...]( value=f32[], unit=Unit("m / s") )
            )} )

        """
        cls_name = self.__class__.__name__
        data = "{\n" + indent(repr(self._data)[1:], "    ")
        return f"{cls_name}({data}\n)"

    def __str__(self) -> str:
        """Return the string representation."""
        return repr(self)

    # ===============================================================
    # Collection

    def asdict(
        self,
        *,
        dict_factory: Callable[[Any], Mapping[str, Quantity]] = dict,
    ) -> Mapping[str, Quantity]:
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
    def components(cls) -> tuple[str, ...]:
        """Vector component names."""
        raise NotImplementedError  # TODO: implement this

    @property
    def units(self) -> MappingProxyType[str, Unit]:
        """Get the units of the vector's components."""
        raise NotImplementedError  # TODO: implement this

    @override
    @property
    def dtypes(self) -> MappingProxyType[str, MappingProxyType[str, jnp.dtype]]:
        """Get the dtypes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> from unxt import Quantity

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> w.dtypes
        mappingproxy({'length': mappingproxy({'x': dtype('float32'), 'y': dtype('float32'), 'z': dtype('float32')}),
                      'speed': mappingproxy({'d_x': dtype('float32'), 'd_y': dtype('float32'), 'd_z': dtype('float32')})})

        """  # noqa: E501
        return MappingProxyType({k: v.dtypes for k, v in self.items()})

    @override
    @property
    def devices(self) -> MappingProxyType[str, MappingProxyType[str, Device]]:
        """Get the devices of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> from unxt import Quantity

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
        >>> from unxt import Quantity

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
        >>> from unxt import Quantity

        >>> w = cx.Space(
        ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
        ... )

        >>> w.sizes
        mappingproxy({'length': 2, 'speed': 2})

        """
        return MappingProxyType({k: v.size for k, v in self.items()})

    # ===============================================================
    # Convenience methods

    # TODO: correctly type annotate this method.
    def represent_as(self, target: type[AbstractVector], /) -> "Space":  # pylint: disable=W0221
        """Represent the vector as another type."""
        return represent_as(self, target)

    @dispatch  # type: ignore[misc]
    @override
    def to_units(
        self: "Space", units: Mapping[u.PhysicalType | str, Unit | str], /
    ) -> "Space":
        """Convert the vector to the given units."""
        raise NotImplementedError


# ===============================================================
# Related dispatches


@dispatch  # type: ignore[misc]
def represent_as(space: Space, target: type[AbstractVector], /) -> Space:
    """Represent the current vector to the target vector."""
    return type(space)(
        {k: temp_represent_as(v, target, space) for k, v in space.items()}
    )


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `replace` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def replace(obj: Space, /, **kwargs: AbstractVector) -> Space:
    """Replace the components of the vector."""
    return type(obj)(**{**obj, **kwargs})


# NOTE: need to set the precedence because `Space` is both a `Mapping` and a
#       `dataclass`, which are both in the `field_items` dispatch table.
@dispatch(precedence=1)  # type: ignore[misc]
def field_items(obj: Space, /) -> ItemsView[str, AbstractVector]:
    """Return the items from a Space."""
    return obj.items()


# =============================================================== Temporary
# These functions are very similar to `represent_as`, but I don't think this is
# the best API. Until we figure out a better way to do this, we'll keep these
# functions here.


# TODO: should this be moved to a different file?
@dispatch
def temp_represent_as(
    current: AbstractPos, target: type[AbstractPos], space: Space, /
) -> AbstractPos:
    """Transform of Poss."""
    return represent_as(current, target)  # space is unnecessary


# TODO: should this be moved to a different file?
@dispatch
def temp_represent_as(
    current: AbstractVel, target: type[AbstractPos], space: Space, /
) -> AbstractVel:
    """Transform of Velocities."""
    return represent_as(current, target.differential_cls, space["length"])


# TODO: should this be moved to a different file?
@dispatch
def temp_represent_as(
    current: AbstractAcc, target: type[AbstractPos], space: Space, /
) -> AbstractAcc:
    """Transform of Accs."""
    return represent_as(
        current,
        target.differential_cls.differential_cls,
        space["speed"],
        space["length"],
    )
