"""ABC for collection of vectors."""

__all__ = ["AbstractVectors"]

from abc import abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any
from typing_extensions import override

import equinox as eqx
from jax import Device

import quaxed.numpy as jnp

from coordinax._src.custom_types import Unit
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector


class AbstractVectors(AbstractVector):
    """A collection of vectors.

    A concrete vector collection class may be attribute-driven, like a
    dataclasses, but must also implement aspects of the `Mapping` API.

    """

    _data: eqx.AbstractVar[dict[str, AbstractVector]]

    # ===============================================================
    # Vector API

    @classmethod
    @abstractmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the Space."""
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Collection

    def asdict(
        self,
        *,
        dict_factory: Callable[[Any], Mapping[str, AbstractVector]] = dict,
    ) -> Mapping[str, AbstractVector]:
        """Return the vector collection as a Mapping.

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
        return MappingProxyType({k: v.dtypes for k, v in self._data.items()})

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
        return MappingProxyType({k: v.devices for k, v in self._data.items()})

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
        return MappingProxyType({k: v.shape for k, v in self._data.items()})

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
        return MappingProxyType({k: v.size for k, v in self._data.items()})
