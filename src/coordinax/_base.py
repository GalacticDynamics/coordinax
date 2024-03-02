"""Representation of coordinates in different systems."""

__all__ = ["AbstractVectorBase", "AbstractVector", "AbstractVectorDifferential"]

import operator
import warnings
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Device
from jaxtyping import ArrayLike
from plum import dispatch

import array_api_jax_compat as xp
from jax_quantity import Quantity

from ._utils import classproperty, dataclass_items, dataclass_values, full_shaped
from coordinax._typing import Unit

if TYPE_CHECKING:
    from typing_extensions import Self

BT = TypeVar("BT", bound="AbstractVectorBase")
VT = TypeVar("VT", bound="AbstractVector")
DT = TypeVar("DT", bound="AbstractVectorDifferential")


_0m = Quantity(0, "meter")
_0d = Quantity(0, "rad")
_pid = Quantity(xp.pi, "rad")
_2pid = Quantity(2 * xp.pi, "rad")


class AbstractVectorBase(eqx.Module):  # type: ignore[misc]
    """Base class for all vector types.

    A vector is a collection of components that can be represented in different
    coordinate systems. This class provides a common interface for all vector
    types. All fields of the vector are expected to be components of the vector.
    """

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVectorBase"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> from coordinax import RadialVector, SphericalVector

        >>> RadialVector._cartesian_cls
        <class 'coordinax._d1.builtin.Cartesian1DVector'>

        >>> SphericalVector._cartesian_cls
        <class 'coordinax._d3.builtin.Cartesian3DVector'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch
    def constructor(
        cls: "type[AbstractVectorBase]", obj: Mapping[str, Quantity], /
    ) -> "AbstractVectorBase":
        """Construct a vector from a mapping.

        Parameters
        ----------
        obj : Mapping[str, Any]
            The mapping of components.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector

        >>> xs = {"x": Quantity(1, "m"), "y": Quantity(2, "m"), "z": Quantity(3, "m")}
        >>> vec = Cartesian3DVector.constructor(xs)
        >>> vec
        Cartesian3DVector(
            x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
        )

        >>> xs = {"x": Quantity([1, 2], "m"), "y": Quantity([3, 4], "m"),
        ...       "z": Quantity([5, 6], "m")}
        >>> vec = Cartesian3DVector.constructor(xs)
        >>> vec
        Cartesian3DVector(
            x=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
            z=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m"))
        )

        """
        return cls(**obj)

    @classmethod
    @dispatch
    def constructor(
        cls: "type[AbstractVectorBase]",
        obj: Quantity | u.Quantity,
        /,  # TODO: shape hint
    ) -> "AbstractVectorBase":
        """Construct a vector from a Quantity array.

        The array is expected to have the components as the last dimension.

        Parameters
        ----------
        obj : Quantity[Any, (*#batch, N), "..."]
            The array of components.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector

        >>> xs = Quantity([1, 2, 3], "meter")
        >>> vec = Cartesian3DVector.constructor(xs)
        >>> vec
        Cartesian3DVector(
            x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
        )

        >>> xs = Quantity(jnp.array([[1, 2, 3], [4, 5, 6]]), "meter")
        >>> vec = Cartesian3DVector.constructor(xs)
        >>> vec
        Cartesian3DVector(
            x=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
            z=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m"))
        )
        >>> vec.x
        Quantity['length'](Array([1., 4.], dtype=float32), unit='m')

        """
        _ = eqx.error_if(
            obj,
            obj.shape[-1] != len(fields(cls)),
            f"Cannot construct {cls} from array with shape {obj.shape}.",
        )
        comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
        return cls(**comps)

    # ===============================================================
    # Array API

    def __array_namespace__(self) -> "ArrayAPINamespace":
        """Return the array API namespace."""
        return xp

    def __getitem__(self, index: Any) -> "Self":
        """Return a new object with the given slice applied.

        Parameters
        ----------
        index : Any
            The slice to apply.

        Returns
        -------
        AbstractVectorBase
            The vector with the slice applied.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian2DVector

        We can slice a vector:

        >>> vec = Cartesian2DVector(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec[0].x
        Quantity['length'](Array([1., 2.], dtype=float32), unit='m')

        """
        full = full_shaped(self)  # TODO: detect if need to make a full-shaped copy
        return replace(full, **{k: v[index] for k, v in dataclass_items(full)})

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector

        We can transpose a vector:

        >>> vec = Cartesian3DVector(x=Quantity([[0, 1], [2, 3]], "m"),
        ...                         y=Quantity([[0, 1], [2, 3]], "m"),
        ...                         z=Quantity([[0, 1], [2, 3]], "m"))
        >>> vec.mT.x
        Quantity['length'](Array([[0., 2.],
                                  [1., 3.]], dtype=float32), unit='m')

        """
        return replace(self, **{k: v.mT for k, v in dataclass_items(self)})

    @property
    def ndim(self) -> int:
        """Number of array dimensions (axes).

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian1DVector

        We can get the number of dimensions of a vector:

        >>> vec = Cartesian1DVector(x=Quantity([1, 2], "m"))
        >>> vec.ndim
        1

        >>> vec = Cartesian1DVector(x=Quantity([[1, 2], [3, 4]], "m"))
        >>> vec.ndim
        2

        ``ndim`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> from coordinax import Cartesian2DVector
        >>> vec = Cartesian2DVector(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec.ndim
        2

        """
        return len(self.shape)

    @property
    def shape(self) -> Any:
        """Get the shape of the vector's components.

        When represented as a single array, the vector has an additional
        dimension at the end for the components.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian1DVector

        We can get the shape of a vector:

        >>> vec = Cartesian1DVector(x=Quantity([1, 2], "m"))
        >>> vec.shape
        (2,)

        >>> vec = Cartesian1DVector(x=Quantity([[1, 2], [3, 4]], "m"))
        >>> vec.shape
        (2, 2)

        ``shape`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> from coordinax import Cartesian2DVector
        >>> vec = Cartesian2DVector(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec.shape
        (2, 2)

        """
        return jnp.broadcast_shapes(*self.shapes.values())

    @property
    def size(self) -> int:
        """Total number of elements in the vector.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian1DVector

        We can get the size of a vector:

        >>> vec = Cartesian1DVector(x=Quantity([1, 2], "m"))
        >>> vec.size
        2

        >>> vec = Cartesian1DVector(x=Quantity([[1, 2], [3, 4]], "m"))
        >>> vec.size
        4

        ``size`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> from coordinax import Cartesian2DVector
        >>> vec = Cartesian2DVector(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec.size
        4

        """
        return int(jnp.prod(xp.asarray(self.shape)))

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector

        We can transpose a vector:

        >>> vec = Cartesian3DVector(x=Quantity([[0, 1], [2, 3]], "m"),
        ...                         y=Quantity([[0, 1], [2, 3]], "m"),
        ...                         z=Quantity([[0, 1], [2, 3]], "m"))
        >>> vec.T.x
        Quantity['length'](Array([[0., 2.],
                                  [1., 3.]], dtype=float32), unit='m')

        """
        return replace(self, **{k: v.T for k, v in dataclass_items(self)})

    def to_device(self, device: None | Device = None) -> "Self":
        """Move the vector to a new device.

        Parameters
        ----------
        device : None, Device
            The device to move the vector to.

        Returns
        -------
        AbstractVectorBase
            The vector moved to the new device.

        Examples
        --------
        We assume the following imports:

        >>> from jax import devices
        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian1DVector

        We can move a vector to a new device:

        >>> vec = Cartesian1DVector(x=Quantity([1, 2], "m"))
        >>> vec.to_device(devices()[0])
        Cartesian1DVector(
            x=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m"))
        )

        """
        return replace(
            self, **{k: v.to_device(device) for k, v in dataclass_items(self)}
        )

    # ===============================================================
    # Further array methods

    def flatten(self) -> "Self":
        """Flatten the vector.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian2DVector

        We can flatten a vector:

        >>> vec = Cartesian2DVector(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec.flatten()
        Cartesian2DVector(
            x=Quantity[PhysicalType('length')](value=f32[4], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[1], unit=Unit("m"))
        )

        """
        return replace(self, **{k: v.flatten() for k, v in dataclass_items(self)})

    def reshape(self, *args: Any, order: str = "C") -> "Self":
        """Reshape the components of the vector.

        Parameters
        ----------
        *args : Any
            The new shape.
        order : str
            The order to use for the reshape.

        Returns
        -------
        AbstractVectorBase
            The vector with the reshaped components.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian2DVector

        We can reshape a vector:

        >>> vec = Cartesian2DVector(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))

        >>> vec.reshape(4)
        Cartesian2DVector(
            x=Quantity[PhysicalType('length')](value=f32[4], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[4], unit=Unit("m"))
        )

        """
        # TODO: enable not needing to make a full-shaped copy
        full = full_shaped(self)
        return replace(
            self, **{k: v.reshape(*args, order=order) for k, v in dataclass_items(full)}
        )

    # ===============================================================
    # Collection

    def asdict(
        self,
        *,
        dict_factory: Callable[[Any], Mapping[str, Quantity]] = dict,  # TODO: full hint
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

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian2DVector

        We can get the vector as a mapping:

        >>> vec = Cartesian2DVector(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec.asdict()
        {'x': Quantity['length'](Array([[1., 2.], [3., 4.]], dtype=float32), unit='m'),
         'y': Quantity['length'](Array(0., dtype=float32), unit='m')}

        """
        return dict_factory(dataclass_items(self))

    @classproperty
    @classmethod
    def components(cls) -> tuple[str, ...]:
        """Vector component names.

        Examples
        --------
        We assume the following imports:

        >>> from coordinax import Cartesian2DVector, SphericalVector, RadialDifferential

        We can get the components of a vector:

        >>> Cartesian2DVector.components
        ('x', 'y')
        >>> SphericalVector.components
        ('r', 'theta', 'phi')
        >>> RadialDifferential.components
        ('d_r',)

        """
        return tuple(f.name for f in fields(cls))

    @property
    def dtypes(self) -> Mapping[str, jnp.dtype]:
        """Get the dtypes of the vector's components."""
        return MappingProxyType({k: v.dtype for k, v in dataclass_items(self)})

    @property
    def devices(self) -> Mapping[str, Device]:
        """Get the devices of the vector's components."""
        return MappingProxyType({k: v.device for k, v in dataclass_items(self)})

    @property
    def shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Get the shapes of the vector's components."""
        return MappingProxyType({k: v.shape for k, v in dataclass_items(self)})

    @property
    def sizes(self) -> Mapping[str, int]:
        """Get the sizes of the vector's components."""
        return MappingProxyType({k: v.size for k, v in dataclass_items(self)})

    # ===============================================================
    # Convenience methods

    @abstractmethod
    def represent_as(self, target: type[BT], /, *args: Any, **kwargs: Any) -> BT:
        """Represent the vector as another type."""
        raise NotImplementedError

    def to_units(
        self,
        units: (
            Mapping[u.PhysicalType | str, Unit] | Literal["consistent"]
        ) = "consistent",
        /,
    ) -> "Self":
        """Convert the vector to the given units.

        Parameters
        ----------
        units : Mapping[PhysicalType | str, Unit] | Literal["consistent"]
            The units to convert to according to the physical type of the
            components. If "consistent", the vector is converted to consistent
            units by looking for the first quantity with each physical type and
            converting all components to the units of that quantity.

        Returns
        -------
        AbstractVectorBase
            The vector with the components converted to the given units.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian2DVector, SphericalVector

        We can convert a vector to the given units:

        >>> cart = Cartesian2DVector(x=Quantity(1, "m"), y=Quantity(2, "km"))
        >>> cart.to_units({"length": "km"})
        Cartesian2DVector(
            x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("km")),
            y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("km"))
        )

        This also works for vectors with different units:

        >>> sph = SphericalVector(r=Quantity(1, "m"), theta=Quantity(45, "deg"),
        ...                       phi=Quantity(3, "rad"))
        >>> sph.to_units({"length": "km", "angle": "deg"})
        SphericalVector(
            r=Quantity[PhysicalType('length')](value=f32[], unit=Unit("km")),
            theta=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
            phi=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg"))
        )

        If all you want is to convert to consistent units, you can use
        ``"consistent"``:

        >>> cart.to_units("consistent")
        Cartesian2DVector(
            x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
        )

        >>> sph.to_units("consistent")
        SphericalVector(
            r=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            theta=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg")),
            phi=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("deg"))
        )

        """
        if units != "consistent":
            units_ = {u.get_physical_type(k): v for k, v in units.items()}
        else:
            units_ = {}
            for v in dataclass_values(self):
                pt = v.unit.physical_type
                if pt not in units_:
                    units_[pt] = v.unit

        return replace(
            self,
            **{k: v.to(units_[v.unit.physical_type]) for k, v in dataclass_items(self)},
        )


# -----------------------------------------------
# Register additional constructors


# TODO: move to the class in py3.11+
@AbstractVectorBase.constructor._f.register  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(  # noqa: D417
    cls: type[AbstractVectorBase], obj: AbstractVectorBase, /
) -> AbstractVectorBase:
    """Construct a vector from another vector.

    Parameters
    ----------
    obj : :class:`coordinax.AbstractVectorBase`
        The vector to construct from.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax_quantity import Quantity
    >>> from coordinax import Cartesian3DVector

    >>> x, y, z = Quantity(1, "meter"), Quantity(2, "meter"), Quantity(3, "meter")
    >>> vec = Cartesian3DVector(x=x, y=y, z=z)
    >>> cart = Cartesian3DVector.constructor(vec)
    >>> cart
    Cartesian3DVector(
      x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
      y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
      z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
    )
    >>> cart.x
    Quantity['length'](Array(1., dtype=float32), unit='m')

    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls} from {type(obj)}."
        raise TypeError(msg)

    # avoid copying if the types are the same. Isinstance is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(dataclass_items(obj)))


#####################################################################


class AbstractVector(AbstractVectorBase):  # pylint: disable=abstract-method
    """Abstract representation of coordinates in different systems."""

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVectorDifferential"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> from coordinax import RadialVector, SphericalVector

        >>> RadialVector.differential_cls
        <class 'coordinax._d1.builtin.RadialDifferential'>

        >>> SphericalVector.differential_cls
        <class 'coordinax._d3.builtin.SphericalDifferential'>

        """
        raise NotImplementedError

    # ===============================================================
    # Array

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        The default implementation is to go through Cartesian coordinates.
        """
        cart = self.represent_as(self._cartesian_cls)
        return (-cart).represent_as(type(self))

    # -----------------------------------------------------
    # Binary arithmetic operations

    def __add__(self, other: Any) -> "Self":
        """Add another object to this vector."""
        if not isinstance(other, AbstractVector):
            return NotImplemented

        # The base implementation is to convert to Cartesian and perform the
        # operation.  Cartesian coordinates do not have any branch cuts or
        # singularities or ranges that need to be handled, so this is a safe
        # default.
        return operator.add(
            self.represent_as(self._cartesian_cls),
            other.represent_as(self._cartesian_cls),
        ).represent_as(type(self))

    def __sub__(self, other: Any) -> "Self":
        """Add another object to this vector."""
        if not isinstance(other, AbstractVector):
            return NotImplemented

        # The base implementation is to convert to Cartesian and perform the
        # operation.  Cartesian coordinates do not have any branch cuts or
        # singularities or ranges that need to be handled, so this is a safe
        # default.
        return operator.sub(
            self.represent_as(self._cartesian_cls),
            other.represent_as(self._cartesian_cls),
        ).represent_as(type(self))

    @dispatch
    def __mul__(self: "AbstractVector", other: Any) -> Any:
        return NotImplemented

    @dispatch
    def __mul__(self: "AbstractVector", other: ArrayLike) -> Any:
        return replace(self, **{k: v * other for k, v in dataclass_items(self)})

    @dispatch
    def __truediv__(self: "AbstractVector", other: Any) -> Any:
        return NotImplemented

    @dispatch
    def __truediv__(self: "AbstractVector", other: ArrayLike) -> Any:
        return replace(self, **{k: v / other for k, v in dataclass_items(self)})

    # ---------------------------------
    # Reverse binary operations

    @dispatch
    def __rmul__(self: "AbstractVector", other: Any) -> Any:
        return NotImplemented

    @dispatch
    def __rmul__(self: "AbstractVector", other: ArrayLike) -> Any:
        return replace(self, **{k: other * v for k, v in dataclass_items(self)})

    # ===============================================================
    # Convenience methods

    @partial(jax.jit, static_argnums=1)
    def represent_as(self, target: type[VT], /, *args: Any, **kwargs: Any) -> VT:
        """Represent the vector as another type.

        Parameters
        ----------
        target : type[AbstractVector]
            The type to represent the vector as.
        *args : Any
            Extra arguments. Raises a warning if any are given.
        **kwargs : Any
            Extra keyword arguments.

        Returns
        -------
        AbstractVector
            The vector represented as the target type.

        Warns
        -----
        UserWarning
            If extra arguments are given.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector, SphericalVector

        We can represent a vector as another type:

        >>> x, y, z = Quantity(1, "meter"), Quantity(2, "meter"), Quantity(3, "meter")
        >>> vec = Cartesian3DVector(x=x, y=y, z=z)
        >>> sph = vec.represent_as(SphericalVector)
        >>> sph
        SphericalVector(
          r=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
          theta=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("rad")),
          phi=Quantity[PhysicalType('angle')](value=f32[], unit=Unit("rad"))
        )
        >>> sph.r
        Quantity['length'](Array(3.7416575, dtype=float32), unit='m')

        """
        if any(args):
            warnings.warn("Extra arguments are ignored.", UserWarning, stacklevel=2)

        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, **kwargs)

    @partial(jax.jit)
    def norm(self) -> Quantity["length"]:
        """Return the norm of the vector.

        Returns
        -------
        Quantity
            The norm of the vector.

        Examples
        --------
        We assume the following imports:
        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector

        We can compute the norm of a vector
        >>> x, y, z = Quantity(1, "meter"), Quantity(2, "meter"), Quantity(3, "meter")
        >>> vec = Cartesian3DVector(x=x, y=y, z=z)
        >>> vec.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='m')

        """
        return self.represent_as(self._cartesian_cls).norm()


#####################################################################


class AbstractVectorDifferential(AbstractVectorBase):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type["AbstractVectorDifferential"]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> from coordinax import RadialDifferential, SphericalDifferential

        >>> RadialDifferential.integral_cls
        <class 'coordinax._d1.builtin.RadialVector'>

        >>> SphericalDifferential.integral_cls
        <class 'coordinax._d3.builtin.SphericalVector'>

        """
        raise NotImplementedError

    # ===============================================================
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import RadialDifferential
        >>> dr = RadialDifferential(Quantity(1, "m/s"))
        >>> -dr
        RadialDifferential( d_r=Quantity[...]( value=f32[], unit=Unit("m / s") ) )

        >>> from coordinax import PolarDifferential
        >>> dp = PolarDifferential(Quantity(1, "m/s"), Quantity(1, "mas/yr"))
        >>> neg_dp = -dp
        >>> neg_dp.d_r
        Quantity['speed'](Array(-1., dtype=float32), unit='m / s')
        >>> neg_dp.d_phi
        Quantity['angular frequency'](Array(-1., dtype=float32), unit='mas / yr')

        """
        return replace(self, **{k: -v for k, v in dataclass_items(self)})

    # ===============================================================
    # Binary operations

    @dispatch  # type: ignore[misc]
    def __mul__(
        self: "AbstractVectorDifferential", other: Quantity
    ) -> "AbstractVector":
        """Multiply the vector by a :class:`jax_quantity.Quantity`.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import RadialDifferential

        >>> dr = RadialDifferential(Quantity(1, "m/s"))
        >>> vec = dr * Quantity(2, "s")
        >>> vec
        RadialVector(r=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")))
        >>> vec.r
        Quantity['length'](Array(2., dtype=float32), unit='m')

        """
        return self.integral_cls.constructor(
            {k[2:]: v * other for k, v in dataclass_items(self)}
        )

    # ===============================================================
    # Convenience methods

    @partial(jax.jit, static_argnums=1)
    def represent_as(
        self, target: type[DT], position: AbstractVector, /, *args: Any, **kwargs: Any
    ) -> DT:
        """Represent the vector as another type."""
        if any(args):
            warnings.warn("Extra arguments are ignored.", UserWarning, stacklevel=2)

        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, position, **kwargs)

    @partial(jax.jit)
    def norm(self, position: AbstractVector, /) -> Quantity["speed"]:
        """Return the norm of the vector."""
        return self.represent_as(self._cartesian_cls, position).norm()
