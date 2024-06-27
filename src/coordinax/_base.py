"""Representation of coordinates in different systems."""

__all__ = [
    # vector classes
    "AbstractVector",
    # other
    "ToUnitsOptions",
]

from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import astropy.units as u
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Device
from plum import dispatch
from typing_extensions import Never

import quaxed.array_api as xp
from unxt import Quantity, unitsystem

from ._utils import classproperty, dataclass_items, dataclass_values, full_shaped
from coordinax._typing import Unit

if TYPE_CHECKING:
    from typing_extensions import Self

BT = TypeVar("BT", bound="AbstractVector")


class ToUnitsOptions(Enum):
    """Options for the units argument of :meth:`AbstractVector.to_units`."""

    consistent = "consistent"
    """Convert to consistent units."""


# ===================================================================


class AbstractVector(eqx.Module):  # type: ignore[misc]
    """Base class for all vector types.

    A vector is a collection of components that can be represented in different
    coordinate systems. This class provides a common interface for all vector
    types. All fields of the vector are expected to be components of the vector.
    """

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVector"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> from coordinax import RadialPosition, SphericalPosition

        >>> RadialPosition._cartesian_cls
        <class 'coordinax...CartesianPosition1D'>

        >>> SphericalPosition._cartesian_cls
        <class 'coordinax...CartesianPosition3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch
    def constructor(
        cls: "type[AbstractVector]", obj: Mapping[str, Quantity], /
    ) -> "AbstractVector":
        """Construct a vector from a mapping.

        Parameters
        ----------
        obj : Mapping[str, `unxt.Quantity`]
            The mapping of components.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D

        >>> xs = {"x": Quantity(1, "m"), "y": Quantity(2, "m"), "z": Quantity(3, "m")}
        >>> vec = CartesianPosition3D.constructor(xs)
        >>> vec
        CartesianPosition3D(
            x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
        )

        >>> xs = {"x": Quantity([1, 2], "m"), "y": Quantity([3, 4], "m"),
        ...       "z": Quantity([5, 6], "m")}
        >>> vec = CartesianPosition3D.constructor(xs)
        >>> vec
        CartesianPosition3D(
            x=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
            z=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m"))
        )

        """
        return cls(**obj)

    @classmethod
    @dispatch
    def constructor(cls: "type[AbstractVector]", obj: Quantity, /) -> "AbstractVector":
        """Construct a vector from a Quantity array.

        The array is expected to have the components as the last dimension.

        Parameters
        ----------
        obj : Quantity[Any, (*#batch, N), "..."]
            The array of components.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D

        >>> xs = Quantity([1, 2, 3], "meter")
        >>> vec = CartesianPosition3D.constructor(xs)
        >>> vec
        CartesianPosition3D(
            x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
        )

        >>> xs = Quantity(jnp.array([[1, 2, 3], [4, 5, 6]]), "meter")
        >>> vec = CartesianPosition3D.constructor(xs)
        >>> vec
        CartesianPosition3D(
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

    # ---------------------------------------------------------------
    # Attributes

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D

        We can transpose a vector:

        >>> vec = CartesianPosition3D(x=Quantity([[0, 1], [2, 3]], "m"),
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

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition1D

        We can get the number of dimensions of a vector:

        >>> vec = CartesianPosition1D(x=Quantity([1, 2], "m"))
        >>> vec.ndim
        1

        >>> vec = CartesianPosition1D(x=Quantity([[1, 2], [3, 4]], "m"))
        >>> vec.ndim
        2

        ``ndim`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> from coordinax import CartesianPosition2D
        >>> vec = CartesianPosition2D(x=Quantity([[1, 2], [3, 4]], "m"),
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

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition1D

        We can get the shape of a vector:

        >>> vec = CartesianPosition1D(x=Quantity([1, 2], "m"))
        >>> vec.shape
        (2,)

        >>> vec = CartesianPosition1D(x=Quantity([[1, 2], [3, 4]], "m"))
        >>> vec.shape
        (2, 2)

        ``shape`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> from coordinax import CartesianPosition2D
        >>> vec = CartesianPosition2D(x=Quantity([[1, 2], [3, 4]], "m"),
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

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition1D

        We can get the size of a vector:

        >>> vec = CartesianPosition1D(x=Quantity([1, 2], "m"))
        >>> vec.size
        2

        >>> vec = CartesianPosition1D(x=Quantity([[1, 2], [3, 4]], "m"))
        >>> vec.size
        4

        ``size`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> from coordinax import CartesianPosition2D
        >>> vec = CartesianPosition2D(x=Quantity([[1, 2], [3, 4]], "m"),
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

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D

        We can transpose a vector:

        >>> vec = CartesianPosition3D(x=Quantity([[0, 1], [2, 3]], "m"),
        ...                         y=Quantity([[0, 1], [2, 3]], "m"),
        ...                         z=Quantity([[0, 1], [2, 3]], "m"))
        >>> vec.T.x
        Quantity['length'](Array([[0., 2.],
                                  [1., 3.]], dtype=float32), unit='m')

        """
        return replace(self, **{k: v.T for k, v in dataclass_items(self)})

    # ---------------------------------------------------------------
    # Methods

    def __abs__(self) -> Quantity:
        return self.norm()

    @dispatch  # type: ignore[misc]
    def __add__(self: "AbstractVector", other: Any) -> "AbstractVector":
        return NotImplemented

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
        AbstractVector
            The vector with the slice applied.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D

        We can slice a vector:

        >>> vec = CartesianPosition2D(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                           y=Quantity(0, "m"))
        >>> vec[0].x
        Quantity['length'](Array([1., 2.], dtype=float32), unit='m')

        """
        full = full_shaped(self)  # TODO: detect if need to make a full-shaped copy
        return replace(full, **{k: v[index] for k, v in dataclass_items(full)})

    @dispatch  # type: ignore[misc]
    def __mul__(self: "AbstractVector", other: Any) -> Any:
        return NotImplemented

    @abstractmethod
    def __neg__(self) -> "Self":
        raise NotImplementedError

    @dispatch  # type: ignore[misc]
    def __rmul__(self: "AbstractVector", other: Any) -> Any:
        return NotImplemented

    def __setitem__(self, k: Any, v: Any) -> Never:
        msg = f"{type(self).__name__} is immutable."
        raise TypeError(msg)

    @dispatch  # type: ignore[misc]
    def __sub__(self: "AbstractVector", other: Any) -> "AbstractVector":
        raise NotImplementedError

    @dispatch  # type: ignore[misc]
    def __truediv__(self: "AbstractVector", other: Any) -> "AbstractVector":
        return NotImplemented

    def to_device(self, device: None | Device = None) -> "Self":
        """Move the vector to a new device.

        Parameters
        ----------
        device : None, Device
            The device to move the vector to.

        Returns
        -------
        AbstractVector
            The vector moved to the new device.

        Examples
        --------
        We assume the following imports:

        >>> from jax import devices
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition1D

        We can move a vector to a new device:

        >>> vec = CartesianPosition1D(x=Quantity([1, 2], "m"))
        >>> vec.to_device(devices()[0])
        CartesianPosition1D(
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

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D

        We can flatten a vector:

        >>> vec = CartesianPosition2D(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec.flatten()
        CartesianPosition2D(
            x=Quantity[PhysicalType('length')](value=f32[4], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[1], unit=Unit("m"))
        )

        """
        return replace(self, **{k: v.flatten() for k, v in dataclass_items(self)})

    def reshape(self, *shape: Any, order: str = "C") -> "Self":
        """Reshape the components of the vector.

        Parameters
        ----------
        *shape : Any
            The new shape.
        order : str
            The order to use for the reshape.

        Returns
        -------
        AbstractVector
            The vector with the reshaped components.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D

        We can reshape a vector:

        >>> vec = CartesianPosition2D(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))

        >>> vec.reshape(4)
        CartesianPosition2D(
            x=Quantity[PhysicalType('length')](value=f32[4], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[4], unit=Unit("m"))
        )

        """
        # TODO: enable not needing to make a full-shaped copy
        full = full_shaped(self)
        return replace(
            self,
            **{k: v.reshape(*shape, order=order) for k, v in dataclass_items(full)},
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

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D

        We can get the vector as a mapping:

        >>> vec = CartesianPosition2D(x=Quantity([[1, 2], [3, 4]], "m"),
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

        >>> from coordinax import CartesianPosition2D, SphericalPosition, RadialVelocity

        We can get the components of a vector:

        >>> CartesianPosition2D.components
        ('x', 'y')
        >>> SphericalPosition.components
        ('r', 'theta', 'phi')
        >>> RadialVelocity.components
        ('d_r',)

        """
        return tuple(f.name for f in fields(cls))

    @property
    def units(self) -> Mapping[str, Unit]:
        """Get the units of the vector's components."""
        return MappingProxyType({k: v.unit for k, v in dataclass_items(self)})

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

    @dispatch
    def to_units(self, units: Any) -> "AbstractVector":
        """Convert the vector to the given units.

        Parameters
        ----------
        units : `unxt.AbstractUnitSystem`
            The units to convert to according to the physical type of the
            components.

        Examples
        --------
        >>> import astropy.units as u
        >>> from unxt import Quantity, UnitSystem
        >>> import coordinax as cx

        >>> units = UnitSystem(u.m, u.s, u.kg, u.rad)

        >>> vec = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "km"))
        >>> vec.to_units(units)
        CartesianPosition3D(
            x=Quantity[...](value=f32[], unit=Unit("m")),
            y=Quantity[...](value=f32[], unit=Unit("m")),
            z=Quantity[...](value=f32[], unit=Unit("m"))
        )

        """
        usys = unitsystem(units)
        return replace(
            self,
            **{
                k: v.to_units(usys[v.unit.physical_type])
                for k, v in dataclass_items(self)
            },
        )

    @dispatch
    def to_units(
        self, units: Mapping[u.PhysicalType | str, Unit | str], /
    ) -> "AbstractVector":
        """Convert the vector to the given units.

        Parameters
        ----------
        units : Mapping[PhysicalType | str, Unit | str]
            The units to convert to according to the physical type of the
            components.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D, SphericalPosition

        We can convert a vector to the given units:

        >>> cart = CartesianPosition2D(x=Quantity(1, "m"), y=Quantity(2, "km"))
        >>> cart.to_units({"length": "km"})
        CartesianPosition2D(
            x=Quantity[...](value=f32[], unit=Unit("km")),
            y=Quantity[...](value=f32[], unit=Unit("km"))
        )

        This also works for vectors with different units:

        >>> sph = SphericalPosition(r=Quantity(1, "m"), theta=Quantity(45, "deg"),
        ...                       phi=Quantity(3, "rad"))
        >>> sph.to_units({"length": "km", "angle": "deg"})
        SphericalPosition(
            r=Distance(value=f32[], unit=Unit("km")),
            theta=Quantity[...](value=f32[], unit=Unit("deg")),
            phi=Quantity[...](value=f32[], unit=Unit("deg")) )

        """
        # Ensure `units_` is PT -> Unit
        units_ = {u.get_physical_type(k): v for k, v in units.items()}
        # Convert to the given units
        return replace(
            self,
            **{
                k: v.to_units(units_[v.unit.physical_type])
                for k, v in dataclass_items(self)
            },
        )

    @dispatch
    def to_units(self, _: Literal[ToUnitsOptions.consistent], /) -> "AbstractVector":
        """Convert the vector to a self-consistent set of units.

        Parameters
        ----------
        units : Literal[ToUnitsOptions.consistent]
            The vector is converted to consistent units by looking for the first
            quantity with each physical type and converting all components to
            the units of that quantity.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D, SphericalPosition
        >>> from coordinax import ToUnitsOptions

        We can convert a vector to the given units:

        >>> cart = CartesianPosition2D(x=Quantity(1, "m"), y=Quantity(2, "km"))

        If all you want is to convert to consistent units, you can use
        ``"consistent"``:

        >>> cart.to_units(ToUnitsOptions.consistent)
        CartesianPosition2D(
            x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
            y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
        )

        >>> sph.to_units(ToUnitsOptions.consistent)
        SphericalPosition(
            r=Distance(value=f32[], unit=Unit("m")),
            theta=Quantity[...](value=f32[], unit=Unit("deg")),
            phi=Quantity[...](value=f32[], unit=Unit("deg")) )

        """
        units_ = {}
        for v in dataclass_values(self):
            pt = v.unit.physical_type
            if pt not in units_:
                units_[pt] = v.unit

        return replace(
            self,
            **{
                k: v.to_units(units_[v.unit.physical_type])
                for k, v in dataclass_items(self)
            },
        )

    # ===============================================================
    # Misc

    def __str__(self) -> str:
        r"""Return a string representation of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> vec = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "m"))
        >>> str(vec)
        '<CartesianPosition3D (x[m], y[m], z[m])\n    [1. 2. 3.]>'

        """
        cls_name = type(self).__name__
        units = self.units
        comps = ", ".join(f"{c}[{units[c]}]" for c in self.components)
        vs = np.array2string(
            xp.stack(
                tuple(v.value for v in xp.broadcast_arrays(*dataclass_values(self))),
                axis=-1,
            ),
            precision=3,
            prefix="    ",
        )
        return f"<{cls_name} ({comps})\n    {vs}>"


# -----------------------------------------------
# Register additional constructors


# TODO: move to the class in py3.11+
@AbstractVector.constructor._f.dispatch  # noqa: SLF001
def constructor(  # noqa: D417
    cls: type[AbstractVector], obj: AbstractVector, /
) -> AbstractVector:
    """Construct a vector from another vector.

    Parameters
    ----------
    obj : :class:`coordinax.AbstractVector`
        The vector to construct from.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from unxt import Quantity
    >>> from coordinax import CartesianPosition3D

    >>> x, y, z = Quantity(1, "meter"), Quantity(2, "meter"), Quantity(3, "meter")
    >>> vec = CartesianPosition3D(x=x, y=y, z=z)
    >>> cart = CartesianPosition3D.constructor(vec)
    >>> cart
    CartesianPosition3D(
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


@AbstractVector.constructor._f.dispatch  # noqa: SLF001
def constructor(
    cls: type[AbstractVector], obj: Mapping[str, u.Quantity], /
) -> AbstractVector:
    """Construct a vector from a mapping.

    Parameters
    ----------
    cls : type[AbstractVector]
        The vector class.
    obj : Mapping[str, `astropy.units.Quantity`]
        The mapping of components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astropy.units import Quantity
    >>> from coordinax import CartesianPosition3D

    >>> xs = {"x": Quantity(1, "m"), "y": Quantity(2, "m"), "z": Quantity(3, "m")}
    >>> vec = CartesianPosition3D.constructor(xs)
    >>> vec
    CartesianPosition3D(
        x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
    )

    >>> xs = {"x": Quantity([1, 2], "m"), "y": Quantity([3, 4], "m"),
    ...       "z": Quantity([5, 6], "m")}
    >>> vec = CartesianPosition3D.constructor(xs)
    >>> vec
    CartesianPosition3D(
        x=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
        y=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")),
        z=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m"))
    )

    """
    return cls(**obj)


# TODO: move to the class in py3.11+
@AbstractVector.constructor._f.dispatch  # noqa: SLF001
def constructor(cls: type[AbstractVector], obj: u.Quantity, /) -> AbstractVector:
    """Construct a vector from an Astropy Quantity array.

    The array is expected to have the components as the last dimension.

    Parameters
    ----------
    cls : type[AbstractVector]
        The vector class.
    obj : Quantity[Any, (*#batch, N), "..."]
        The array of components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astropy.units import Quantity
    >>> from coordinax import CartesianPosition3D

    >>> xs = Quantity([1, 2, 3], "meter")
    >>> vec = CartesianPosition3D.constructor(xs)
    >>> vec
    CartesianPosition3D(
        x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
    )

    >>> xs = Quantity(jnp.array([[1, 2, 3], [4, 5, 6]]), "meter")
    >>> vec = CartesianPosition3D.constructor(xs)
    >>> vec
    CartesianPosition3D(
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
    return cls(**{f.name: obj[..., i] for i, f in enumerate(fields(cls))})
