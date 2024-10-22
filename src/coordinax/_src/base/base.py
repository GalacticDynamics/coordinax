"""Representation of coordinates in different systems."""

__all__ = [
    # vector classes
    "AbstractVector",
    # other
    "ToUnitsOptions",
]

from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import fields
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, TypeVar

import jax
import numpy as np
from astropy.units import PhysicalType as Dimensions
from jax import Device, tree
from jaxtyping import Array, ArrayLike, Bool
from plum import dispatch
from quax import ArrayValue, register

import quaxed.lax as qlax
import quaxed.numpy as jnp
from dataclassish import field_items, field_values, replace
from unxt import (
    AbstractQuantity,
    Quantity,
    dimensions,
    dimensions_of,
    uconvert,
    units_of,
    unitsystem,
)

from coordinax._src.typing import Unit
from coordinax._src.utils import classproperty, full_shaped

if TYPE_CHECKING:
    from typing_extensions import Self

VT = TypeVar("VT", bound="AbstractVector")


class ToUnitsOptions(Enum):
    """Options for the units argument of `AbstractVector.to_units`."""

    consistent = "consistent"
    """Convert to consistent units."""


# ===================================================================


class AbstractVector(ArrayValue):  # type: ignore[misc]
    """Base class for all vector types.

    A vector is a collection of components that can be represented in different
    coordinate systems. This class provides a common interface for all vector
    types. All fields of the vector are expected to be components of the vector.
    """

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @dispatch
    def from_(
        cls: "type[AbstractVector]", obj: Mapping[str, AbstractQuantity], /
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
        >>> import coordinax as cx

        >>> xs = {"x": Quantity(1, "m"), "y": Quantity(2, "m"), "z": Quantity(3, "m")}
        >>> vec = cx.CartesianPos3D.from_(xs)
        >>> vec
        CartesianPos3D(
            x=Quantity[...](value=f32[], unit=Unit("m")),
            y=Quantity[...](value=f32[], unit=Unit("m")),
            z=Quantity[...](value=f32[], unit=Unit("m"))
        )

        >>> xs = {"x": Quantity([1, 2], "m"), "y": Quantity([3, 4], "m"),
        ...       "z": Quantity([5, 6], "m")}
        >>> vec = cx.CartesianPos3D.from_(xs)
        >>> vec
        CartesianPos3D(
            x=Quantity[...](value=f32[2], unit=Unit("m")),
            y=Quantity[...](value=f32[2], unit=Unit("m")),
            z=Quantity[...](value=f32[2], unit=Unit("m"))
        )

        """
        return cls(**obj)

    @classmethod
    @dispatch
    def from_(
        cls: "type[AbstractVector]", obj: ArrayLike | list[Any], unit: Unit | str, /
    ) -> "AbstractVector":
        """Construct a vector from an array and unit.

        The array is expected to have the components as the last dimension.

        Parameters
        ----------
        obj : ArrayLike[Any, (*#batch, N), "..."]
            The array of components.
        unit : Unit | str
            The unit of the quantity

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "meter")
        >>> vec
        CartesianPos3D(
            x=Quantity[...](value=f32[], unit=Unit("m")),
            y=Quantity[...](value=f32[], unit=Unit("m")),
            z=Quantity[...](value=f32[], unit=Unit("m"))
        )

        >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> vec = cx.CartesianPos3D.from_(xs, "meter")
        >>> vec
        CartesianPos3D(
            x=Quantity[...](value=f32[2], unit=Unit("m")),
            y=Quantity[...](value=f32[2], unit=Unit("m")),
            z=Quantity[...](value=f32[2], unit=Unit("m"))
        )
        >>> vec.x
        Quantity['length'](Array([1., 4.], dtype=float32), unit='m')

        """
        obj = Quantity.from_(jnp.asarray(obj), unit)
        return cls.from_(obj)  # re-dispatch

    # ===============================================================
    # Quax

    def materialise(self) -> NoReturn:
        """Materialise the vector for `quax`.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")

        >>> try: vec.materialise()
        ... except RuntimeError as e: print(e)
        Refusing to materialise `CartesianPos3D`.

        """
        msg = f"Refusing to materialise `{type(self).__name__}`."
        raise RuntimeError(msg)

    @abstractmethod
    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        raise NotImplementedError  # pragma: no cover

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
        >>> import coordinax as cx

        We can transpose a vector:

        >>> vec = cx.CartesianPos3D(x=Quantity([[0, 1], [2, 3]], "m"),
        ...                         y=Quantity([[0, 1], [2, 3]], "m"),
        ...                         z=Quantity([[0, 1], [2, 3]], "m"))
        >>> vec.mT.x
        Quantity['length'](Array([[0., 2.],
                                  [1., 3.]], dtype=float32), unit='m')

        """
        return replace(self, **{k: v.mT for k, v in field_items(self)})

    @property
    def ndim(self) -> int:
        """Number of array dimensions (axes).

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> import coordinax as cx

        We can get the number of dimensions of a vector:

        >>> vec = cx.CartesianPos2D.from_([1, 2], "m")
        >>> vec.ndim
        0

        >>> vec = cx.CartesianPos2D.from_([[1, 2], [3, 4]], "m")
        >>> vec.ndim
        1

        ``ndim`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> vec = cx.CartesianPos2D(x=Quantity([[1, 2], [3, 4]], "m"),
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
        >>> import coordinax as cx

        We can get the shape of a vector:

        >>> vec = cx.CartesianPos1D(x=Quantity([1, 2], "m"))
        >>> vec.shape
        (2,)

        >>> vec = cx.CartesianPos1D(x=Quantity([[1, 2], [3, 4]], "m"))
        >>> vec.shape
        (2, 2)

        ``shape`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> vec = cx.CartesianPos2D(x=Quantity([[1, 2], [3, 4]], "m"),
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
        >>> import coordinax as cx

        We can get the size of a vector:

        >>> vec = cx.CartesianPos2D.from_([1, 2], "m")
        >>> vec.size
        1

        >>> vec = cx.CartesianPos2D.from_([[1, 2], [3, 4]], "m")
        >>> vec.size
        2

        ``size`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> vec = cx.CartesianPos2D(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec.size
        4

        """
        return int(jnp.prod(jnp.asarray(self.shape)))

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> import coordinax as cx

        We can transpose a vector:

        >>> vec = cx.CartesianPos3D(x=Quantity([[0, 1], [2, 3]], "m"),
        ...                         y=Quantity([[0, 1], [2, 3]], "m"),
        ...                         z=Quantity([[0, 1], [2, 3]], "m"))
        >>> vec.T.x
        Quantity['length'](Array([[0., 2.],
                                  [1., 3.]], dtype=float32), unit='m')

        """
        return replace(self, **{k: v.T for k, v in field_items(self)})

    # ---------------------------------------------------------------
    # Methods

    def __eq__(self: "AbstractVector", other: object) -> Any:
        """Check if the vector is equal to another object.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> from unxt import Quantity
        >>> import coordinax as cx

        Positions are covered by a separate dispatch. So here we show velocities
        and accelerations:

        >>> vel1 = cx.CartesianVel1D(Quantity([1, 2, 3], "km/s"))
        >>> vel2 = cx.CartesianVel1D(Quantity([1, 0, 3], "km/s"))
        >>> jnp.equal(vel1, vel2)
        Array([ True,  False,  True], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False,  True], dtype=bool)

        >>> acc1 = cx.CartesianAcc1D(Quantity([1, 2, 3], "km/s2"))
        >>> acc2 = cx.CartesianAcc1D(Quantity([1, 0, 3], "km/s2"))
        >>> jnp.equal(acc1, acc2)
        Array([ True,  False,  True], dtype=bool)
        >>> acc1 == acc2
        Array([ True, False,  True], dtype=bool)

        >>> vel1 = cx.RadialVel(Quantity([1, 2, 3], "km/s"))
        >>> vel2 = cx.RadialVel(Quantity([1, 0, 3], "km/s"))
        >>> jnp.equal(vel1, vel2)
        Array([ True,  False,  True], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False,  True], dtype=bool)

        >>> acc1 = cx.RadialAcc(Quantity([1, 2, 3], "km/s2"))
        >>> acc2 = cx.RadialAcc(Quantity([1, 0, 3], "km/s2"))
        >>> jnp.equal(acc1, acc2)
        Array([ True,  False,  True], dtype=bool)
        >>> acc1 == acc2
        Array([ True, False,  True], dtype=bool)

        >>> vel1 = cx.CartesianVel2D.from_([[1, 3], [2, 4]], "km/s")
        >>> vel2 = cx.CartesianVel2D.from_([[1, 3], [0, 4]], "km/s")
        >>> vel1.d_x
        Quantity['speed'](Array([1., 2.], dtype=float32), unit='km / s')
        >>> jnp.equal(vel1, vel2)
        Array([ True, False], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False], dtype=bool)

        >>> acc1 = cx.CartesianAcc2D.from_([[1, 3], [2, 4]], "km/s2")
        >>> acc2 = cx.CartesianAcc2D.from_([[1, 3], [0, 4]], "km/s2")
        >>> acc1.d2_x
        Quantity['acceleration'](Array([1., 2.], dtype=float32), unit='km / s2')
        >>> jnp.equal(acc1, acc2)
        Array([ True, False], dtype=bool)
        >>> acc1 == acc2
        Array([ True, False], dtype=bool)

        >>> vel1 = cx.CartesianVel3D.from_([[1, 4], [2, 5], [3, 6]], "km/s")
        >>> vel2 = cx.CartesianVel3D.from_([[1, 4], [0, 5], [3, 0]], "km/s")
        >>> vel1.d_x
        Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')
        >>> jnp.equal(vel1, vel2)
        Array([ True, False, False], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False, False], dtype=bool)

        """
        if type(other) is not type(self):
            return NotImplemented

        comp_tree = tree.map(jnp.equal, self, other)
        comp_leaves = jnp.array(tree.leaves(comp_tree))
        return jax.numpy.logical_and.reduce(comp_leaves)

    def __len__(self) -> int:
        """Return the length of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        Scalar vectors have length 0:

        >>> vec = cx.CartesianPos1D.from_([1], "m")
        >>> len(vec)
        0

        Vectors with certain lengths:

        >>> vec = cx.CartesianPos1D(Quantity([1], "m"))
        >>> len(vec)
        1

        >>> vec = cx.CartesianPos1D(Quantity([1, 2], "m"))
        >>> len(vec)
        2

        """
        return self.shape[0] if self.ndim > 0 else 0

    def __abs__(self) -> Quantity:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos2D.from_([3, 4], "m")
        >>> abs(vec)
        Quantity['length'](Array(5., dtype=float32), unit='m')

        """
        return self.norm()

    def __array_namespace__(self) -> "ArrayAPINamespace":
        """Return the array API namespace.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos2D.from_([3, 4], "m")
        >>> vec.__array_namespace__()
        <module 'quaxed.numpy' from ...>

        """
        return jnp

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
        >>> import coordinax as cx

        We can slice a vector:

        >>> vec = cx.CartesianPos2D(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec[0].x
        Quantity['length'](Array([1., 2.], dtype=float32), unit='m')

        """
        full = full_shaped(self)  # TODO: detect if need to make a full-shaped copy
        return replace(full, **{k: v[index] for k, v in field_items(full)})

    def __add__(self: "AbstractVector", other: Any) -> "AbstractVector":
        """Add another object to this vector."""
        return qlax.add(self, other)

    def __mul__(self: "AbstractVector", other: Any) -> Any:
        """Multiply the vector by a scalar.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> (vec * 2).x
        Quantity['length'](Array(2., dtype=float32), unit='m')

        """
        return qlax.mul(self, other)

    @abstractmethod
    def __neg__(self) -> "Self":
        raise NotImplementedError

    def __rmul__(self: "AbstractVector", other: Any) -> Any:
        """Multiply the vector by a scalar.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> (2 * vec).x
        Quantity['length'](Array(2., dtype=float32), unit='m')

        """
        return qlax.mul(other, self)

    def __setitem__(self, k: Any, v: Any) -> NoReturn:
        msg = f"{type(self).__name__} is immutable."
        raise TypeError(msg)

    def __sub__(self: "AbstractVector", other: Any) -> "AbstractVector":
        """Subtract an object from this vector."""
        return qlax.sub(self, other)

    def __truediv__(self: "AbstractVector", other: Any) -> "AbstractVector":
        return qlax.div(self, other)

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
        >>> from jax import devices
        >>> from unxt import Quantity
        >>> import coordinax as cx

        We can move a vector to a new device:

        >>> vec = cx.CartesianPos1D(Quantity([1, 2], "m"))
        >>> vec.to_device(devices()[0])
        CartesianPos1D(x=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")))

        """
        return replace(self, **{k: v.to_device(device) for k, v in field_items(self)})

    # ===============================================================
    # Further array methods

    def flatten(self) -> "Self":
        """Flatten the vector.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> import coordinax as cx

        We can flatten a vector:

        >>> vec = cx.CartesianPos2D(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec.flatten()
        CartesianPos2D(
            x=Quantity[...](value=f32[4], unit=Unit("m")),
            y=Quantity[...](value=f32[1], unit=Unit("m"))
        )

        """
        return replace(self, **{k: v.flatten() for k, v in field_items(self)})

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
        >>> import coordinax as cx

        We can reshape a vector:

        >>> vec = cx.CartesianPos2D(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))

        >>> vec.reshape(4)
        CartesianPos2D(
            x=Quantity[...](value=f32[4], unit=Unit("m")),
            y=Quantity[...](value=f32[4], unit=Unit("m"))
        )

        """
        # TODO: enable not needing to make a full-shaped copy
        full = full_shaped(self)
        return replace(
            self,
            **{k: v.reshape(*shape, order=order) for k, v in field_items(full)},
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
        >>> import coordinax as cx

        We can get the vector as a mapping:

        >>> vec = cx.CartesianPos2D(x=Quantity([[1, 2], [3, 4]], "m"),
        ...                         y=Quantity(0, "m"))
        >>> vec.asdict()
        {'x': Quantity['length'](Array([[1., 2.], [3., 4.]], dtype=float32), unit='m'),
         'y': Quantity['length'](Array(0., dtype=float32), unit='m')}

        """
        return dict_factory(field_items(self))

    @classproperty
    @classmethod
    def components(cls) -> tuple[str, ...]:
        """Vector component names.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.CartesianPos2D.components
        ('x', 'y')
        >>> cx.SphericalPos.components
        ('r', 'theta', 'phi')
        >>> cx.RadialVel.components
        ('d_r',)

        """
        return tuple(f.name for f in fields(cls))

    @property
    def units(self) -> MappingProxyType[str, Unit]:
        """Get the units of the vector's components."""
        return MappingProxyType({k: units_of(v) for k, v in field_items(self)})

    @property
    def dtypes(self) -> MappingProxyType[str, jnp.dtype]:
        """Get the dtypes of the vector's components."""
        return MappingProxyType({k: v.dtype for k, v in field_items(self)})

    @property
    def devices(self) -> MappingProxyType[str, Device]:
        """Get the devices of the vector's components."""
        return MappingProxyType({k: v.device for k, v in field_items(self)})

    @property
    def shapes(self) -> MappingProxyType[str, tuple[int, ...]]:
        """Get the shapes of the vector's components."""
        return MappingProxyType({k: v.shape for k, v in field_items(self)})

    @property
    def sizes(self) -> MappingProxyType[str, int]:
        """Get the sizes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.CartesianPos2D.from_([1, 2], "m").sizes
        mappingproxy({'x': 1, 'y': 1})

        >>> cx.CartesianPos2D.from_([[1, 2], [1, 2]], "m").sizes
        mappingproxy({'x': 2, 'y': 2})

        """
        return MappingProxyType({k: v.size for k, v in field_items(self)})

    # ===============================================================
    # Convenience methods

    @abstractmethod
    def represent_as(self, target: type[VT], /, *args: Any, **kwargs: Any) -> VT:
        """Represent the vector as another type."""
        raise NotImplementedError  # pragma: no cover

    @dispatch
    def to_units(self, usys: Any, /) -> "AbstractVector":
        """Convert the vector to the given units.

        Parameters
        ----------
        usys : Any
            The units to convert to according to the physical type of the
            components. This is passed to [`unxt.unitsystem`][].

        Examples
        --------
        >>> import astropy.units as u
        >>> from unxt import Quantity, unitsystem
        >>> import coordinax as cx

        >>> usys = unitsystem(u.m, u.s, u.kg, u.rad)

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.to_units(usys)
        CartesianPos3D(
            x=Quantity[...](value=f32[], unit=Unit("m")),
            y=Quantity[...](value=f32[], unit=Unit("m")),
            z=Quantity[...](value=f32[], unit=Unit("m"))
        )

        """
        usys = unitsystem(usys)
        return replace(
            self,
            **{k: uconvert(usys[dimensions_of(v)], v) for k, v in field_items(self)},
        )

    @dispatch
    def to_units(
        self: "AbstractVector", usys: Mapping[Dimensions | str, Unit | str], /
    ) -> "AbstractVector":
        """Convert the vector to the given units.

        Parameters
        ----------
        usys : Mapping[Dimensions | str, Unit | str]
            The units to convert to according to the physical type of the
            components.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        We can convert a vector to the given units:

        >>> cart = cx.CartesianPos2D(x=Quantity(1, "m"), y=Quantity(2, "km"))
        >>> cart.to_units({"length": "km"})
        CartesianPos2D(
            x=Quantity[...](value=f32[], unit=Unit("km")),
            y=Quantity[...](value=f32[], unit=Unit("km"))
        )

        This also works for vectors with different units:

        >>> sph = cx.SphericalPos(r=Quantity(1, "m"), theta=Quantity(45, "deg"),
        ...                       phi=Quantity(3, "rad"))
        >>> sph.to_units({"length": "km", "angle": "deg"})
        SphericalPos(
            r=Distance(value=f32[], unit=Unit("km")),
            theta=Quantity[...](value=f32[], unit=Unit("deg")),
            phi=Quantity[...](value=f32[], unit=Unit("deg")) )

        """
        # Ensure `units_` is PT -> Unit
        units_ = {dimensions(k): v for k, v in usys.items()}
        # Convert to the given units
        return replace(
            self,
            **{k: uconvert(units_[dimensions_of(v)], v) for k, v in field_items(self)},
        )

    @dispatch
    def to_units(
        self: "AbstractVector", _: Literal[ToUnitsOptions.consistent], /
    ) -> "AbstractVector":
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
        >>> import coordinax as cx

        We can convert a vector to the given units:

        >>> cart = cx.CartesianPos2D(x=Quantity(1, "m"), y=Quantity(2, "km"))

        If all you want is to convert to consistent units, you can use
        ``"consistent"``:

        >>> cart.to_units(cx.ToUnitsOptions.consistent)
        CartesianPos2D(
            x=Quantity[...](value=f32[], unit=Unit("m")),
            y=Quantity[...](value=f32[], unit=Unit("m"))
        )

        >>> sph = cart.represent_as(cx.SphericalPos)
        >>> sph.to_units(cx.ToUnitsOptions.consistent)
        SphericalPos(
            r=Distance(value=f32[], unit=Unit("m")),
            theta=Quantity[...](value=f32[], unit=Unit("rad")),
            phi=Quantity[...](value=f32[], unit=Unit("rad"))
        )

        """
        dim2unit = {}
        units_ = {}
        for k, v in field_items(self):
            pt = dimensions_of(v)
            if pt not in dim2unit:
                dim2unit[pt] = units_of(v)
            units_[k] = dim2unit[pt]

        return replace(
            self,
            **{k: uconvert(units_[k], v) for k, v in field_items(self)},
        )

    # ===============================================================
    # Misc

    def __str__(self) -> str:
        r"""Return a string representation of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> str(vec)
        '<CartesianPos3D (x[m], y[m], z[m])\n    [1. 2. 3.]>'

        """
        cls_name = type(self).__name__
        units_ = self.units
        comps = ", ".join(f"{c}[{units_[c]}]" for c in self.components)
        vs = np.array2string(
            jnp.stack(
                tuple(v.value for v in jnp.broadcast_arrays(*field_values(self))),
                axis=-1,
            ),
            precision=3,
            prefix="    ",
        )
        return f"<{cls_name} ({comps})\n    {vs}>"


# ===============================================================
# Register additional constructors


@AbstractVector.from_._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def from_(cls: type[AbstractVector], obj: AbstractVector, /) -> AbstractVector:
    """Construct a vector from another vector.

    Parameters
    ----------
    cls : type[AbstractVector], positional-only
        The vector class.
    obj : :class:`coordinax.AbstractVector`, positional-only
        The vector to construct from.

    Examples
    --------
    >>> import coordinax as cx

    Positions:

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")

    >>> cart = cx.CartesianPos3D.from_(q)
    >>> cart
    CartesianPos3D(
      x=Quantity[...](value=f32[], unit=Unit("km")),
      y=Quantity[...](value=f32[], unit=Unit("km")),
      z=Quantity[...](value=f32[], unit=Unit("km"))
    )

    >>> cx.AbstractPos3D.from_(cart) is cart
    True

    >>> sph = cart.represent_as(cx.SphericalPos)
    >>> cx.AbstractPos3D.from_(sph) is sph
    True

    >>> cyl = cart.represent_as(cx.CylindricalPos)
    >>> cx.AbstractPos3D.from_(cyl) is cyl
    True

    Velocities:

    >>> p = cx.CartesianVel3D.from_([1, 2, 3], "km/s")

    >>> cart = cx.CartesianVel3D.from_(p)
    >>> cx.AbstractVel3D.from_(cart) is cart
    True

    >>> sph = cart.represent_as(cx.SphericalVel, q)
    >>> cx.AbstractVel3D.from_(sph) is sph
    True

    >>> cyl = cart.represent_as(cx.CylindricalVel, q)
    >>> cx.AbstractVel3D.from_(cyl) is cyl
    True

    Accelerations:

    >>> p = cx.CartesianVel3D.from_([1, 1, 1], "km/s")

    >>> cart = cx.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> cx.AbstractAcc3D.from_(cart) is cart
    True

    >>> sph = cart.represent_as(cx.SphericalAcc, p, q)
    >>> cx.AbstractAcc3D.from_(sph) is sph
    True

    >>> cyl = cart.represent_as(cx.CylindricalAcc, p, q)
    >>> cx.AbstractAcc3D.from_(cyl) is cyl
    True

    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls} from {type(obj)}."
        raise TypeError(msg)

    # Avoid copying if the types are the same. `isinstance` is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(field_items(obj)))


# ===============================================================
# Register primitives


@register(jax.lax.eq_p)  # type: ignore[misc]
def _eq_vec_vec(lhs: AbstractVector, rhs: AbstractVector, /) -> Bool[Array, "..."]:
    """Element-wise equality of two vectors."""
    return lhs == rhs
