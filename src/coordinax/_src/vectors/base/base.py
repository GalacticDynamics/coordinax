"""Representation of coordinates in different systems."""

__all__ = ["AbstractVector"]

from abc import abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

import jax
import numpy as np
from jax import Device, tree
from jaxtyping import Array, ArrayLike, Bool, DTypeLike
from plum import dispatch
from quax import ArrayValue, register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items, field_values, fields, replace
from unxt.quantity import AbstractQuantity

from .flags import AttrFilter
from .mixins import AstropyRepresentationAPIMixin, IPythonReprMixin
from coordinax._src.typing import Unit
from coordinax._src.utils import classproperty, is_any_quantity
from coordinax._src.vectors.api import vconvert, vector
from coordinax._src.vectors.utils import full_shaped

if TYPE_CHECKING:
    from typing import Self

VT = TypeVar("VT", bound="AbstractVector")


class AbstractVector(IPythonReprMixin, AstropyRepresentationAPIMixin, ArrayValue):  # type: ignore[misc]
    """Base class for all vector types.

    A vector is a collection of components that can be represented in different
    coordinate systems. This class provides a common interface for all vector
    types. All fields of the vector are expected to be components of the vector.

    """

    @abstractmethod
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos2D._dimensionality()
        2

        """
        raise NotImplementedError

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    def from_(
        cls: "type[AbstractVector]", *args: Any, **kwargs: Any
    ) -> "AbstractVector":
        """Create a vector from arguments.

        See `coordinax.vector` for more information.

        """
        return vector(cls, *args, **kwargs)

    # ===============================================================
    # Vector API

    def vconvert(self, target: type, *args: Any, **kwargs: Any) -> "AbstractVector":
        """Represent the vector as another type.

        This just forwards to `coordinax.vconvert`.

        Parameters
        ----------
        target : type[`coordinax.AbstractVector`]
            The type to represent the vector as.
        *args, **kwargs : Any
            Extra arguments. These are passed to `coordinax.vconvert` and
            might be used, depending on the dispatched method. E.g. for
            transforming an acceleration, generally the first argument is the
            velocity (`coordinax.AbstractVel`) followed by the position
            (`coordinax.AbstractPos`) at which the acceleration is defined. In
            general this is a required argument, though it is not for
            Cartesian-to-Cartesian transforms -- see
            https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for
            more information.

        Examples
        --------
        >>> import coordinax as cx

        Transforming a Position:

        >>> q_cart = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> q_sph = q_cart.vconvert(cx.SphericalPos)
        >>> q_sph
        SphericalPos( ... )
        >>> q_sph.r
        Distance(Array(3.7416575, dtype=float32), unit='m')

        Transforming a Velocity:

        >>> v_cart = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
        >>> v_sph = v_cart.vconvert(cx.SphericalVel, q_cart)
        >>> v_sph
        SphericalVel( ... )

        Transforming an Acceleration:

        >>> a_cart = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "m/s2")
        >>> a_sph = a_cart.vconvert(cx.vecs.SphericalAcc, v_cart, q_cart)
        >>> a_sph
        SphericalAcc( ... )
        >>> a_sph.d2_r
        Quantity['acceleration'](Array(13.363062, dtype=float32), unit='m / s2')

        """
        return vconvert(target, self, *args, **kwargs)

    # ===============================================================
    # Quantity API

    @dispatch(precedence=-1)
    def uconvert(self, *args: Any, **kwargs: Any) -> "AbstractVector":
        """Convert the vector to the given units."""
        return u.uconvert(*args, self, **kwargs)

    @dispatch
    def uconvert(self, usys: Any, /) -> "AbstractVector":
        """Convert the vector to the given units.

        Parameters
        ----------
        usys : Any
            The units to convert to according to the physical type of the
            components. This is passed to [`unxt.unitsystem`][].

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> usys = u.unitsystem("m", "s", "kg", "rad")

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> newvec = vec.uconvert(usys)
        >>> print(newvec)
        <CartesianPos3D (x[m], y[m], z[m])
            [1000. 2000. 3000.]>

        """
        return u.uconvert(usys, self)

    # ===============================================================
    # Quax API

    @property
    def shape(self) -> Any:
        """Get the shape of the vector's components.

        When represented as a single array, the vector has an additional
        dimension at the end for the components.

        Examples
        --------
        We assume the following imports:

        >>> import unxt as u
        >>> import coordinax as cx

        We can get the shape of a vector:

        >>> vec = cx.vecs.CartesianPos1D(x=u.Quantity([1, 2], "m"))
        >>> vec.shape
        (2,)

        >>> vec = cx.vecs.CartesianPos1D(x=u.Quantity([[1, 2], [3, 4]], "m"))
        >>> vec.shape
        (2, 2)

        ``shape`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))
        >>> vec.shape
        (2, 2)

        """
        return jnp.broadcast_shapes(*self.shapes.values())

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

    def __array_namespace__(self) -> "ArrayAPINamespace":
        """Return the array API namespace.

        Here we return the `quaxed.numpy` module, which is a drop-in replacement
        for `jax.numpy`, but allows for array-ish objects to be used in place of
        `jax` arrays. See `quax` for more information.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianPos2D.from_([3, 4], "m")
        >>> vec.__array_namespace__()
        <module 'quaxed.numpy' from ...>

        """
        return jnp

    # ---------------------------------------------------------------
    # attributes

    # Missing attributes:
    # - dtype
    # - device

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        We assume the following imports:

        >>> import unxt as u
        >>> import coordinax as cx

        We can transpose a vector:

        >>> vec = cx.CartesianPos3D(x=u.Quantity([[0, 1], [2, 3]], "m"),
        ...                         y=u.Quantity([[0, 1], [2, 3]], "m"),
        ...                         z=u.Quantity([[0, 1], [2, 3]], "m"))
        >>> vec.mT.x
        Quantity['length'](Array([[0, 2],
                                  [1, 3]], dtype=int32), unit='m')

        """
        return replace(self, **{k: v.mT for k, v in field_items(AttrFilter, self)})

    @property
    def ndim(self) -> int:
        """Number of array dimensions (axes).

        Examples
        --------
        We assume the following imports:

        >>> import unxt as u
        >>> import coordinax as cx

        We can get the number of dimensions of a vector:

        >>> vec = cx.vecs.CartesianPos2D.from_([1, 2], "m")
        >>> vec.ndim
        0

        >>> vec = cx.vecs.CartesianPos2D.from_([[1, 2], [3, 4]], "m")
        >>> vec.ndim
        1

        ``ndim`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))
        >>> vec.ndim
        2

        """
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements in the vector.

        Examples
        --------
        We assume the following imports:

        >>> import unxt as u
        >>> import coordinax as cx

        We can get the size of a vector:

        >>> vec = cx.vecs.CartesianPos2D.from_([1, 2], "m")
        >>> vec.size
        1

        >>> vec = cx.vecs.CartesianPos2D.from_([[1, 2], [3, 4]], "m")
        >>> vec.size
        2

        ``size`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))
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

        >>> import unxt as u
        >>> import coordinax as cx

        We can transpose a vector:

        >>> vec = cx.CartesianPos3D(x=u.Quantity([[0, 1], [2, 3]], "m"),
        ...                         y=u.Quantity([[0, 1], [2, 3]], "m"),
        ...                         z=u.Quantity([[0, 1], [2, 3]], "m"))
        >>> vec.T.x
        Quantity['length'](Array([[0, 2],
                                  [1, 3]], dtype=int32), unit='m')

        """
        return replace(self, **{k: v.T for k, v in field_items(AttrFilter, self)})

    # ---------------------------------------------------------------
    # arithmetic operators

    def __pos__(self) -> "AbstractVector":
        """Return the plus of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, -2, 3], "m")
        >>> print(+vec)
        <CartesianPos3D (x[m], y[m], z[m])
            [ 1 -2  3]>

        """
        return replace(self, **{k: +v for k, v in field_items(AttrFilter, self)})

    @abstractmethod
    def __neg__(self) -> "Self":
        raise NotImplementedError

    __add__ = qlax.add
    # TODO: __radd__

    __sub__ = qlax.sub
    # TODO: __rsub__

    def __mul__(self: "AbstractVector", other: Any) -> Any:
        """Multiply the vector by a scalar.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> print(vec * 2)
        <CartesianPos3D (x[m], y[m], z[m])
            [2 4 6]>

        """
        return qlax.mul(self, other)

    def __rmul__(self: "AbstractVector", other: Any) -> Any:
        """Multiply the vector by a scalar.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> print(2 * vec)
        <CartesianPos3D (x[m], y[m], z[m])
            [2 4 6]>

        """
        return qlax.mul(other, self)

    def __truediv__(self: "AbstractVector", other: Any) -> "AbstractVector":
        return qlax.div(self, other)

    # TODO: __rtruediv__

    # TODO: __floordiv__
    # TODO: __rfloordiv__

    # TODO: __mod__
    # TODO: __rmod__

    # TODO: __pow__
    # TODO: __rpow__

    # ---------------------------------------------------------------
    # array operators

    # TODO: __matmul__
    # TODO: __rmatmul__

    # ---------------------------------------------------------------
    # bitwise operators
    # TODO: handle edge cases, e.g. boolean Quantity, not in Astropy

    # TODO: __invert__
    # TODO: __and__
    # TODO: __rand__
    # TODO: __or__
    # TODO: __ror__
    # TODO: __xor__
    # TODO: __rxor__
    # TODO: __lshift__
    # TODO: __rlshift__
    # TODO: __rshift__
    # TODO: __rrshift__

    # ---------------------------------------------------------------
    # comparison operators

    # TODO: __lt__
    # TODO: __le__
    # TODO: __eq__
    # TODO: __ge__
    # TODO: __gt__
    # TODO: __ne__

    def __eq__(self: "AbstractVector", other: object) -> Any:
        """Check if the vector is equal to another object.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        We can compare non-vector objects:

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
        >>> vec == 2
        False

        Positions are covered by a separate dispatch. So here we show velocities
        and accelerations:

        >>> vel1 = cx.vecs.CartesianVel1D(u.Quantity([1, 2, 3], "km/s"))
        >>> vel2 = cx.vecs.CartesianVel1D(u.Quantity([1, 0, 3], "km/s"))
        >>> jnp.equal(vel1, vel2)
        Array([ True,  False,  True], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False,  True], dtype=bool)

        >>> acc1 = cx.vecs.CartesianAcc1D(u.Quantity([1, 2, 3], "km/s2"))
        >>> acc2 = cx.vecs.CartesianAcc1D(u.Quantity([1, 0, 3], "km/s2"))
        >>> jnp.equal(acc1, acc2)
        Array([ True,  False,  True], dtype=bool)
        >>> acc1 == acc2
        Array([ True, False,  True], dtype=bool)

        >>> vel1 = cx.vecs.RadialVel(u.Quantity([1, 2, 3], "km/s"))
        >>> vel2 = cx.vecs.RadialVel(u.Quantity([1, 0, 3], "km/s"))
        >>> jnp.equal(vel1, vel2)
        Array([ True,  False,  True], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False,  True], dtype=bool)

        >>> acc1 = cx.vecs.RadialAcc(u.Quantity([1, 2, 3], "km/s2"))
        >>> acc2 = cx.vecs.RadialAcc(u.Quantity([1, 0, 3], "km/s2"))
        >>> jnp.equal(acc1, acc2)
        Array([ True,  False,  True], dtype=bool)
        >>> acc1 == acc2
        Array([ True, False,  True], dtype=bool)

        >>> vel1 = cx.vecs.CartesianVel2D.from_([[1, 3], [2, 4]], "km/s")
        >>> vel2 = cx.vecs.CartesianVel2D.from_([[1, 3], [0, 4]], "km/s")
        >>> vel1.d_x
        Quantity['speed'](Array([1, 2], dtype=int32), unit='km / s')
        >>> jnp.equal(vel1, vel2)
        Array([ True, False], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False], dtype=bool)

        >>> acc1 = cx.vecs.CartesianAcc2D.from_([[1, 3], [2, 4]], "km/s2")
        >>> acc2 = cx.vecs.CartesianAcc2D.from_([[1, 3], [0, 4]], "km/s2")
        >>> acc1.d2_x
        Quantity['acceleration'](Array([1, 2], dtype=int32), unit='km / s2')
        >>> jnp.equal(acc1, acc2)
        Array([ True, False], dtype=bool)
        >>> acc1 == acc2
        Array([ True, False], dtype=bool)

        >>> vel1 = cx.CartesianVel3D.from_([[1, 2, 3], [4, 5, 6]], "km/s")
        >>> vel2 = cx.CartesianVel3D.from_([[1, 2, 3], [4, 5, 0]], "km/s")
        >>> vel1.d_x
        Quantity['speed'](Array([1, 4], dtype=int32), unit='km / s')
        >>> jnp.equal(vel1, vel2)
        Array([ True, False], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False], dtype=bool)

        >>> q = cx.vecs.CylindricalPos(rho=u.Quantity([1.0, 2.0], "kpc"),
        ...                            phi=u.Quantity([0.0, 0.2], "rad"),
        ...                            z=u.Quantity(0.0, "kpc"))
        >>> q == q
        Array([ True,  True], dtype=bool)

        """
        if type(other) is not type(self):
            return NotImplemented

        # Map the equality over the leaves, which are Quantities.
        # The equality should likewise be mapped over the Quantities.
        comp_tree = tree.map(
            jnp.equal,
            tree.leaves(self, is_leaf=is_any_quantity),
            tree.leaves(other, is_leaf=is_any_quantity),
            is_leaf=is_any_quantity,
        )

        # Reduce the equality over the leaves.
        return jax.tree.reduce(jnp.logical_and, comp_tree)

    # ---------------------------------------------------------------
    # methods

    def __abs__(self) -> u.Quantity:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianPos2D.from_([3, 4], "m")
        >>> abs(vec)
        Quantity['length'](Array(5., dtype=float32), unit='m')

        """
        return self.norm()

    # TODO: __bool__
    # TODO: __complex__
    # TODO: __dlpack__
    # TODO: __dlpack_device__
    # TODO: __float__

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

        >>> import unxt as u
        >>> import coordinax as cx

        We can slice a vector:

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))
        >>> vec[0].x
        Quantity['length'](Array([1, 2], dtype=int32), unit='m')

        """
        full = full_shaped(self)  # TODO: detect if need to make a full-shaped copy
        return replace(full, **{k: v[index] for k, v in field_items(AttrFilter, full)})

    # TODO: __index__
    # TODO: __int__

    def __setitem__(self, k: Any, v: Any) -> NoReturn:
        """Fail to set an item in the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        We can't set an item in a vector:

        >>> vec = cx.vecs.CartesianPos2D.from_([[1, 2], [3, 4]], "m")
        >>> try: vec[0] = u.Quantity(1, "m")
        ... except TypeError as e: print(e)
        CartesianPos2D is immutable.

        """
        msg = f"{type(self).__name__} is immutable."
        raise TypeError(msg)

    def to_device(self, device: None | Device = None) -> "Self":
        """Move the vector to a new device.

        Examples
        --------
        >>> from jax import devices
        >>> import unxt as u
        >>> import coordinax as cx

        We can move a vector to a new device:

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
        >>> vec.to_device(devices()[0])
        CartesianPos1D(x=Quantity[PhysicalType('length')](value=i32[2], unit=Unit("m")))

        """
        return replace(self, **{k: v.to_device(device) for k, v in field_items(self)})

    # -------------------------------

    @dispatch
    def astype(
        self: "AbstractVector", dtype: Any, /, **kwargs: Any
    ) -> "AbstractVector":
        """Cast the vector to a new dtype.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        We can cast a vector to a new dtype:

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
        >>> vec.astype(jnp.float32)
        CartesianPos1D(x=Quantity[...](value=f32[2], unit=Unit("m")))

        >>> import quaxed.numpy as jnp
        >>> jnp.astype(vec, jnp.float32)
        CartesianPos1D(x=Quantity[...](value=f32[2], unit=Unit("m")))

        """
        return replace(
            self, **{k: v.astype(dtype, **kwargs) for k, v in field_items(self)}
        )

    @dispatch
    def astype(
        self: "AbstractVector", dtypes: Mapping[str, DTypeLike], /, **kwargs: Any
    ) -> "AbstractVector":
        """Cast the vector to a new dtype.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        We can cast a vector to a new dtype:

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
        >>> vec.astype({"x": jnp.float32})
        CartesianPos1D(x=Quantity[PhysicalType('length')](value=f32[2], unit=Unit("m")))

        """
        return replace(
            self,
            **{
                k: (v.astype(dtypes[k], **kwargs) if k in dtypes else v)
                for k, v in field_items(self)
            },
        )

    # ===============================================================
    # JAX API

    def __len__(self) -> int:
        """Return the length of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        Scalar vectors have length 0:

        >>> vec = cx.vecs.CartesianPos1D.from_([1], "m")
        >>> len(vec)
        0

        Vectors with certain lengths:

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1], "m"))
        >>> len(vec)
        1

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
        >>> len(vec)
        2

        """
        return self.shape[0] if self.ndim > 0 else 0

    def flatten(self) -> "Self":
        """Flatten the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))
        >>> vec.shape
        (2, 2)

        >>> vec.flatten().shape
        (4,)

        """
        return replace(
            self, **{k: v.flatten() for k, v in field_items(AttrFilter, self)}
        )

    def ravel(self) -> "Self":
        """Ravel the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))
        >>> vec.shape
        (2, 2)

        >>> vec.ravel().shape
        (4,)

        """
        return replace(self, **{k: v.ravel() for k, v in field_items(AttrFilter, self)})

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

        >>> import unxt as u
        >>> import coordinax as cx

        We can reshape a vector:

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))

        >>> vec.reshape(4)
        CartesianPos2D(
            x=Quantity[...](value=i32[4], unit=Unit("m")),
            y=Quantity[...](value=...i32[4], unit=Unit("m"))
        )

        """
        # TODO: enable not needing to make a full-shaped copy
        full = full_shaped(self)
        return replace(
            self,
            **{
                k: v.reshape(*shape, order=order)
                for k, v in field_items(AttrFilter, full)
            },
        )

    # ===============================================================
    # Collection

    def asdict(
        self, *, dict_factory: Callable[[Any], Mapping[str, AbstractQuantity]] = dict
    ) -> Mapping[str, AbstractQuantity]:
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

        >>> import unxt as u
        >>> import coordinax as cx

        We can get the vector as a mapping:

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))
        >>> vec.asdict()
        {'x': Quantity['length'](Array([[1, 2], [3, 4]], dtype=int32), unit='m'),
         'y': Quantity['length'](Array(0, dtype=int32, ...), unit='m')}

        """
        return dict_factory(field_items(self))

    @classproperty
    @classmethod
    def components(cls) -> tuple[str, ...]:
        """Vector component names.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos2D.components
        ('x', 'y')
        >>> cx.SphericalPos.components
        ('r', 'theta', 'phi')
        >>> cx.vecs.RadialVel.components
        ('d_r',)

        """
        return tuple(f.name for f in fields(AttrFilter, cls))

    @property
    def units(self) -> MappingProxyType[str, Unit]:
        """Get the units of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.units
        mappingproxy({'x': Unit("km"), 'y': Unit("km"), 'z': Unit("km")})

        """
        return MappingProxyType(
            {k: u.unit_of(v) for k, v in field_items(AttrFilter, self)}
        )

    @property
    def dtypes(self) -> MappingProxyType[str, jnp.dtype]:
        """Get the dtypes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.dtypes
        mappingproxy({'x': dtype('int32'), 'y': dtype('int32'),
                      'z': dtype('int32')})

        """
        return MappingProxyType({k: v.dtype for k, v in field_items(AttrFilter, self)})

    @property
    def devices(self) -> MappingProxyType[str, Device]:
        """Get the devices of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.devices
        mappingproxy({'x': CpuDevice(id=0), 'y': CpuDevice(id=0),
                      'z': CpuDevice(id=0)})

        """
        return MappingProxyType({k: v.device for k, v in field_items(AttrFilter, self)})

    @property
    def shapes(self) -> MappingProxyType[str, tuple[int, ...]]:
        """Get the shapes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.shapes
        mappingproxy({'x': (), 'y': (), 'z': ()})

        """
        return MappingProxyType({k: v.shape for k, v in field_items(AttrFilter, self)})

    @property
    def sizes(self) -> MappingProxyType[str, int]:
        """Get the sizes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos2D.from_([1, 2], "m").sizes
        mappingproxy({'x': 1, 'y': 1})

        >>> cx.vecs.CartesianPos2D.from_([[1, 2], [1, 2]], "m").sizes
        mappingproxy({'x': 2, 'y': 2})

        """
        return MappingProxyType({k: v.size for k, v in field_items(AttrFilter, self)})

    # ===============================================================
    # Python API

    def _str_repr_(self, *, precision: int) -> str:
        cls_name = type(self).__name__
        units_ = self.units
        # make the components string
        comps = ", ".join(f"{c}[{units_[c]}]" for c in self.components)
        # make the values string
        # TODO: add the VectorAttr, which are filtered out.
        fvals = field_values(AttrFilter, self)
        fvstack = jnp.stack(
            tuple(map(u.ustrip, jnp.broadcast_arrays(*fvals))),
            axis=-1,
        )
        vs = np.array2string(np.array(fvstack), precision=precision, prefix="    ")
        # return the string
        return f"<{cls_name} ({comps})\n    {vs}>"

    def __str__(self) -> str:
        r"""Return a string representation of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        Showing a vector with only axis fields

        >>> vec1 = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> print(str(vec1))
        <CartesianPos3D (x[m], y[m], z[m])
            [1 2 3]>

        Showing a vector with additional attributes

        >>> vec2 = vec1.vconvert(cx.vecs.ProlateSpheroidalPos, Delta=u.Quantity(1, "m"))
        >>> print(str(vec2))
        <ProlateSpheroidalPos (mu[m2], nu[m2], phi[rad])
            [14.374  0.626  1.107]>

        """
        return self._str_repr_(precision=3)


# ===============================================================
# Register additional constructors


@dispatch
def vector(obj: AbstractVector, /) -> AbstractVector:
    """Construct a vector from a vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> cx.vector(cart) is cart
    True

    """
    return obj


@dispatch
def vector(cls: type[AbstractVector], obj: Mapping[str, Any], /) -> AbstractVector:
    """Construct a vector from a mapping.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = {"x": u.Quantity(1, "m"), "y": u.Quantity(2, "m"),
    ...       "z": u.Quantity(3, "m")}
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1 2 3]>

    >>> xs = {"x": u.Quantity([1, 2], "m"), "y": u.Quantity([3, 4], "m"),
    ...       "z": u.Quantity([5, 6], "m")}
    >>> vec = cx.CartesianPos3D.from_(xs)
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [[1 3 5]
        [2 4 6]]>

    """
    return cls(**obj)


@dispatch
def vector(cls: type[AbstractVector], obj: AbstractQuantity, /) -> AbstractVector:
    """Construct a vector from a quantity.

    This will fail for most non-position vectors, except Cartesian vectors,
    since they generally do not have the same dimensions, nor can be converted
    from a Cartesian vector without additional information.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Mismatch:

    >>> try: cx.vecs.CartesianPos1D.from_(u.Quantity([1, 2, 3], "m"))
    ... except ValueError as e: print(e)
    Cannot construct <class 'coordinax...CartesianPos1D'> from 3 components.

    Pos 1D:

    >>> cx.vecs.CartesianPos1D.from_(u.Quantity(1, "meter"))
    CartesianPos1D( x=Quantity[...](value=...i32[], unit=Unit("m")) )

    >>> cx.vecs.CartesianPos1D.from_(u.Quantity([1], "meter"))
    CartesianPos1D(x=Quantity[...](value=i32[], unit=Unit("m")))

    >>> cx.vecs.CartesianPos1D.from_(cx.Distance(1, "meter"))
    CartesianPos1D( x=Quantity[...](value=...i32[], unit=Unit("m")) )

    >>> cx.vecs.RadialPos.from_(u.Quantity(1, "meter"))
    RadialPos(r=Distance(value=...i32[], unit=Unit("m")))

    >>> cx.vecs.RadialPos.from_(u.Quantity([1], "meter"))
    RadialPos(r=Distance(value=...i32[], unit=Unit("m")))

    Vel 1D:

    >>> cx.vecs.CartesianVel1D.from_(u.Quantity(1, "m/s"))
    CartesianVel1D( d_x=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )

    >>> cx.vecs.CartesianVel1D.from_(u.Quantity([1], "m/s"))
    CartesianVel1D( d_x=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    >>> cx.vecs.RadialVel.from_(u.Quantity(1, "m/s"))
    RadialVel( d_r=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )

    >>> cx.vecs.RadialVel.from_(u.Quantity([1], "m/s"))
    RadialVel( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    Acc 1D:

    >>> cx.vecs.CartesianAcc1D.from_(u.Quantity(1, "m/s2"))
    CartesianAcc1D( d2_x=... )

    >>> cx.vecs.CartesianAcc1D.from_(u.Quantity([1], "m/s2"))
    CartesianAcc1D( d2_x=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    >>> cx.vecs.RadialAcc.from_(u.Quantity(1, "m/s2"))
    RadialAcc( d2_r=... )

    >>> cx.vecs.RadialAcc.from_(u.Quantity([1], "m/s2"))
    RadialAcc( d2_r=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    Pos 2D:

    >>> vec = cx.vecs.CartesianPos2D.from_(u.Quantity([1, 2], "m"))
    >>> vec
    CartesianPos2D(
        x=Quantity[...](value=i32[], unit=Unit("m")),
        y=Quantity[...](value=i32[], unit=Unit("m"))
    )

    Vel 2D:

    >>> vec = cx.vecs.CartesianVel2D.from_(u.Quantity([1, 2], "m/s"))
    >>> print(vec)
    <CartesianVel2D (d_x[m / s], d_y[m / s])
        [1 2]>

    Acc 2D:

    >>> vec = cx.vecs.CartesianAcc2D.from_(u.Quantity([1, 2], "m/s2"))
    >>> print(vec)
    <CartesianAcc2D (d2_x[m / s2], d2_y[m / s2])
        [1 2]>

    Pos 3D:

    >>> vec = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1 2 3]>

    Vel 3D:

    >>> vec = cx.CartesianVel3D.from_(u.Quantity([1, 2, 3], "m/s"))
    >>> print(vec)
    <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
        [1 2 3]>

    Acc 3D:

    >>> vec = cx.vecs.CartesianAcc3D.from_(u.Quantity([1, 2, 3], "m/s2"))
    >>> print(vec)
    <CartesianAcc3D (d2_x[m / s2], d2_y[m / s2], d2_z[m / s2])
        [1 2 3]>

    Generic 3D:

    >>> vec = cx.vecs.CartesianGeneric3D.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <CartesianGeneric3D (x[m], y[m], z[m])
        [1 2 3]>

    """
    # Ensure the object is at least 1D
    obj = jnp.atleast_1d(obj)

    # Check the dimensions
    if obj.shape[-1] != cls._dimensionality():
        msg = f"Cannot construct {cls} from {obj.shape[-1]} components."
        raise ValueError(msg)

    # Map the components
    comps = {k: obj[..., i] for i, k in enumerate(cls.components)}

    # Construct the vector from the mapping
    return cls.from_(comps)


@dispatch
def vector(
    cls: type[AbstractVector], obj: ArrayLike | list[Any], unit: Unit | str, /
) -> AbstractVector:
    """Construct a vector from an array and unit.

    The ``ArrayLike[Any, (*#batch, N), "..."]`` is expected to have the
    components as the last dimension.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "meter")
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1 2 3]>

    >>> xs = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> vec = cx.CartesianPos3D.from_(xs, "meter")
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [[1 2 3]
        [4 5 6]]>

    """
    obj = u.Quantity.from_(jnp.asarray(obj), unit)
    return cls.from_(obj)  # re-dispatch


@dispatch
def vector(cls: type[AbstractVector], obj: AbstractVector, /) -> AbstractVector:
    """Construct a vector from another vector.

    Raises
    ------
    TypeError
        If the object is not an instance of the vector class.

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
    >>> print(cart)
    <CartesianPos3D (x[km], y[km], z[km])
        [1 2 3]>

    >>> cx.vecs.AbstractPos3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.SphericalPos)
    >>> cx.vecs.AbstractPos3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalPos)
    >>> cx.vecs.AbstractPos3D.from_(cyl) is cyl
    True

    Velocities:

    >>> p = cx.CartesianVel3D.from_([1, 2, 3], "km/s")

    >>> cart = cx.CartesianVel3D.from_(p)
    >>> cx.vecs.AbstractVel3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.SphericalVel, q)
    >>> cx.vecs.AbstractVel3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalVel, q)
    >>> cx.vecs.AbstractVel3D.from_(cyl) is cyl
    True

    Accelerations:

    >>> p = cx.CartesianVel3D.from_([1, 1, 1], "km/s")

    >>> cart = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> cx.vecs.AbstractAcc3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.vecs.SphericalAcc, p, q)
    >>> cx.vecs.AbstractAcc3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalAcc, p, q)
    >>> cx.vecs.AbstractAcc3D.from_(cyl) is cyl
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
def eq_vec_vec(lhs: AbstractVector, rhs: AbstractVector, /) -> Bool[Array, "..."]:
    """Element-wise equality of two vectors."""
    return lhs == rhs
