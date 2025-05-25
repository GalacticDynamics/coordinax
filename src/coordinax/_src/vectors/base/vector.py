"""Representation of coordinates in different systems."""

__all__ = ["AbstractVector"]

from abc import abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, TypeVar

import jax
import numpy as np
import quax_blocks
import wadler_lindig as wl
from jaxtyping import DTypeLike
from plum import dispatch
from quax import ArrayValue

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items, field_values, fields, replace

from .attribute import VectorAttribute
from .flags import AttrFilter
from coordinax._src.custom_types import Unit
from coordinax._src.utils import classproperty
from coordinax._src.vectors.api import vconvert, vector
from coordinax._src.vectors.mixins import (
    AstropyRepresentationAPIMixin,
    IPythonReprMixin,
)
from coordinax._src.vectors.utils import full_shaped

if TYPE_CHECKING:
    from typing import Self

VT = TypeVar("VT", bound="AbstractVector")


class AbstractVector(
    IPythonReprMixin,
    AstropyRepresentationAPIMixin,
    quax_blocks.LaxBinaryOpsMixin[Any, Any],  # TODO: type annotation
    quax_blocks.LaxUnaryMixin[Any],
    quax_blocks.NumpyInvertMixin[Any],
    quax_blocks.LaxRoundMixin["AbstractVector"],
    quax_blocks.LaxLenMixin,
    ArrayValue,
):
    """Base class for all vector types.

    A vector is a collection of components that can be represented in different
    coordinate systems. This class provides a common interface for all vector
    types. All fields of the vector are expected to be components of the vector.

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

    # TODO: 1) a better variable name. 2) make this public (and frozen)
    #       3) a params_fields property that's better than `components`
    _AUX_FIELDS: ClassVar[tuple[str, ...]]

    def __init_subclass__(cls) -> None:
        """Determine properties of the vector class."""
        # Note that this is called before `dataclass` has had a chance to
        # process the class, so we cannot just call `fields(cls)` to get the
        # fields. Instead, we parse `cls.__annotation__` directly.

        # To find the auxiliary fields, we look for fields that are instances of
        # `VectorAttribute`. This is done by looking at the `__dict__` rather
        # than doing `getattr` since `VectorAttribute` is a descriptor and will
        # raise an error when accessed before the dataclass construction is
        # complete.
        aux = [
            k
            for k in cls.__annotations__
            if isinstance(cls.__dict__.get(k), VectorAttribute)
        ]
        cls._AUX_FIELDS = tuple(aux)

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

    # TODO: make public
    @abstractmethod
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos2D._dimensionality()
        2

        """
        raise NotImplementedError  # pragma: no cover

    @dispatch
    def vconvert(
        self: "AbstractVector", target: type, *args: Any, **kwargs: Any
    ) -> "AbstractVector":
        """Represent the vector as another type.

        This just forwards to `coordinax.vconvert`.

        Parameters
        ----------
        target : type[`coordinax.AbstractVector`]
            The type to represent the vector as.
        *args, **kwargs
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
        >>> import unxt as u
        >>> import coordinax.vecs as cxv

        Transforming a Position:

        >>> q_cart = cxv.CartesianPos3D.from_([1, 2, 3], "m")

        >>> q_sph = q_cart.vconvert(cxv.SphericalPos)
        >>> print(q_sph)
        <SphericalPos: (r[m], theta[rad], phi[rad])
            [3.742 0.641 1.107]>

        >>> q_ps = q_cart.vconvert(cxv.ProlateSpheroidalPos, Delta=u.Quantity(1.5, "m"))
        >>> print(q_ps)
        <ProlateSpheroidalPos: (mu[m2], nu[m2], phi[rad])
         Delta=Quantity(1.5, unit='m')
            [14.89   1.36   1.107]>

        >>> print((q_ps.vconvert(cxv.CartesianPos3D) - q_cart).round(3))
        <CartesianPos3D: (x, y, z) [m]
            [-0.  0.  0.]>

        Transforming a Velocity:

        >>> v_cart = cxv.CartesianVel3D.from_([1, 2, 3], "m/s")
        >>> v_sph = v_cart.vconvert(cxv.SphericalVel, q_cart)
        >>> print(v_sph)
        <SphericalVel: (r[m / s], theta[rad / s], phi[rad / s])
            [ 3.742e+00 -8.941e-08  0.000e+00]>

        Transforming an Acceleration:

        >>> a_cart = cxv.CartesianAcc3D.from_([7, 8, 9], "m/s2")
        >>> a_sph = a_cart.vconvert(cxv.SphericalAcc, v_cart, q_cart)
        >>> print(a_sph)
        <SphericalAcc: (r[m / s2], theta[rad / s2], phi[rad / s2])
            [13.363  0.767 -1.2  ]>

        """
        return vconvert(target, self, *args, **kwargs)

    # TODO: a better name
    @property
    def _auxiliary_data(self) -> dict[str, Any]:
        """Get the auxiliary data of the vector.

        This is a dictionary of auxiliary data that is not part of the vector
        itself. This is used for storing additional information about the
        vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianPos3D.from_([1, 2, 3], "m")
        >>> vec._auxiliary_data
        {}

        >>> vec = cx.vecs.ProlateSpheroidalPos(
        ...     mu=u.Quantity(3, "m2"), nu=u.Quantity(2, "m2"),
        ...     phi=u.Quantity(4, "rad"), Delta=u.Quantity(1.5, "m"))
        >>> vec._auxiliary_data
        {'Delta': Quantity(Array(1.5, dtype=float32, ...), unit='m')}

        """
        return {k: getattr(self, k) for k in self._AUX_FIELDS}

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
        usys
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
        <CartesianPos3D: (x, y, z) [m]
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

    # ===============================================================
    # Array API

    def __array_namespace__(self) -> Any:
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
    # Attributes

    # `.dtype`, `.shape`, `.size` handled by Quax
    # TODO: .device

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
        Quantity(Array([[0, 2],
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
        Quantity(Array([[0, 2],
                                  [1, 3]], dtype=int32), unit='m')

        """
        return replace(self, **{k: v.T for k, v in field_items(AttrFilter, self)})

    # ---------------------------------------------------------------
    # comparison operators

    # TODO: use quax_blocks.LaxEqMixin
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

        And positions.

        >>> q = cx.vecs.CylindricalPos(rho=u.Quantity([1.0, 2.0], "kpc"),
        ...                            phi=u.Quantity([0.0, 0.2], "rad"),
        ...                            z=u.Quantity(0.0, "kpc"))
        >>> q == q
        Array([ True,  True], dtype=bool)

        Showing the broadcasting, then element-wise comparison of two vectors:

        >>> vec1 = cx.CartesianPos3D.from_([[1, 2, 3], [1, 2, 4]], "m")
        >>> vec2 = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> jnp.equal(vec1, vec2)
        Array([ True, False], dtype=bool)

        Showing the change of representation:

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> vec1 = vec.vconvert(cx.SphericalPos)
        >>> vec2 = vec.vconvert(cx.vecs.MathSphericalPos)
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        Quick run-through of each dimensionality:

        >>> vec1 = cx.vecs.CartesianPos1D.from_([1], "m")
        >>> vec2 = cx.vecs.RadialPos.from_([1], "m")
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        >>> vec1 = cx.vecs.CartesianPos2D.from_([2, 0], "m")
        >>> vec2 = cx.vecs.PolarPos(r=u.Quantity(2, "m"), phi=u.Quantity(0, "rad"))
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        Now we show velocities and accelerations:

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
        >>> vel1.x
        Quantity(Array([1, 2], dtype=int32), unit='km / s')
        >>> jnp.equal(vel1, vel2)
        Array([ True, False], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False], dtype=bool)

        >>> acc1 = cx.vecs.CartesianAcc2D.from_([[1, 3], [2, 4]], "km/s2")
        >>> acc2 = cx.vecs.CartesianAcc2D.from_([[1, 3], [0, 4]], "km/s2")
        >>> acc1.x
        Quantity(Array([1, 2], dtype=int32), unit='km / s2')
        >>> jnp.equal(acc1, acc2)
        Array([ True, False], dtype=bool)
        >>> acc1 == acc2
        Array([ True, False], dtype=bool)

        >>> vel1 = cx.CartesianVel3D.from_([[1, 2, 3], [4, 5, 6]], "km/s")
        >>> vel2 = cx.CartesianVel3D.from_([[1, 2, 3], [4, 5, 0]], "km/s")
        >>> vel1.x
        Quantity(Array([1, 4], dtype=int32), unit='km / s')
        >>> jnp.equal(vel1, vel2)
        Array([ True, False], dtype=bool)
        >>> vel1 == vel2
        Array([ True, False], dtype=bool)

        """
        if type(other) is not type(self):
            return NotImplemented

        return jnp.equal(self, other)  # type: ignore[arg-type]

    # ---------------------------------------------------------------
    # methods

    def __complex__(self) -> NoReturn:
        """Convert a zero-dimensional object to a Python complex object."""
        raise NotImplementedError  # pragma: no cover

    # TODO: .__dlpack__, __dlpack_device__

    def __float__(self) -> NoReturn:
        """Convert a zero-dimensional object to a Python float object."""
        raise NotImplementedError  # pragma: no cover

    def __getitem__(self, index: Any) -> "Self":
        """Return a new object with the given slice applied.

        Parameters
        ----------
        index
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
        Quantity(Array([1, 2], dtype=int32), unit='m')

        """
        full = full_shaped(self)  # TODO: detect if need to make a full-shaped copy
        return replace(full, **{k: v[index] for k, v in field_items(AttrFilter, full)})

    def __index__(self) -> NoReturn:
        """Convert the vector to an integer index."""
        raise NotImplementedError  # pragma: no cover

    def __int__(self) -> NoReturn:
        """Convert the vector to an integer."""
        raise NotImplementedError  # pragma: no cover

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

    # ===============================================================
    # JAX API

    # TODO: repeat(), round(), sort(), squeeze(), swapaxes(), transpose(),
    # view() addressable_shards, at, committed, globarl_shards,
    # is_fully_addressable, is_fully_replcated, nbytes, sharding

    @dispatch
    def astype(
        self: "AbstractVector", dtype: Any, /, **kwargs: Any
    ) -> "AbstractVector":
        """Cast the vector to a new dtype.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        We can cast a vector to a new dtype:

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
        >>> vec.astype(jnp.float32)
        CartesianPos1D(x=Quantity([1. 2.], unit='m'))

        >>> jnp.astype(vec, jnp.float32)
        CartesianPos1D(x=Quantity([1. 2.], unit='m'))

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
        >>> vec
        CartesianPos1D(x=Quantity([1 2], unit='m'))

        >>> vec.astype({"x": jnp.float32})
        CartesianPos1D(x=Quantity([1. 2.], unit='m'))

        """
        return replace(
            self,
            **{
                k: (v.astype(dtypes[k], **kwargs) if k in dtypes else v)
                for k, v in field_items(self)
            },
        )

    # -------------------------------

    def copy(self) -> "Self":
        """Return a copy of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> print(vec.copy())
        <CartesianPos3D: (x, y, z) [m]
            [1 2 3]>

        """
        return replace(self)  # TODO: should .copy be called on the components?

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
        *shape
            The new shape.
        order
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
        CartesianPos2D(x=Quantity([1 2 3 4], unit='m'), y=Quantity([0 0 0 0], unit='m'))

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

    def round(self, decimals: int = 0) -> "Self":
        """Round the components of the vector.

        Parameters
        ----------
        decimals
            The number of decimals to round to.

        Returns
        -------
        AbstractVector
            The vector with the rounded components.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        We can round a vector:

        >>> vec = cx.vecs.CartesianPos2D.from_([[1.1, 2.2], [3.3, 4.4]], "m")
        >>> vec.round(0)
        CartesianPos2D(x=Quantity([1. 3.], unit='m'), y=Quantity([2. 4.], unit='m'))

        """
        return replace(
            self, **{k: v.round(decimals) for k, v in field_items(AttrFilter, self)}
        )

    def to_device(self, device: None | jax.Device = None) -> "Self":
        """Move the vector to a new device.

        Examples
        --------
        >>> from jax import devices
        >>> import unxt as u
        >>> import coordinax as cx

        We can move a vector to a new device:

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
        >>> vec.to_device(devices()[0])
        CartesianPos1D(x=Quantity([1 2], unit='m'))

        """
        changes = {
            k: v.to_device(device)
            for k, v in field_items(self)
            if hasattr(v, "to_device")
        }
        return replace(self, **changes)

    # ===============================================================
    # Convenience methods

    def asdict(
        self, *, dict_factory: Callable[[Any], Mapping[str, u.AbstractQuantity]] = dict
    ) -> Mapping[str, u.AbstractQuantity]:
        """Return the vector as a Mapping.

        Parameters
        ----------
        dict_factory
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
        {'x': Quantity(Array([[1, 2], [3, 4]], dtype=int32), unit='m'),
         'y': Quantity(Array(0, dtype=int32, ...), unit='m')}

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
        ('r',)

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

    @classproperty
    @classmethod
    def dimensions(cls) -> dict[str, u.dims.AbstractDimension]:
        """Vector physical dimensions.

        Examples
        --------
        >>> import coordinax as cx

        >>> def pprint(d):
        ...     print({k: str(v).split("/")[0] for k, v in d.items()})

        >>> pprint(cx.vecs.CartesianPos1D.dimensions)
        {'x': 'length'}

        >>> pprint(cx.vecs.CartesianVel1D.dimensions)
        {'x': 'speed'}

        >>> pprint(cx.vecs.CartesianAcc1D.dimensions)
        {'x': 'acceleration'}

        >>> pprint(cx.vecs.RadialPos.dimensions)
        {'r': 'length'}

        >>> pprint(cx.vecs.RadialVel.dimensions)
        {'r': 'speed'}

        >>> pprint(cx.vecs.RadialAcc.dimensions)
        {'r': 'acceleration'}

        >>> pprint(cx.vecs.CartesianPos2D.dimensions)
        {'x': 'length', 'y': 'length'}

        >>> pprint(cx.vecs.CartesianVel2D.dimensions)
        {'x': 'speed', 'y': 'speed'}

        >>> pprint(cx.vecs.CartesianAcc2D.dimensions)
        {'x': 'acceleration', 'y': 'acceleration'}

        >>> pprint(cx.vecs.PolarPos.dimensions)
        {'r': 'length', 'phi': 'angle'}

        >>> pprint(cx.vecs.PolarVel.dimensions)
        {'r': 'speed', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.PolarAcc.dimensions)
        {'r': 'acceleration', 'phi': 'angular acceleration'}

        >>> pprint(cx.vecs.TwoSpherePos.dimensions)
        {'theta': 'angle', 'phi': 'angle'}

        >>> pprint(cx.vecs.TwoSphereVel.dimensions)
        {'theta': 'angular frequency', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.TwoSphereAcc.dimensions)
        {'theta': 'angular acceleration', 'phi': 'angular acceleration'}

        >>> pprint(cx.vecs.CartesianPos3D.dimensions)
        {'x': 'length', 'y': 'length', 'z': 'length'}

        >>> pprint(cx.vecs.CartesianVel3D.dimensions)
        {'x': 'speed', 'y': 'speed', 'z': 'speed'}

        >>> pprint(cx.vecs.CartesianAcc3D.dimensions)
        {'x': 'acceleration', 'y': 'acceleration', 'z': 'acceleration'}

        >>> pprint(cx.vecs.CylindricalPos.dimensions)
        {'rho': 'length', 'phi': 'angle', 'z': 'length'}

        >>> pprint(cx.vecs.CylindricalVel.dimensions)
        {'rho': 'speed', 'phi': 'angular frequency', 'z': 'speed'}

        >>> pprint(cx.vecs.CylindricalAcc.dimensions)
        {'rho': 'acceleration', 'phi': 'angular acceleration', 'z': 'acceleration'}

        >>> pprint(cx.vecs.SphericalPos.dimensions)
        {'r': 'length', 'theta': 'angle', 'phi': 'angle'}

        >>> pprint(cx.vecs.SphericalVel.dimensions)
        {'r': 'speed', 'theta': 'angular frequency', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.SphericalAcc.dimensions)
        {'r': 'acceleration', 'theta': 'angular acceleration', 'phi': 'angular acceleration'}

        >>> pprint(cx.vecs.LonLatSphericalPos.dimensions)
        {'lon': 'angle', 'lat': 'angle', 'distance': 'length'}

        >>> pprint(cx.vecs.LonLatSphericalVel.dimensions)
        {'lon': 'angular frequency', 'lat': 'angular frequency', 'distance': 'speed'}

        >>> pprint(cx.vecs.LonLatSphericalAcc.dimensions)
        {'lon': 'angular acceleration', 'lat': 'angular acceleration', 'distance': 'acceleration'}

        >>> pprint(cx.vecs.MathSphericalPos.dimensions)
        {'r': 'length', 'theta': 'angle', 'phi': 'angle'}

        >>> pprint(cx.vecs.MathSphericalVel.dimensions)
        {'r': 'speed', 'theta': 'angular frequency', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.MathSphericalAcc.dimensions)
        {'r': 'acceleration', 'theta': 'angular acceleration', 'phi': 'angular acceleration'}

        >>> pprint(cx.vecs.ProlateSpheroidalPos.dimensions)
        {'mu': 'area', 'nu': 'area', 'phi': 'angle'}

        >>> pprint(cx.vecs.ProlateSpheroidalVel.dimensions)
        {'mu': 'diffusivity', 'nu': 'diffusivity', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.ProlateSpheroidalAcc.dimensions)
        {'mu': 'dose of ionizing radiation', 'nu': 'dose of ionizing radiation', 'phi': 'angular acceleration'}

        >>> cx.vecs.Cartesian3D.dimensions
        <property object at ...>

        >>> cx.vecs.FourVector.dimensions
        <property object at ...>

        >>> pprint(cx.vecs.CartesianPosND.dimensions)
        {'q': 'length'}

        >>> pprint(cx.vecs.CartesianVelND.dimensions)
        {'q': 'speed'}

        >>> pprint(cx.vecs.CartesianAccND.dimensions)
        {'q': 'acceleration'}

        >>> pprint(cx.vecs.PoincarePolarVector.dimensions)
        {'rho': 'length', 'pp_phi': 'unknown', 'z': 'length',
         'dt_rho': 'speed', 'dt_pp_phi': 'unknown', 'dt_z': 'speed'}

        """  # noqa: E501
        return {f.name: u.dimension_of(f.type) for f in fields(AttrFilter, cls)}

    @property
    def dtypes(self) -> MappingProxyType[str, jnp.dtype[Any]]:
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
    def devices(self) -> MappingProxyType[str, jax.Device]:
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
    # Wadler-Lindig

    def __pdoc__(
        self,
        *,
        vector_form: bool = False,
        short_arrays: bool = True,
        **kwargs: Any,
    ) -> wl.AbstractDoc:
        """Return the Wadler-Lindig docstring for the vector.

        Parameters
        ----------
        vector_form
            If True, return the vector form of the docstring.
        short_arrays
            If True, use short arrays for the docstring.
        **kwargs
            Additional keyword arguments to pass to the Wadler-Lindig docstring
            formatter.

        """
        if not vector_form:
            # TODO: not use private API.
            return wl._definitions._pformat_dataclass(
                self, short_arrays=short_arrays, **kwargs
            )

        # -----------------------------

        cls_name = wl.TextDoc(self.__class__.__name__)

        # make the components string
        units_ = self.units
        if len(set(units_.values())) == 1:
            cdocs = [wl.TextDoc(f"{c}") for c in self.components]
            end = wl.TextDoc(f") [{units_[self.components[0]]}]")
        else:
            cdocs = [wl.TextDoc(f"{c}[{units_[c]}]") for c in self.components]
            end = wl.TextDoc(")")
        comps_doc = wl.bracketed(
            begin=wl.TextDoc("("), docs=cdocs, sep=wl.comma, end=end, indent=4
        )

        # make the aux fields string
        if not self._AUX_FIELDS:
            aux_doc = wl.TextDoc("")
        else:
            aux_doc = (
                wl.TextDoc("\n")  # force a line break
                + wl.bracketed(
                    begin=wl.TextDoc(" "),  # indent to opening "<"
                    docs=wl.named_objs(
                        [(k, getattr(self, k)) for k in self._AUX_FIELDS],
                        short_arrays=False,
                        compact_arrays=True,
                        **kwargs,
                    ),
                    sep=wl.comma,
                    end=wl.TextDoc(""),  # no end bracket
                    indent=1,
                )
            )

        # make the values string
        # TODO: avoid casting to numpy arrays
        fvals = tuple(
            map(u.ustrip, jnp.broadcast_arrays(*field_values(AttrFilter, self)))
        )
        fvs = np.array2string(
            np.stack(fvals, axis=-1),
            precision=kwargs.pop("precision", 3),
            threshold=kwargs.pop("threshold", 1000),
            suffix=">",
        )
        vs_doc = wl.TextDoc(fvs).nest(4)

        # TODO: figure out how to avoid the hacky line breaks
        return (
            (  # header
                (wl.TextDoc("<") + cls_name + wl.TextDoc(":")).group()
                + wl.BreakDoc(" ")
                + comps_doc
            ).group()
            + aux_doc.group()  # aux fields
            + wl.TextDoc("\n")  # force a line break
            + (  # values (including end >)
                wl.TextDoc("    ") + vs_doc + wl.TextDoc(">")
            ).group()
        )

    # ===============================================================
    # Python API

    def __hash__(self) -> int:
        """Return the hash of the vector.

        This is the hash of the fields, however since jax arrays are
         not hashable this will generally raise an exception.
        Defining the `__hash__` method is required for the vector to
        be considered immutable, e.g. by `dataclasses.dataclass`.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> try:
        ...     hash(vec)
        ... except TypeError as e:
        ...     print(e)
        unhashable type: 'jaxlib...ArrayImpl'

        """
        return hash(tuple(field_items(self)))

    def __repr__(self) -> str:
        """Return a string representation of the vector.

        This uses the `eqxuinox.tree_pformat` function to format the vector,
        which internally uses the `wadler_lindig` algorithm to format the string
        representation of the vector.

        """
        return wl.pformat(
            self, vector_form=False, short_arrays=False, compact_arrays=True
        )

    def __str__(self) -> str:
        r"""Return a string representation of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        Showing a vector with only axis fields

        >>> vec1 = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> print(str(vec1))
        <CartesianPos3D: (x, y, z) [m]
            [1 2 3]>

        Showing a vector with additional attributes

        >>> vec2 = vec1.vconvert(cx.vecs.ProlateSpheroidalPos, Delta=u.Quantity(1, "m"))
        >>> print(str(vec2))
        <ProlateSpheroidalPos: (mu[m2], nu[m2], phi[rad])
         Delta=Quantity(1, unit='m')
            [14.374  0.626  1.107]>

        """
        return wl.pformat(self, vector_form=True, precision=3)
