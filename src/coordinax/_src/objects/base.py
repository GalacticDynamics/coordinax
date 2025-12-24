"""Vector."""

__all__ = ("AbstractVectorLike", "is_vectorlike")

from abc import abstractmethod

from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast
from typing_extensions import TypeIs

import equinox as eqx
import jax
import jax.tree as jtu
import plum
import quax_blocks
import wadler_lindig as wl
from quax import ArrayValue

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items, replace

if TYPE_CHECKING:
    from typing import Self

Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
V = TypeVar("V")


class AbstractVectorLike(
    ArrayValue,
    quax_blocks.LaxBinaryOpsMixin[Any, Any],  # TODO: type annotation
    quax_blocks.LaxRoundMixin["AbstractVectorLike"],
    quax_blocks.LaxUnaryMixin[Any],
):
    """Base class for all vector-like objects.

    Methods
    -------
    from_
        Create an instance from the arguments.
    vconvert
        Convert the vector(s) to another type.
        For example, a Cartesian position vector can be converted to a
        spherical position vector.

    """

    # ---------------------------------
    # Constructors

    @classmethod
    @plum.dispatch.abstract
    def from_(
        cls: "type[AbstractVectorLike]", *args: Any, **kwargs: Any
    ) -> "AbstractVectorLike":
        """Create a vector-like object from arguments."""
        raise NotImplementedError  # pragma: no cover

    # # ===============================================================
    # # Vector API

    # @plum.dispatch
    # def vconvert(
    #     self,
    #     target: AbstractChart,  # type: ignore[type-arg]
    #     /,
    #     *args: Any,
    #     **kwargs: Any,
    # ) -> "AbstractVectorLike":
    #     """Represent the vector as another type.

    #     This just forwards to `coordinax.vconvert`.

    #     Parameters
    #     ----------
    #     target : type[`coordinax.AbstractVectorLike`]
    #         The type to represent the vector as.
    #     *args, **kwargs
    #         Extra arguments. These are passed to `coordinax.vconvert` and
    #         might be used, depending on the dispatched method.

    #     """
    #     return cxapi.vconvert(target, self, *args, **kwargs)

    # ===============================================================
    # Quantity API

    @plum.dispatch(precedence=-1)
    def uconvert(self, *args: Any, **kwargs: Any) -> "AbstractVectorLike":
        """Convert the vector to the given units.

        This just forwards to `unxt.uconvert`, reversing the order of the
        arguments to match the `unxt` API.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.Vector.from_([1, 2, 3], "km")
        >>> print(vec.uconvert({"length": "km"}))
        <Vector: chart=Cart3D, role=Pos (x, y, z) [km]
            [1 2 3]>

        """
        return u.uconvert(*args, self, **kwargs)

    @plum.dispatch
    def uconvert(self, usys: u.AbstractUnitSystem, /) -> "AbstractVectorLike":
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

        >>> vec = cx.Vector.from_([1, 2, 3], "km")

        >>> print(vec.uconvert(usys))
        <Vector: chart=Cart3D, role=Pos (x, y, z) [m]
            [1000. 2000. 3000.]>

        >>> print(vec.uconvert("galactic"))
        <Vector: chart=Cart3D, role=Pos (x, y, z) [kpc]
            [3.241e-17 6.482e-17 9.722e-17]>

        """
        return u.uconvert(usys, self)

    # ===============================================================
    # Quax API

    def materialise(self) -> NoReturn:
        """Materialise the vector for `quax`.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.Vector.from_([1, 2, 3], "m")

        >>> try: vec.materialise()
        ... except RuntimeError as e: print(e)
        Refusing to materialise `Vector`.

        """
        msg = f"Refusing to materialise `{type(self).__name__}`."
        raise RuntimeError(msg)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        raise NotImplementedError  # pragma: no cover

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
        >>> vec = cx.Vector.from_([3, 4], "m")
        >>> ns = vec.__array_namespace__()
        >>> ns
        <module 'quaxed.numpy' from ...>

        >>> _ = ns.multiply(vec, 2)

        """
        return jnp

    # ---------------------------------
    # comparison operators

    # TODO: use quax_blocks.LaxEqMixin
    def __eq__(self: "AbstractVectorLike", other: object) -> Any:
        """Check if the vector is equal to another object."""
        if type(other) is not type(self):
            return NotImplemented

        return jnp.equal(self, cast("AbstractVectorLike", other))

    # ---------------------------------
    # methods

    def __complex__(self) -> NoReturn:
        """Convert a zero-dimensional object to a Python complex object."""
        raise NotImplementedError  # pragma: no cover

    # TODO: .__dlpack__, __dlpack_device__

    def __float__(self) -> NoReturn:
        """Convert a zero-dimensional object to a Python float object."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def __getitem__(self, index: Any) -> "Self":
        """Return a new object with the given slice applied.

        Parameters
        ----------
        index
            The slice to apply.

        """
        raise NotImplementedError  # pragma: no cover

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

        >>> vec = cx.Vector.from_(u.Q([[1, 2], [3, 4]], "m"))
        >>> try: vec[0] = u.Q(1, "m")
        ... except TypeError as e: print(e)
        Vector is immutable.

        """
        msg = f"{type(self).__name__} is immutable."
        raise TypeError(msg)

    # ===============================================================
    # JAX API

    # TODO: repeat(), round(), sort(), squeeze(), swapaxes(), transpose(),
    # view() addressable_shards, at, committed, globarl_shards,
    # is_fully_addressable, is_fully_replcated, nbytes, sharding

    @plum.dispatch
    def astype(
        self: "AbstractVectorLike", dtype: Any, /, **kwargs: Any
    ) -> "AbstractVectorLike":
        """Cast the vector to a new dtype.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        We can cast a vector to a new dtype:

        >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
        >>> print(vec.astype(jnp.float32))
        <Vector: chart=Cart3D, role=Pos (x, y, z) [m]
            [1. 2. 3.]>

        >>> print(jnp.astype(vec, jnp.float32))
        <Vector: chart=Cart3D, role=Pos (x, y, z) [m]
            [1. 2. 3.]>

        """
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "astype"))
        dynamic = jtu.map(lambda x: x.astype(dtype, **kwargs), dynamic)
        return eqx.combine(dynamic, static)

    # -------------------------------

    def copy(self) -> "Self":
        """Return a copy of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.Vector.from_([1, 2, 3], "m")
        >>> print(vec.copy())
        <Vector: chart=Cart3D, role=Pos (x, y, z) [m]
            [1 2 3]>

        """
        return replace(self)

    def flatten(self) -> "Self":
        """Flatten the vector."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "flatten"))
        dynamic = jtu.map(lambda x: x.flatten(), dynamic)
        return eqx.combine(dynamic, static)

    def ravel(self) -> "Self":
        """Return a flattened vector."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "ravel"))
        dynamic = jtu.map(lambda x: x.ravel(), dynamic)
        return eqx.combine(dynamic, static)

    def reshape(self, *shape: int) -> "Self":
        """Return a reshaped vector."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "reshape"))
        dynamic = jtu.map(lambda x: x.reshape(*shape), dynamic)
        return eqx.combine(dynamic, static)

    def round(self, decimals: int = 0) -> "Self":
        """Return a rounded vector."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "round"))
        dynamic = jtu.map(lambda x: x.round(decimals), dynamic)
        return eqx.combine(dynamic, static)

    def to_device(self, device: None | jax.Device = None) -> "Self":
        """Move the vector to a new device."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "to_device"))
        dynamic = jtu.map(lambda x: x.to_device(device), dynamic)
        return eqx.combine(dynamic, static)

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
        >>> vec = cx.Vector.from_([1, 2, 3], "m")
        >>> try:
        ...     hash(vec)
        ... except TypeError as e:
        ...     print(e)
        unhashable type: 'jaxlib...ArrayImpl'

        """
        return hash(tuple(field_items(self)))

    def __repr__(self) -> str:
        """Return a string representation of the vector.

        This uses the `equinox.tree_pformat` function to format the vector,
        which internally uses the `wadler_lindig` algorithm to format the string
        representation of the vector.

        """
        return wl.pformat(self, vector_form=False, short_arrays="compact")

    def __str__(self) -> str:
        """Return a string representation of the vector-like object."""
        return wl.pformat(self, vector_form=True, precision=3)


def is_vectorlike(obj: Any, /) -> TypeIs[AbstractVectorLike]:
    """Check if the object is a `AbstractVectorLike` object.

    Examples
    --------
    >>> import coordinax as cx
    >>> from coordinax._src.vectors.base import is_vectorlike

    >>> vec = cx.Vector.from_([1, 2, 3], "m")
    >>> is_vectorlike(vec)
    True

    >>> space = cx.FiberPoint.from_(vec)
    >>> is_vectorlike(space)
    True

    >>> import coordinax.frames as cxf
    >>> coord = cxf.Coordinate.from_(vec, cxf.ICRS())
    >>> is_vectorlike(coord)
    True

    >>> is_vectorlike(42)
    False

    """
    return isinstance(obj, AbstractVectorLike)
