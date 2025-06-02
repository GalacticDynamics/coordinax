"""Representation of coordinates in different systems."""

__all__ = ["AbstractVectorLike"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

import jax
import quax_blocks
import wadler_lindig as wl
from jaxtyping import DTypeLike
from plum import dispatch
from quax import ArrayValue

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items, field_values, replace

from .flags import AttrFilter
from coordinax._src.vectors.api import vconvert
from coordinax._src.vectors.utils import full_shaped

if TYPE_CHECKING:
    from typing import Self

VT = TypeVar("VT", bound="AbstractVectorLike")


class AbstractVectorLike(
    ArrayValue,
    quax_blocks.LaxBinaryOpsMixin[Any, Any],  # TODO: type annotation
    quax_blocks.LaxRoundMixin["AbstractVectorLike"],
    quax_blocks.LaxUnaryMixin[Any],
):
    """Base class for all vector-like types.

    A vector is a collection of components that can be represented in different
    coordinate systems. This class provides a common interface for all
    vector-like types, which includes vectors but also other types like
    collections of vectors that share some properties and methods.

    Methods
    -------
    vconvert
        Convert the vector(s) to another type.
        For example, a Cartesian position vector can be converted to a
        spherical position vector.
    uconvert
        Convert the vector(s) to a different unit system.
        For example, a Cartesian position vector in meters can be converted to
        kilometers.

    astype
        Cast the fields of the vector to a new dtype.
    copy
        Return a copy of the vector.
    flatten
        Flatten the fields of the vector.
    ravel
        Ravel the fields of the vector.
    reshape
        Reshape the fields of the vector.
    round
        Round the fields of the vector to a given number of decimals.
    to_device
        Move the fields of the vector to a new device.

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

    # ---------------------------------
    # Constructors

    @classmethod
    @dispatch.abstract
    def from_(
        cls: "type[AbstractVectorLike]", *args: Any, **kwargs: Any
    ) -> "AbstractVectorLike":
        """Create a vector-like object from arguments.

        Examples
        --------
        >>> import coordinax.vecs as cxv

        >>> q = cxv.CartesianPos3D.from_([1, 2, 3], "m")
        >>> print(q)
        <CartesianPos3D: (x, y, z) [m]
            [1 2 3]>

        >>> v = cxv.CartesianVel3D.from_([1, 2, 3], "m/s")
        >>> print(v)
        <CartesianVel3D: (x, y, z) [m / s]
            [1 2 3]>

        >>> space = cxv.Space.from_(q)
        >>> print(space)
        Space({
         'length': <CartesianPos3D: (x, y, z) [m]
                       [1 2 3]>
        })

        """
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Vector API

    @dispatch
    def vconvert(
        self: "AbstractVectorLike", target: type, *args: Any, **kwargs: Any
    ) -> "AbstractVectorLike":
        """Represent the vector as another type.

        This just forwards to `coordinax.vconvert`.

        Parameters
        ----------
        target : type[`coordinax.AbstractVectorLike`]
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

    # ===============================================================
    # Quantity API

    @dispatch(precedence=-1)
    def uconvert(self, *args: Any, **kwargs: Any) -> "AbstractVectorLike":
        """Convert the vector to the given units.

        This just forwards to `unxt.uconvert`, reversing the order of the
        arguments to match the `unxt` API.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.uconvert({"length": "km"})
        CartesianPos3D(
            x=Quantity(1, unit='km'), y=Quantity(2, unit='km'), z=Quantity(3, unit='km')
        )

        """
        return u.uconvert(*args, self, **kwargs)

    @dispatch
    def uconvert(self, usys: Any, /) -> "AbstractVectorLike":
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

        >>> print(vec.uconvert(usys))
        <CartesianPos3D: (x, y, z) [m]
            [1000. 2000. 3000.]>

        >>> print(vec.uconvert("galactic"))
        <CartesianPos3D: (x, y, z) [kpc]
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
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")

        >>> try: vec.materialise()
        ... except RuntimeError as e: print(e)
        Refusing to materialise `CartesianPos3D`.

        """
        msg = f"Refusing to materialise `{type(self).__name__}`."
        raise RuntimeError(msg)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector.

        Examples
        --------
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

        >>> space = cx.Space(length=vec)
        >>> space.shape
        (2, 2)

        """
        shapes = [v.shape for v in field_values(AttrFilter, self)]
        return jnp.broadcast_shapes(*shapes)

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
        >>> ns = vec.__array_namespace__()
        >>> ns
        <module 'quaxed.numpy' from ...>

        >>> ns.multiply(vec, 2)
        CartesianPos2D(x=Quantity(6, unit='m'), y=Quantity(8, unit='m'))

        """
        return jnp

    # ---------------------------------
    # comparison operators

    # TODO: use quax_blocks.LaxEqMixin
    def __eq__(self: "AbstractVectorLike", other: object) -> Any:
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

    # ---------------------------------
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
        self: "AbstractVectorLike", dtype: Any, /, **kwargs: Any
    ) -> "AbstractVectorLike":
        """Cast the vector to a new dtype.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        We can cast a vector to a new dtype:

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
        >>> vec.astype(jnp.float32)
        CartesianPos1D(x=Quantity([1., 2.], unit='m'))

        >>> jnp.astype(vec, jnp.float32)
        CartesianPos1D(x=Quantity([1., 2.], unit='m'))

        """
        return replace(
            self, **{k: v.astype(dtype, **kwargs) for k, v in field_items(self)}
        )

    @dispatch
    def astype(
        self: "AbstractVectorLike", dtypes: Mapping[str, DTypeLike], /, **kwargs: Any
    ) -> "AbstractVectorLike":
        """Cast the vector to a new dtype.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        We can cast a vector to a new dtype:

        >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
        >>> vec
        CartesianPos1D(x=Quantity([1, 2], unit='m'))

        >>> vec.astype({"x": jnp.float32})
        CartesianPos1D(x=Quantity([1., 2.], unit='m'))

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

        Examples
        --------
        We assume the following imports:

        >>> import unxt as u
        >>> import coordinax as cx

        We can reshape a vector:

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))

        >>> vec.reshape(4)
        CartesianPos2D(x=Quantity([1, 2, 3, 4], unit='m'),
                       y=Quantity([0, 0, 0, 0], unit='m'))

        """
        # TODO: enable not needing to make a full-shaped copy
        full = full_shaped(self)
        changes = {
            k: v.reshape(*shape, order=order) for k, v in field_items(AttrFilter, full)
        }
        return replace(self, **changes)

    def round(self, decimals: int = 0) -> "Self":
        """Round the components of the vector.

        Parameters
        ----------
        decimals
            The number of decimals to round to.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        We can round a vector:

        >>> vec = cx.vecs.CartesianPos2D.from_([[1.1, 2.2], [3.3, 4.4]], "m")
        >>> vec.round(0)
        CartesianPos2D(x=Quantity([1., 3.], unit='m'), y=Quantity([2., 4.], unit='m'))

        """
        changes = {k: v.round(decimals) for k, v in field_items(AttrFilter, self)}
        return replace(self, **changes)

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
        CartesianPos1D(x=Quantity([1, 2], unit='m'))

        """
        changes = {
            k: v.to_device(device)
            for k, v in field_items(self)
            if hasattr(v, "to_device")
        }
        return replace(self, **changes)

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
        """Return a string representation of the vector-like object."""
        return wl.pformat(self, vector_form=True, precision=3)
