"""Vector."""

__all__ = ("KinematicSpace",)

from collections.abc import ItemsView, Iterable, KeysView, Mapping, ValuesView
from typing import Any, Generic, cast, final
from typing_extensions import TypeVar

import equinox as eqx
import jax
import plum

import quaxed.numpy as jnp
from xmmutablemap import ImmutableMap

from . import api, representations as r
from .base import AbstractVectorLike
from .custom_types import Dimension, DimensionLike
from .utils import can_broadcast_shapes, dimension_name
from .vector import Vector

Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
V = TypeVar("V")
PosT = TypeVar("PosT", bound=r.AbstractPos, default=r.AbstractPos)


@final
class KinematicSpace(
    AbstractVectorLike,
    ImmutableMap[str, Vector],  # type: ignore[misc]
    Generic[PosT],
):
    """A collection of vectors based on a position.

    Parameters
    ----------
    *args, **kwargs
        See input to `dict` for the input data.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.nvecs as cxv

    >>> x = cxv.Vector.from_(u.Q([1, 2, 3], "km"), cxv.CartPos3D)
    >>> v = cxv.Vector.from_(u.Q([4, 5, 6], "km/s"), cxv.CartVel3D)
    >>> a = cxv.Vector.from_(u.Q([7, 8, 9], "km/s^2"), cxv.CartAcc3D)
    All the vectors can be brought together into a space:

    >>> space = cx.KinematicSpace(length=x, speed=v, acceleration=a)
    >>> print(space)
    KinematicSpace({
       'length': <CartesianPos3D: (x, y, z) [km]
           [1 2 3]>,
       'speed': <CartesianVel3D: (x, y, z) [km / s]
           [4 5 6]>,
       'acceleration': <CartesianAcc3D: (x, y, z) [km / s2]
           [7 8 9]>
    })

    The vectors can initialized from `unxt.Quantity` objects and can have
    (brodcastable) batch shapes:

    >>> w = cx.KinematicSpace(
    ...     length=u.Q([[8.5, 0, 0], [10, 0, 0]], "kpc"),
    ...     speed=u.Q([0, 200, 0], "km/s"))
    >>> print(w)
    KinematicSpace({
       'length': <CartesianPos3D: (x, y, z) [kpc]
           [[ 8.5  0.   0. ]
            [10.   0.   0. ]]>,
       'speed': <CartesianVel3D: (x, y, z) [km / s]
           [  0 200   0]>
    })

    The vectors can then be accessed by key:

    >>> space["length"]
    CartesianPos3D( ... )

    The space can also be converted to different representations. If the
    conversion is on the primary vector, the other vectors will be
    correspondingly converted.

    >>> space.vconvert(cx.SphericalPos)
    KinematicSpace({
        'length': SphericalPos( ... ),
        'speed': SphericalVel( ... ),
        'acceleration': SphericalAcc( ... )
    })

    >>> cx.vconvert(cx.SphericalPos, space)
    KinematicSpace({
        'length': SphericalPos( ... ),
        'speed': SphericalVel( ... ),
        'acceleration': SphericalAcc( ... )
    })

    Actions on the space are done on the contained vectors.

    >>> w = cx.KinematicSpace(
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

    >>> space = cx.KinematicSpace.from_({"length": u.Q([1, 2, 3], "km"),
    ...                         "speed": u.Q([4, 5, 6], "km/s")})
    >>> print(space)
    KinematicSpace({
       'length': <CartesianPos3D: (x, y, z) [km]
           [1 2 3]>,
       'speed': <CartesianVel3D: (x, y, z) [km / s]
           [4 5 6]>
    })

    """

    # TODO: https://peps.python.org/pep-0728/#the-extra-items-class-parameter
    _data: dict[str, Vector[Any, Any]] = eqx.field(repr=False)

    def __init__(  # pylint: disable=super-init-not-called  # TODO: resolve this
        self,
        /,
        *args: Mapping[DimensionLike, Any]
        | tuple[DimensionLike, Any]
        | Iterable[tuple[DimensionLike, Any]],
        **kwargs: Any,
    ) -> None:
        # Fast-path for the common case of a single dict input.
        if len(args) == 1 and isinstance(args[0], Mapping) and not kwargs:
            self._data = dict(args[0])
            return

        # Consolidate the inputs into a single dict, then process keys & values.
        raw = dict(*args, **kwargs)  # process the input data

        # Process the keys
        keys = [dimension_name(k) for k in raw]
        # Convert the values to vectors
        values = [api.vector(v) for v in raw.values()]

        # Check that the shapes are broadcastable
        values = eqx.error_if(
            values,
            not can_broadcast_shapes([v.shape for v in values]),
            "vector shapes are not broadcastable.",
        )

        self._data = dict(zip(keys, values, strict=True))

    def q(self) -> PosT:
        """Get the position vector of the space."""
        return cast("PosT", self._data["length"])

    # ===============================================================
    # Mapping API

    @plum.dispatch
    def __getitem__(self, key: Any, /) -> Any:
        """Get the vector by key.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cxv
        >>> w = cxv.KinematicSpace(
        ...     length=u.Q([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=u.Q([[[1, 2, 3], [4, 5, 6]]], "m/s"))

        By number:

        >>> w[0]
        KinematicSpace({
            'length': CartesianPos3D(
                x=Quantity([1, 4], unit='m'),
                y=Quantity([2, 5], unit='m'),
                z=Quantity([3, 6], unit='m')
            ),
            'speed': CartesianVel3D(
                x=Quantity([1, 4], unit='m / s'),
                y=Quantity([2, 5], unit='m / s'),
                z=Quantity([3, 6], unit='m / s')
            )
        })

        By slice:

        >>> w[1:]
        KinematicSpace({
            'length': CartesianPos3D(
                x=Quantity([], unit='m'), y=Quantity([], unit='m'),
                z=Quantity([], unit='m')
            ),
            'speed': CartesianVel3D(
                x=Quantity([], unit='m / s'),
                y=Quantity([], unit='m / s'),
                z=Quantity([], unit='m / s')
            )
        })

        By Ellipsis:

        >>> w[...]
        KinematicSpace({
            'length': CartesianPos3D(
                x=Quantity([[1, 4]], unit='m'),
                y=Quantity([[2, 5]], unit='m'),
                z=Quantity([[3, 6]], unit='m')
            ),
            'speed': CartesianVel3D(
                x=Quantity([[1, 4]], unit='m / s'),
                y=Quantity([[2, 5]], unit='m / s'),
                z=Quantity([[3, 6]], unit='m / s')
            )
        })

        By tuple[int, ...]:

        >>> w[(0, 1)]
        KinematicSpace({
            'length': CartesianPos3D(
                x=Quantity(4, unit='m'), y=Quantity(5, unit='m'),
                z=Quantity(6, unit='m')
            ),
            'speed': CartesianVel3D(
                x=Quantity(4, unit='m / s'),
                y=Quantity(5, unit='m / s'),
                z=Quantity(6, unit='m / s')
            )
        })

        This also supports numpy index arrays. But this example section
        highlights core python indexing.

        """
        return KinematicSpace(**{k: v[key] for k, v in self.items()})

    @plum.dispatch
    def __getitem__(self, key: str | Dimension) -> Any:
        """Get the vector by key.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> w = cx.KinematicSpace(
        ...     length=u.Q([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=u.Q([[[1, 2, 3], [4, 5, 6]]], "m/s"))
        By string key (which is the dimension name):

        >>> print(w["length"])
        <CartesianPos3D: (x, y, z) [m]
            [[[1 2 3]
              [4 5 6]]]>

        By the actual dimension object:

        >>> import unxt as u
        >>> print(w[u.dimension("length")])
        <CartesianPos3D: (x, y, z) [m]
            [[[1 2 3]
              [4 5 6]]]>

        """
        if isinstance(key, Dimension):
            key = dimension_name(key)

        return ImmutableMap.__getitem__(self, key)

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[Vector]:  # type: ignore[type-arg]
        return self._data.values()

    def items(self) -> ItemsView[str, Vector]:  # type: ignore[type-arg]
        return self._data.items()

    # ===============================================================
    # Quax API

    # TODO: unify this with AvalMixin
    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> w = cx.KinematicSpace(
        ...     length=u.Q([[[1, 2, 3], [4, 5, 6]]], "m"),
        ...     speed=u.Q([7, 8, 9], "m/s") )
        >>> w.aval()
        ShapedArray(int32[1,2,6])

        """
        avals = [v.aval() for v in self.values()]
        shapes = [a.shape for a in avals]
        shape = (
            *jnp.broadcast_shapes(*[s[:-1] for s in shapes]),
            sum(s[-1] for s in shapes),
        )
        dtype = jnp.result_type(*map(jnp.dtype, avals))
        return jax.core.ShapedArray(shape, dtype)

    # ===============================================================
    # Array API

    def __eq__(self: "AbstractVectorLike", other: object) -> Any:
        """Check if the vector is equal to another object."""
        if type(other) is not type(self):
            return NotImplemented

        return jnp.equal(self, cast("AbstractVectorLike", other))

    def __hash__(self) -> int:
        return hash(tuple(self.items()))
