"""Built-in 4-vector classes."""

__all__ = ["FourVector"]

from dataclasses import KW_ONLY, replace
from functools import partial
from typing import TYPE_CHECKING, Any, final

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Shaped

import quaxed.array_api as xp
from unxt import Quantity

from .base import AbstractPosition4D
from coordinax._base import AbstractVector
from coordinax._d3.base import AbstractPosition3D
from coordinax._d3.cartesian import CartesianPosition3D
from coordinax._typing import BatchableLength, BatchableTime, ScalarTime
from coordinax._utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Never

##############################################################################
# Position


@final
class FourVector(AbstractPosition4D):
    """3+1 vector representation.

    The 3+1 vector representation is a 4-vector with 3 spatial coordinates and 1
    time coordinate.

    Parameters
    ----------
    t : Quantity[float, (*batch,), "time"]
        Time coordinate.
    q : AbstractPosition3D[float, (*batch, 3)]
        Spatial coordinates.
    c : Quantity[float, (), "speed"], optional
        Speed of light, by default ``Quantity(299_792.458, "km/s")``.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax import FourVector, CartesianPosition3D

    Create a 3+1 vector with a time and 3 spatial coordinates:

    >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
    >>> w
    FourVector(
      t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
      q=CartesianPosition3D( ... )
    )

    Note that we used a shortcut to create the 3D vector by passing a ``(*batch,
    3)`` array to the `q` argument. This assumes that `q` is a
    :class:`coordinax.CartesianPosition3D` and uses the
    :meth:`coordinax.CartesianPosition3D.constructor` method to create the 3D vector.

    We can also create the 3D vector explicitly:

    >>> q = CartesianPosition3D(x=Quantity(1, "m"), y=Quantity(2, "m"),
    ...                       z=Quantity(3, "m"))
    >>> w = FourVector(t=Quantity(1, "s"), q=q)

    """

    t: BatchableTime | ScalarTime = eqx.field(
        converter=partial(Quantity["time"].constructor, dtype=float)
    )
    """Time coordinate."""

    q: AbstractPosition3D = eqx.field(
        converter=lambda q: (
            q
            if isinstance(q, AbstractPosition3D)
            else CartesianPosition3D.constructor(q)
        )
    )
    """Spatial coordinates."""

    _: KW_ONLY
    c: Shaped[Quantity["speed"], ""] = eqx.field(
        default=Quantity(299_792.458, "km/s"), repr=False
    )
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""

    def __check_init__(self) -> None:
        """Check that the initialization is valid."""
        # Check the shapes are the same, allowing for broadcasting of the time.
        shape = jnp.broadcast_shapes(self.t.shape, self.q.shape)
        if shape != self.q.shape:
            msg = "t and q must be broadcastable to the same shape."
            raise ValueError(msg)

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @AbstractVector.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
    def constructor(
        cls: "type[FourVector]", obj: Shaped[Quantity, "*batch 4"], /
    ) -> "FourVector":
        """Construct a vector from a Quantity array.

        The array is expected to have the components as the last dimension.

        Parameters
        ----------
        obj : Quantity[Any, (*#batch, 4), "..."]
            The array of components.
            The 4 components are the (c x) time, x, y, z.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from unxt import Quantity
        >>> from coordinax import FourVector

        >>> xs = Quantity([0, 1, 2, 3], "meter")  # [ct, x, y, z]
        >>> vec = FourVector.constructor(xs)
        >>> vec
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("m s / km")),
            q=CartesianPosition3D( ... )
        )

        >>> xs = Quantity(jnp.array([[0, 1, 2, 3], [10, 4, 5, 6]]), "meter")
        >>> vec = FourVector.constructor(xs)
        >>> vec
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[2], unit=Unit("m s / km")),
            q=CartesianPosition3D( ... )
        )

        >>> vec.x
        Quantity['length'](Array([1., 4.], dtype=float32), unit='m')

        """
        _ = eqx.error_if(
            obj,
            obj.shape[-1] != 4,
            f"Cannot construct {cls} from array with shape {obj.shape}.",
        )
        c = cls.__dataclass_fields__["c"].default
        return cls(t=obj[..., 0] / c, q=obj[..., 1:])

    # ===============================================================

    def __getattr__(self, name: str) -> Any:
        """Get the attribute from the 3-vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import FourVector, CartesianPosition3D

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w.x
        Quantity['length'](Array(1., dtype=float32), unit='m')

        """
        return getattr(self.q, name)

    # -------------------------------------------

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    @classproperty
    @classmethod
    def differential_cls(cls) -> "Never":  # type: ignore[override]
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    # -------------------------------------------
    # Unary operations

    def __neg__(self) -> "FourVector":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import FourVector, CartesianPosition3D

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> -w
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
            q=CartesianPosition3D( ... )
        )

        """
        return replace(self, t=-self.t, q=-self.q)

    # -------------------------------------------
    # Binary operations

    @AbstractVector.__add__.dispatch  # type: ignore[misc]
    def __add__(self: "FourVector", other: "FourVector") -> "FourVector":
        """Add two 4-vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import FourVector, CartesianPosition3D

        >>> w1 = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w2 = FourVector(t=Quantity(2, "s"), q=Quantity([4, 5, 6], "m"))
        >>> w3 = w1 + w2
        >>> w3
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
            q=CartesianPosition3D( ... )
        )

        >>> w3.t
        Quantity['time'](Array(3., dtype=float32), unit='s')

        >>> w3.x
        Quantity['length'](Array(5., dtype=float32), unit='m')

        """
        return replace(self, t=self.t + other.t, q=self.q + other.q)

    @AbstractVector.__sub__.dispatch  # type: ignore[misc]
    def __sub__(self: "FourVector", other: "FourVector") -> "FourVector":
        """Add two 4-vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import FourVector, CartesianPosition3D

        >>> w1 = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w2 = FourVector(t=Quantity(2, "s"), q=Quantity([4, 5, 6], "m"))
        >>> w3 = w1 - w2
        >>> w3
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
            q=CartesianPosition3D( ... )
        )

        >>> w3.t
        Quantity['time'](Array(-1., dtype=float32), unit='s')

        >>> w3.x
        Quantity['length'](Array(-3., dtype=float32), unit='m')

        """
        return replace(self, t=self.t - other.t, q=self.q - other.q)

    # -------------------------------------------

    @partial(jax.jit)
    def norm2(self) -> Shaped[Quantity["area"], "*#batch"]:
        r"""Return the squared vector norm :math:`(ct)^2 - (x^2 + y^2 + z^2)`.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import FourVector, CartesianPosition3D

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w.norm2()
        Quantity['area'](Array(8.987552e+16, dtype=float32), unit='m2')

        """
        return -(self.q.norm() ** 2) + (self.c * self.t) ** 2  # for units

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        r"""Return the vector norm :math:`\sqrt{(ct)^2 - (x^2 + y^2 + z^2)}`.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import FourVector, CartesianPosition3D

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w.norm()
        Quantity['length'](Array(2.9979248e+08+0.j, dtype=complex64), unit='m')

        """
        return xp.sqrt(xp.asarray(self.norm2(), dtype=complex))


# -----------------------------------------------
# Register additional constructors


# TODO: move to the class in py3.11+
@FourVector.constructor._f.dispatch  # type: ignore[misc]  # noqa: SLF001
def constructor(
    cls: type[FourVector], obj: Shaped[u.Quantity, "*batch 4"], /
) -> FourVector:
    """Construct a vector from a Quantity array.

    The array is expected to have the components as the last dimension.

    Parameters
    ----------
    cls : type[FourVector]
        The class.
    obj : Quantity[Any, (*#batch, 4), "..."]
        The array of components.
        The 4 components are the (c x) time, x, y, z.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astropy.units import Quantity
    >>> from coordinax import FourVector

    >>> xs = Quantity([0, 1, 2, 3], "meter")  # [ct, x, y, z]
    >>> vec = FourVector.constructor(xs)
    >>> vec
    FourVector(
        t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("m s / km")),
        q=CartesianPosition3D( ... )
    )

    >>> xs = Quantity(jnp.array([[0, 1, 2, 3], [10, 4, 5, 6]]), "meter")
    >>> vec = FourVector.constructor(xs)
    >>> vec
    FourVector(
        t=Quantity[PhysicalType('time')](value=f32[2], unit=Unit("m s / km")),
        q=CartesianPosition3D( ... )
    )

    >>> vec.x
    Quantity['length'](Array([1., 4.], dtype=float32), unit='m')

    """
    _ = eqx.error_if(
        obj,
        obj.shape[-1] != 4,
        f"Cannot construct {cls} from array with shape {obj.shape}.",
    )
    c = cls.__dataclass_fields__["c"].default
    return cls(t=obj[..., 0] / c, q=obj[..., 1:])
