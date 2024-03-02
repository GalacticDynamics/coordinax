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

import array_api_jax_compat as xp
from jax_quantity import Quantity

from .base import Abstract4DVector
from coordinax._base import AbstractVectorBase
from coordinax._d3.base import Abstract3DVector
from coordinax._d3.builtin import Cartesian3DVector
from coordinax._typing import BatchableLength, BatchableTime, ScalarTime
from coordinax._utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Never

##############################################################################
# Position


@final
class FourVector(Abstract4DVector):
    """3+1 vector representation.

    The 3+1 vector representation is a 4-vector with 3 spatial coordinates and 1
    time coordinate.

    Parameters
    ----------
    t : Quantity[float, (*batch,), "time"]
        Time coordinate.
    q : Abstract3DVector[float, (*batch, 3)]
        Spatial coordinates.
    c : Quantity[float, (), "speed"], optional
        Speed of light, by default ``Quantity(299_792.458, "km/s")``.

    Examples
    --------
    >>> from jax_quantity import Quantity
    >>> from coordinax import FourVector, Cartesian3DVector

    Create a 3+1 vector with a time and 3 spatial coordinates:

    >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
    >>> w
    FourVector(
      t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
      q=Cartesian3DVector( ... )
    )

    Note that we used a shortcut to create the 3D vector by passing a ``(*batch,
    3)`` array to the `q` argument. This assumes that `q` is a
    :class:`coordinax.Cartesian3DVector` and uses the
    :meth:`coordinax.Cartesian3DVector.constructor` method to create the 3D vector.

    We can also create the 3D vector explicitly:

    >>> q = Cartesian3DVector(x=Quantity(1, "m"), y=Quantity(2, "m"),
    ...                       z=Quantity(3, "m"))
    >>> w = FourVector(t=Quantity(1, "s"), q=q)

    """

    t: BatchableTime | ScalarTime = eqx.field(
        converter=partial(Quantity["time"].constructor, dtype=float)
    )
    """Time coordinate."""

    q: Abstract3DVector = eqx.field(
        converter=lambda q: (
            q if isinstance(q, Abstract3DVector) else Cartesian3DVector.constructor(q)
        )
    )
    """Spatial coordinates."""

    _: KW_ONLY
    c: Shaped[Quantity["speed"], ""] = eqx.field(
        default_factory=lambda: Quantity(299_792.458, "km/s"),
        repr=False,
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
    @AbstractVectorBase.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
    def constructor(
        cls: "type[FourVector]",
        obj: Quantity | u.Quantity,
        /,  # TODO: shape hint
    ) -> "FourVector":
        """Construct a vector from a Quantity array.

        The array is expected to have the components as the last dimension.

        Parameters
        ----------
        obj : Quantity[Any, (*#batch, 4), "..."]
            The array of components.
            The 4 components are the (c x) time, x, y, z.

        Returns
        -------
        FourVector
            The vector constructed from the array.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from jax_quantity import Quantity
        >>> from coordinax import FourVector

        >>> xs = Quantity([0, 1, 2, 3], "meter")  # [ct, x, y, z]
        >>> vec = FourVector.constructor(xs)
        >>> vec
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("m s / km")),
            q=Cartesian3DVector( ... )
        )

        >>> xs = Quantity(jnp.array([[0, 1, 2, 3], [10, 4, 5, 6]]), "meter")
        >>> vec = FourVector.constructor(xs)
        >>> vec
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[2], unit=Unit("m s / km")),
            q=Cartesian3DVector( ... )
        )

        >>> vec.x
        Quantity['length'](Array([1., 4.], dtype=float32), unit='m')

        """
        _ = eqx.error_if(
            obj,
            obj.shape[-1] != 4,
            f"Cannot construct {cls} from array with shape {obj.shape}.",
        )
        c = cls.__dataclass_fields__["c"].default_factory()
        comps = {"t": obj[..., 0] / c, "q": obj[..., 1:]}
        return cls(**comps)

    # ===============================================================

    def __getattr__(self, name: str) -> Any:
        """Get the attribute from the 3-vector.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import FourVector, Cartesian3DVector

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w.x
        Quantity['length'](Array(1., dtype=float32), unit='m')

        """
        return getattr(self.q, name)

    # -------------------------------------------

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVectorBase]:
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
        >>> from jax_quantity import Quantity
        >>> from coordinax import FourVector, Cartesian3DVector

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> -w
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
            q=Cartesian3DVector( ... )
        )

        """
        return replace(self, t=-self.t, q=-self.q)

    # -------------------------------------------
    # Binary operations

    def __add__(self, other: Any) -> "FourVector":
        """Add two 4-vectors.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import FourVector, Cartesian3DVector

        >>> w1 = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w2 = FourVector(t=Quantity(2, "s"), q=Quantity([4, 5, 6], "m"))
        >>> w3 = w1 + w2
        >>> w3
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
            q=Cartesian3DVector( ... )
        )

        >>> w3.t
        Quantity['time'](Array(3., dtype=float32), unit='s')

        >>> w3.x
        Quantity['length'](Array(5., dtype=float32), unit='m')

        """
        if not isinstance(other, FourVector):
            return NotImplemented

        return replace(self, t=self.t + other.t, q=self.q + other.q)

    def __sub__(self, other: Any) -> "FourVector":
        """Add two 4-vectors.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import FourVector, Cartesian3DVector

        >>> w1 = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w2 = FourVector(t=Quantity(2, "s"), q=Quantity([4, 5, 6], "m"))
        >>> w3 = w1 - w2
        >>> w3
        FourVector(
            t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
            q=Cartesian3DVector( ... )
        )

        >>> w3.t
        Quantity['time'](Array(-1., dtype=float32), unit='s')

        >>> w3.x
        Quantity['length'](Array(-3., dtype=float32), unit='m')

        """
        if not isinstance(other, FourVector):
            return NotImplemented

        return replace(self, t=self.t - other.t, q=self.q - other.q)

    # -------------------------------------------

    @partial(jax.jit)
    def norm2(self) -> Shaped[Quantity["area"], "*#batch"]:
        r"""Return the squared vector norm :math:`(ct)^2 - (x^2 + y^2 + z^2)`.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import FourVector, Cartesian3DVector

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
        >>> from jax_quantity import Quantity
        >>> from coordinax import FourVector, Cartesian3DVector

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w.norm()
        Quantity['length'](Array(2.9979248e+08+0.j, dtype=complex64), unit='m')

        """
        return xp.sqrt(xp.asarray(self.norm2(), dtype=complex))
