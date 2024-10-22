"""Representation of velocities in different systems."""

__all__ = ["AbstractVel"]

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar
from typing_extensions import override

import equinox as eqx
import jax
from quax import register

import quaxed.numpy as jnp
from dataclassish import field_items
from unxt import Quantity

from .base import AbstractVector
from .base_pos import AbstractPos
from coordinax._src.utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Self

VelT = TypeVar("VelT", bound="AbstractVel")

DIFFERENTIAL_CLASSES: set[type["AbstractVel"]] = set()


class AbstractVel(AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        DIFFERENTIAL_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVector"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialVel._cartesian_cls
        <class 'coordinax...CartesianVel1D'>

        >>> cx.SphericalVel._cartesian_cls
        <class 'coordinax...CartesianVel3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type["AbstractPos"]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialVel.integral_cls.__name__
        'RadialPos'

        >>> cx.SphericalVel.integral_cls.__name__
        'SphericalPos'

        """
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractAcc"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialVel.differential_cls.__name__
        'RadialAcc'

        >>> cx.SphericalVel.differential_cls.__name__
        'SphericalAcc'

        """
        raise NotImplementedError

    # ===============================================================
    # Quax

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        raise NotImplementedError

    # ===============================================================
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> dr = cx.RadialVel.from_([1], "m/s")
        >>> -dr
        RadialVel( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

        >>> dp = cx.PolarVel(Quantity(1, "m/s"), Quantity(1, "mas/yr"))
        >>> neg_dp = -dp
        >>> neg_dp.d_r
        Quantity['speed'](Array(-1., dtype=float32), unit='m / s')
        >>> neg_dp.d_phi
        Quantity['angular frequency'](Array(-1., dtype=float32), unit='mas / yr')

        """
        return jax.tree.map(jnp.negative, self)

    # ===============================================================
    # Convenience methods

    @override
    def represent_as(
        self,
        target: type[VelT],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> VelT:
        """Represent the vector as another type.

        This just forwards to `coordinax.represent_as`.

        Parameters
        ----------
        target : type[`coordinax.AbstractVel`]
            The type to represent the vector as.
        *args, **kwargs : Any
            Extra arguments. These are passed to `coordinax.represent_as` and
            might be used, depending on the dispatched method. Generally the
            first argument is the position (`coordinax.AbstractPos`) at
            which the velocity is defined. In general this is a required
            argument, though it is not for Cartesian-to-Cartesian transforms --
            see https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates
            for more information.

        Returns
        -------
        `coordinax.AbstractVel`
            The vector represented as the target type.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> p = cx.CartesianVel3D.from_([4, 5, 6], "m/s")
        >>> sph = p.represent_as(cx.SphericalVel, q)
        >>> sph
        SphericalVel(
            d_r=Quantity[...)]( value=f32[], unit=Unit("m / s") ),
            d_theta=Quantity[...]( value=f32[], unit=Unit("rad / s") ),
            d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
        )
        >>> sph.d_r
        Quantity['speed'](Array(8.55236, dtype=float32), unit='m / s')

        """
        from coordinax import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, *args, **kwargs)

    @partial(eqx.filter_jit, inline=True)
    def norm(self, position: AbstractPos, /) -> Quantity["speed"]:
        """Return the norm of the vector."""
        return self.represent_as(self._cartesian_cls, position).norm()


# ---------------------------------------------------------


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vel_q(self: AbstractVel, other: Quantity["time"]) -> AbstractPos:
    """Multiply the vector by a time :class:`unxt.Quantity` to get a position.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dr = cx.RadialVel(Quantity(1, "m/s"))
    >>> vec = dr * Quantity(2, "s")
    >>> vec
    RadialPos(r=Distance(value=f32[], unit=Unit("m")))
    >>> vec.r
    Distance(Array(2., dtype=float32), unit='m')

    >>> lax.mul(dr, Quantity(2, "s")).r
    Distance(Array(2., dtype=float32), unit='m')

    """
    return self.integral_cls.from_({k[2:]: v * other for k, v in field_items(self)})
