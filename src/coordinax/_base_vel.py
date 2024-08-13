"""Representation of velocities in different systems."""

__all__ = ["AbstractVelocity"]

from abc import abstractmethod
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import jax

from dataclassish import field_items
from unxt import Quantity

from ._base import AbstractVector
from ._base_pos import AbstractPosition
from ._utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Self

VelT = TypeVar("VelT", bound="AbstractVelocity")

DIFFERENTIAL_CLASSES: set[type["AbstractVelocity"]] = set()


class AbstractVelocity(AbstractVector):  # pylint: disable=abstract-method
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

        >>> cx.RadialVelocity._cartesian_cls
        <class 'coordinax...CartesianVelocity1D'>

        >>> cx.SphericalVelocity._cartesian_cls
        <class 'coordinax...CartesianVelocity3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type["AbstractPosition"]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialVelocity.integral_cls.__name__
        'RadialPosition'

        >>> cx.SphericalVelocity.integral_cls.__name__
        'SphericalPosition'

        """
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractAcceleration"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialVelocity.differential_cls.__name__
        'RadialAcceleration'

        >>> cx.SphericalVelocity.differential_cls.__name__
        'SphericalAcceleration'

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

        >>> dr = cx.RadialVelocity.constructor([1], "m/s")
        >>> -dr
        RadialVelocity( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

        >>> dp = cx.PolarVelocity(Quantity(1, "m/s"), Quantity(1, "mas/yr"))
        >>> neg_dp = -dp
        >>> neg_dp.d_r
        Quantity['speed'](Array(-1., dtype=float32), unit='m / s')
        >>> neg_dp.d_phi
        Quantity['angular frequency'](Array(-1., dtype=float32), unit='mas / yr')

        """
        return replace(self, **{k: -v for k, v in field_items(self)})

    # ===============================================================
    # Binary operations

    @AbstractVector.__mul__.dispatch  # type: ignore[misc]
    def __mul__(self: "AbstractVelocity", other: Quantity) -> "AbstractPosition":
        """Multiply the vector by a :class:`unxt.Quantity`.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> dr = cx.RadialVelocity(Quantity(1, "m/s"))
        >>> vec = dr * Quantity(2, "s")
        >>> vec
        RadialPosition(r=Distance(value=f32[], unit=Unit("m")))
        >>> vec.r
        Distance(Array(2., dtype=float32), unit='m')

        """
        return self.integral_cls.constructor(
            {k[2:]: v * other for k, v in field_items(self)}
        )

    # ===============================================================
    # Convenience methods

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
        target : type[`coordinax.AbstractVelocity`]
            The type to represent the vector as.
        *args, **kwargs : Any
            Extra arguments. These are passed to `coordinax.represent_as` and
            might be used, depending on the dispatched method. Generally the
            first argument is the position (`coordinax.AbstractPosition`) at
            which the velocity is defined. In general this is a required
            argument, though it is not for Cartesian-to-Cartesian transforms --
            see https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates
            for more information.

        Returns
        -------
        `coordinax.AbstractVelocity`
            The vector represented as the target type.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "m")
        >>> p = cx.CartesianVelocity3D.constructor([4, 5, 6], "m/s")
        >>> sph = p.represent_as(cx.SphericalVelocity, q)
        >>> sph
        SphericalVelocity(
            d_r=Quantity[...)]( value=f32[], unit=Unit("m / s") ),
            d_theta=Quantity[...]( value=f32[], unit=Unit("rad / s") ),
            d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
        )
        >>> sph.d_r
        Quantity['speed'](Array(8.55236, dtype=float32), unit='m / s')

        """
        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, *args, **kwargs)

    @partial(jax.jit)
    def norm(self, position: AbstractPosition, /) -> Quantity["speed"]:
        """Return the norm of the vector."""
        return self.represent_as(self._cartesian_cls, position).norm()


# =============================================================================


class AdditionMixin(AbstractVector):
    """Mixin for addition operations."""

    # TODO: use dispatch
    def __add__(self, other: Any, /) -> "Self":
        """Add two differentials.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> q = cx.CartesianVelocity3D.constructor([1, 2, 3], "km/s")
        >>> q2 = q + q
        >>> q2.d_y
        Quantity['speed'](Array(4., dtype=float32), unit='km / s')

        """
        if not isinstance(other, self._cartesian_cls):
            msg = f"Cannot add {type(other)!r} to {self._cartesian_cls!r}."
            raise TypeError(msg)

        return replace(self, **{k: v + getattr(other, k) for k, v in field_items(self)})

    # TODO: use dispatch
    def __sub__(self, other: Any, /) -> "Self":
        """Subtract two differentials.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.CartesianVelocity3D.constructor([1, 2, 3], "km/s")
        >>> q2 = q - q
        >>> q2.d_y
        Quantity['speed'](Array(0., dtype=float32), unit='km / s')

        """
        if not isinstance(other, self._cartesian_cls):
            msg = f"Cannot subtract {type(other)!r} from {self._cartesian_cls!r}."
            raise TypeError(msg)

        return replace(self, **{k: v - getattr(other, k) for k, v in field_items(self)})
