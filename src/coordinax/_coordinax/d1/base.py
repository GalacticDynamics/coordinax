"""Representation of coordinates in different systems."""

__all__ = ["AbstractPosition1D", "AbstractVelocity1D", "AbstractAcceleration1D"]


from abc import abstractmethod
from dataclasses import fields

from jaxtyping import Shaped

import quaxed.numpy as jnp
from unxt import Quantity

from coordinax._coordinax.base import AbstractVector
from coordinax._coordinax.base_acc import AbstractAcceleration
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.base_vel import AbstractVelocity
from coordinax._coordinax.utils import classproperty

#####################################################################


class AbstractPosition1D(AbstractPosition):
    """Abstract representation of 1D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianPosition1D

        return CartesianPosition1D

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVelocity1D"]:
        raise NotImplementedError


# -------------------------------------------------------------------


@AbstractPosition1D.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[AbstractPosition1D],
    obj: Shaped[Quantity["length"], "*batch"] | Shaped[Quantity["length"], "*batch 1"],
    /,
) -> AbstractPosition1D:
    """Construct a 1D position.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cx.CartesianPosition1D.constructor(Quantity(1, "meter"))
    CartesianPosition1D( x=Quantity[...](value=f32[], unit=Unit("m")) )

    >>> cx.CartesianPosition1D.constructor(Quantity([1], "meter"))
    CartesianPosition1D( x=Quantity[...](value=f32[], unit=Unit("m")) )

    >>> cx.RadialPosition.constructor(Quantity(1, "meter"))
    RadialPosition(r=Distance(value=f32[], unit=Unit("m")))

    >>> cx.RadialPosition.constructor(Quantity([1], "meter"))
    RadialPosition(r=Distance(value=f32[], unit=Unit("m")))

    """
    comps = {f.name: jnp.atleast_1d(obj)[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


#####################################################################


class AbstractVelocity1D(AbstractVelocity):
    """Abstract representation of 1D differentials in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianVelocity1D

        return CartesianVelocity1D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractPosition1D]:
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type[AbstractAcceleration]:
        raise NotImplementedError


# -------------------------------------------------------------------


@AbstractVelocity1D.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[AbstractVelocity1D],
    obj: Shaped[Quantity["speed"], "*batch"] | Shaped[Quantity["speed"], "*batch 1"],
    /,
) -> AbstractVelocity1D:
    """Construct a 1D velocity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cx.CartesianVelocity1D.constructor(Quantity(1, "m/s"))
    CartesianVelocity1D( d_x=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    >>> cx.CartesianVelocity1D.constructor(Quantity([1], "m/s"))
    CartesianVelocity1D( d_x=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    >>> cx.RadialVelocity.constructor(Quantity(1, "m/s"))
    RadialVelocity( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    >>> cx.RadialVelocity.constructor(Quantity([1], "m/s"))
    RadialVelocity( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    """
    comps = {f.name: jnp.atleast_1d(obj)[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


#####################################################################


class AbstractAcceleration1D(AbstractAcceleration):
    """Abstract representation of 1D acceleration in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianAcceleration1D

        return CartesianAcceleration1D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVelocity1D]:
        raise NotImplementedError


# -------------------------------------------------------------------


@AbstractAcceleration1D.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[AbstractAcceleration1D],
    obj: Shaped[Quantity["acceleration"], "*batch"]
    | Shaped[Quantity["acceleration"], "*batch 1"],
    /,
) -> AbstractAcceleration1D:
    """Construct a 1D acceleration.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cx.CartesianAcceleration1D.constructor(Quantity(1, "m/s2"))
    CartesianAcceleration1D( d2_x=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    >>> cx.CartesianAcceleration1D.constructor(Quantity([1], "m/s2"))
    CartesianAcceleration1D( d2_x=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    >>> cx.RadialAcceleration.constructor(Quantity(1, "m/s2"))
    RadialAcceleration( d2_r=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    >>> cx.RadialAcceleration.constructor(Quantity([1], "m/s2"))
    RadialAcceleration( d2_r=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    """
    comps = {f.name: jnp.atleast_1d(obj)[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)
