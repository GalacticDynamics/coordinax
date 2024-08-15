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


# TODO: move to the class in py3.11+
@AbstractVector.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[AbstractPosition1D],
    x: Shaped[Quantity["length"], "*batch"] | Shaped[Quantity["length"], "*batch 1"],
    /,
) -> AbstractPosition1D:
    """Construct a 1D vector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cx.CartesianPosition1D.constructor(Quantity(1, "meter"))
    CartesianPosition1D(
        x=Quantity[...](value=f32[], unit=Unit("m"))
    )

    >>> cx.CartesianPosition1D.constructor(Quantity([1], "meter"))
    CartesianPosition1D(
        x=Quantity[...](value=f32[], unit=Unit("m"))
    )

    """
    return cls(**{fields(cls)[0].name: jnp.atleast_1d(x)[..., 0]})


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
