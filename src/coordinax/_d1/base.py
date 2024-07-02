"""Representation of coordinates in different systems."""

__all__ = ["AbstractPosition1D", "AbstractVelocity1D", "AbstractAcceleration1D"]


from abc import abstractmethod
from dataclasses import fields

from jaxtyping import Shaped

from unxt import Quantity

from coordinax._base import AbstractVector
from coordinax._base_acc import AbstractAcceleration
from coordinax._base_pos import AbstractPosition
from coordinax._base_vel import AbstractVelocity
from coordinax._utils import classproperty


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
@AbstractPosition.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[AbstractPosition1D], x: Shaped[Quantity["length"], ""], /
) -> AbstractPosition1D:
    """Construct a 1D vector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax import CartesianPosition1D

    >>> q = CartesianPosition1D.constructor(Quantity(1, "kpc"))
    >>> q
    CartesianPosition1D(
        x=Quantity[PhysicalType('length')](value=f32[1], unit=Unit("kpc"))
    )

    """
    return cls(**{fields(cls)[0].name: x.reshape(1)})


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
