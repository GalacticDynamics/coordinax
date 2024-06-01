"""Representation of coordinates in different systems."""

__all__ = ["Abstract1DVector", "Abstract1DVectorDifferential"]


from abc import abstractmethod
from dataclasses import fields

from jaxtyping import Shaped

from unxt import Quantity

from coordinax._base import AbstractVector
from coordinax._base_pos import AbstractPosition
from coordinax._base_vel import AbstractVelocity
from coordinax._utils import classproperty


class Abstract1DVector(AbstractPosition):
    """Abstract representation of 1D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .builtin import Cartesian1DVector

        return Cartesian1DVector

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["Abstract1DVectorDifferential"]:
        raise NotImplementedError


# TODO: move to the class in py3.11+
@AbstractPosition.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: "type[Abstract1DVector]", x: Shaped[Quantity["length"], ""], /
) -> "Abstract1DVector":
    """Construct a 1D vector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax import Cartesian1DVector

    >>> q = Cartesian1DVector.constructor(Quantity(1, "kpc"))
    >>> q
    Cartesian1DVector(
        x=Quantity[PhysicalType('length')](value=f32[1], unit=Unit("kpc"))
    )

    """
    return cls(**{fields(cls)[0].name: x.reshape(1)})


class Abstract1DVectorDifferential(AbstractVelocity):
    """Abstract representation of 1D differentials in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .builtin import CartesianDifferential1D

        return CartesianDifferential1D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[Abstract1DVector]:
        raise NotImplementedError
