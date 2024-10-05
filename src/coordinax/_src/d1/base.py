"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos1D", "AbstractVel1D", "AbstractAcc1D"]


from abc import abstractmethod
from dataclasses import fields

from jaxtyping import Shaped

import quaxed.numpy as jnp
from unxt import Quantity

from coordinax._src.base import AbstractAcc, AbstractPos, AbstractVector, AbstractVel
from coordinax._src.utils import classproperty

#####################################################################


class AbstractPos1D(AbstractPos):
    """Abstract representation of 1D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianPos1D

        return CartesianPos1D

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVel1D"]:
        raise NotImplementedError


# -------------------------------------------------------------------


@AbstractPos1D.from_._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def from_(
    cls: type[AbstractPos1D],
    obj: Shaped[Quantity["length"], "*batch"] | Shaped[Quantity["length"], "*batch 1"],
    /,
) -> AbstractPos1D:
    """Construct a 1D position.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cx.CartesianPos1D.from_(Quantity(1, "meter"))
    CartesianPos1D(x=Quantity[...](value=f32[], unit=Unit("m")))

    >>> cx.CartesianPos1D.from_(Quantity([1], "meter"))
    CartesianPos1D(x=Quantity[...](value=f32[], unit=Unit("m")))

    >>> cx.RadialPos.from_(Quantity(1, "meter"))
    RadialPos(r=Distance(value=f32[], unit=Unit("m")))

    >>> cx.RadialPos.from_(Quantity([1], "meter"))
    RadialPos(r=Distance(value=f32[], unit=Unit("m")))

    """
    comps = {f.name: jnp.atleast_1d(obj)[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


#####################################################################


class AbstractVel1D(AbstractVel):
    """Abstract representation of 1D differentials in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianVel1D

        return CartesianVel1D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractPos1D]:
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type[AbstractAcc]:
        raise NotImplementedError


# -------------------------------------------------------------------


@AbstractVel1D.from_._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def from_(
    cls: type[AbstractVel1D],
    obj: Shaped[Quantity["speed"], "*batch"] | Shaped[Quantity["speed"], "*batch 1"],
    /,
) -> AbstractVel1D:
    """Construct a 1D velocity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cx.CartesianVel1D.from_(Quantity(1, "m/s"))
    CartesianVel1D( d_x=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )

    >>> cx.CartesianVel1D.from_(Quantity([1], "m/s"))
    CartesianVel1D( d_x=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    >>> cx.RadialVel.from_(Quantity(1, "m/s"))
    RadialVel( d_r=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )

    >>> cx.RadialVel.from_(Quantity([1], "m/s"))
    RadialVel( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    """
    comps = {f.name: jnp.atleast_1d(obj)[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


#####################################################################


class AbstractAcc1D(AbstractAcc):
    """Abstract representation of 1D acceleration in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianAcc1D

        return CartesianAcc1D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVel1D]:
        raise NotImplementedError


# -------------------------------------------------------------------


@AbstractAcc1D.from_._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def from_(
    cls: type[AbstractAcc1D],
    obj: Shaped[Quantity["acceleration"], "*batch"]
    | Shaped[Quantity["acceleration"], "*batch 1"],
    /,
) -> AbstractAcc1D:
    """Construct a 1D acceleration.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cx.CartesianAcc1D.from_(Quantity(1, "m/s2"))
    CartesianAcc1D( d2_x=... )

    >>> cx.CartesianAcc1D.from_(Quantity([1], "m/s2"))
    CartesianAcc1D( d2_x=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    >>> cx.RadialAcc.from_(Quantity(1, "m/s2"))
    RadialAcc( d2_r=... )

    >>> cx.RadialAcc.from_(Quantity([1], "m/s2"))
    RadialAcc( d2_r=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    """
    comps = {f.name: jnp.atleast_1d(obj)[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)
