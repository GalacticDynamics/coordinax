"""Representation of coordinates in different systems."""

__all__ = ["AbstractAcc1D", "AbstractPos1D", "AbstractVel1D"]


from abc import abstractmethod
from dataclasses import fields

from jaxtyping import Shaped

import quaxed.numpy as jnp
from unxt.quantity import AbstractQuantity

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import (
    AbstractAcc,
    AbstractPos,
    AbstractVector,
    AbstractVel,
)

#####################################################################


class AbstractPos1D(AbstractPos):
    """Abstract representation of 1D coordinates in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos1D._dimensionality()
        1

        """
        return 1

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


@AbstractVector.from_.dispatch  # type: ignore[misc]
def from_(
    cls: type[AbstractPos1D],
    obj: Shaped[AbstractQuantity, "*batch"] | Shaped[AbstractQuantity, "*batch 1"],
    /,
) -> AbstractPos1D:
    """Construct a 1D position.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> cx.vecs.CartesianPos1D.from_(u.Quantity(1, "meter"))
    CartesianPos1D( x=Quantity[...](value=...i32[], unit=Unit("m")) )

    >>> cx.vecs.CartesianPos1D.from_(u.Quantity([1], "meter"))
    CartesianPos1D(x=Quantity[...](value=i32[], unit=Unit("m")))

    >>> cx.vecs.CartesianPos1D.from_(cx.Distance(1, "meter"))
    CartesianPos1D( x=Quantity[...](value=...i32[], unit=Unit("m")) )

    >>> cx.vecs.RadialPos.from_(u.Quantity(1, "meter"))
    RadialPos(r=Distance(value=...i32[], unit=Unit("m")))

    >>> cx.vecs.RadialPos.from_(u.Quantity([1], "meter"))
    RadialPos(r=Distance(value=...i32[], unit=Unit("m")))

    """
    obj = jnp.atleast_1d(obj)
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls.from_(comps)


#####################################################################


class AbstractVel1D(AbstractVel):
    """Abstract representation of 1D differentials in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianVel1D._dimensionality()
        1

        """
        return 1

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


@AbstractVector.from_.dispatch  # type: ignore[misc]
def from_(
    cls: type[AbstractVel1D],
    obj: Shaped[AbstractQuantity, "*batch"] | Shaped[AbstractQuantity, "*batch 1"],
    /,
) -> AbstractVel1D:
    """Construct a 1D velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> cx.vecs.CartesianVel1D.from_(u.Quantity(1, "m/s"))
    CartesianVel1D( d_x=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )

    >>> cx.vecs.CartesianVel1D.from_(u.Quantity([1], "m/s"))
    CartesianVel1D( d_x=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    >>> cx.vecs.RadialVel.from_(u.Quantity(1, "m/s"))
    RadialVel( d_r=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )

    >>> cx.vecs.RadialVel.from_(u.Quantity([1], "m/s"))
    RadialVel( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    """
    obj = jnp.atleast_1d(obj)
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls.from_(comps)


#####################################################################


class AbstractAcc1D(AbstractAcc):
    """Abstract representation of 1D acceleration in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianAcc1D._dimensionality()
        1

        """
        return 1

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


@AbstractVector.from_.dispatch  # type: ignore[misc]
def from_(
    cls: type[AbstractAcc1D],
    obj: Shaped[AbstractQuantity, "*batch"] | Shaped[AbstractQuantity, "*batch 1"],
    /,
) -> AbstractAcc1D:
    """Construct a 1D acceleration.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> cx.vecs.CartesianAcc1D.from_(u.Quantity(1, "m/s2"))
    CartesianAcc1D( d2_x=... )

    >>> cx.vecs.CartesianAcc1D.from_(u.Quantity([1], "m/s2"))
    CartesianAcc1D( d2_x=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    >>> cx.vecs.RadialAcc.from_(u.Quantity(1, "m/s2"))
    RadialAcc( d2_r=... )

    >>> cx.vecs.RadialAcc.from_(u.Quantity([1], "m/s2"))
    RadialAcc( d2_r=Quantity[...](value=i32[], unit=Unit("m / s2")) )

    """
    obj = jnp.atleast_1d(obj)
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls.from_(comps)
