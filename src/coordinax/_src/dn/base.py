"""Representation of coordinates in different systems."""

__all__ = ["AbstractPosND", "AbstractVelND", "AbstractAccND"]


from abc import abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import equinox as eqx

import quaxed.lax as qlax
import quaxed.numpy as jnp

from coordinax._src.base import (
    AbstractAcc,
    AbstractPos,
    AbstractVector,
    AbstractVel,
)
from coordinax._src.utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Never


class AbstractPosND(AbstractPos):
    """Abstract representation of N-D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianPosND

        return CartesianPosND

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> "Never":  # type: ignore[override]
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    # ===============================================================
    # Array API

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        The last axis is interpreted as the feature axis. The matrix
        transpose is performed on the last two batch axes.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> vec = cx.CartesianPosND(Quantity([[[1, 2, 3]],
        ...                                   [[4, 5, 6]]], "m"))
        >>> vec.shape
        (2, 1)

        >>> vec.mT.shape
        (1, 2)

        """
        ndim = self.ndim
        ndim = eqx.error_if(
            ndim,
            ndim < 2,
            f"x must be at least two-dimensional for matrix_transpose; got {ndim=}",
        )
        axes = (*range(ndim - 3), ndim - 1, ndim - 2, ndim)
        return replace(self, q=qlax.transpose(self.q, axes))

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the vector's components.

        When represented as a single array, the vector has an additional
        dimension at the end for the components.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPosND.from_([[[1, 2]], [[3, 4]]], "m")
        >>> vec.shape
        (2, 1)

        """
        return self.q.shape[:-1]

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPosND.from_([[[1, 2]], [[3, 4]]], "m")
        >>> vec.shape
        (2, 1)

        >>> vec.T.shape
        (1, 2)

        """
        return replace(
            self, q=qlax.transpose(self.q, (*range(self.ndim)[::-1], self.ndim))
        )

    # ===============================================================
    # Further array methods

    def flatten(self) -> "Self":
        """Flatten the N-dimensional position.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPosND.from_([[[1, 2]], [[3, 4]]], "m")
        >>> vec.shape
        (2, 1)

        >>> vec.flatten().shape
        (2,)

        """
        return replace(self, q=jnp.reshape(self.q, (self.size, self.q.shape[-1]), "C"))

    def reshape(self, *shape: Any, order: str = "C") -> "Self":
        """Reshape the N-dimensional position.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPosND.from_([[1, 2], [3, 4]], "m")
        >>> vec.shape
        (2,)

        >>> vec.reshape(1, 2, 1).shape
        (1, 2, 1)

        """
        return replace(self, q=self.q.reshape(*shape, self.q.shape[-1], order=order))


#####################################################################


class AbstractVelND(AbstractVel):
    """Abstract representation of N-D vector differentials."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        """Get the Cartesian velocity class.

        Examples
        --------
        >>> import coordinax as cx
        >>> cx.CartesianVelND._cartesian_cls
        <class 'coordinax...CartesianVelND'>

        """
        from .cartesian import CartesianVelND

        return CartesianVelND

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractPosND]:
        """Get the integral class."""
        raise NotImplementedError

    # ===============================================================
    # Array API

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        The last axis is interpreted as the feature axis. The matrix
        transpose is performed on the last two batch axes.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianVelND.from_([[[1, 2]], [[3, 4]]], "m/s")
        >>> vec.shape
        (2, 1)

        >>> vec.mT.shape
        (1, 2)

        """
        ndim = self.ndim
        ndim = eqx.error_if(
            ndim,
            ndim < 2,
            f"x must be at least two-dimensional for matrix_transpose; got {ndim=}",
        )
        axes = (*range(ndim - 3), ndim - 1, ndim - 2, ndim)
        return replace(self, d_q=qlax.transpose(self.d_q, axes))

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the vector's components.

        When represented as a single array, the vector has an additional
        dimension at the end for the components.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianVelND.from_([[1, 2], [3, 4]], "m/s")
        >>> vec.shape
        (2,)

        """
        return self.d_q.shape[:-1]

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianVelND.from_([[[1, 2]], [[3, 4]]], "m/s")
        >>> vec.shape
        (2, 1)

        >>> vec.T.shape
        (1, 2)

        """
        return replace(
            self, d_q=qlax.transpose(self.d_q, (*range(self.ndim)[::-1], self.ndim))
        )

    # ===============================================================
    # Further array methods

    def flatten(self) -> "Self":
        """Flatten the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianVelND.from_([[1, 2], [3, 4]], "m/s")
        >>> vec.flatten()
        CartesianVelND(
            d_q=Quantity[...]( value=f32[2,2], unit=Unit("m / s") )
        )

        """
        return replace(
            self, d_q=jnp.reshape(self.d_q, (self.size, self.d_q.shape[-1]), "C")
        )

    def reshape(self, *shape: Any, order: str = "C") -> "Self":
        """Reshape the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> vec = cx.CartesianVelND(Quantity([1, 2, 3], "m/s"))
        >>> vec.shape
        ()

        >>> vec.reshape(1, 1).shape
        (1, 1)

        """
        return replace(
            self, d_q=self.d_q.reshape(*shape, self.d_q.shape[-1], order=order)
        )


#####################################################################


class AbstractAccND(AbstractAcc):
    """Abstract representation of N-D vector differentials."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        """Get the Cartesian acceleration class.

        Examples
        --------
        >>> import coordinax as cx
        >>> cx.CartesianAccND._cartesian_cls
        <class 'coordinax...CartesianAccND'>

        """
        from .cartesian import CartesianAccND

        return CartesianAccND

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVelND]:
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Array API

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        The last axis is interpreted as the feature axis. The matrix
        transpose is performed on the last two batch axes.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> vec = cx.CartesianAccND(Quantity([[[1, 2, 3]],
        ...                                            [[4, 5, 6]]], "m/s2"))
        >>> vec.shape
        (2, 1)

        >>> vec.mT.shape
        (1, 2)

        """
        ndim = self.ndim
        ndim = eqx.error_if(
            ndim,
            ndim < 2,
            f"x must be at least two-dimensional for matrix_transpose; got {ndim=}",
        )
        axes = (*range(ndim - 3), ndim - 1, ndim - 2, ndim)
        return replace(self, d2_q=qlax.transpose(self.d2_q, axes))

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the vector's components.

        When represented as a single array, the vector has an additional
        dimension at the end for the components.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> vec = cx.CartesianAccND(Quantity([[[1, 2, 3]],
        ...                                            [[4, 5, 6]]], "m/s2"))
        >>> vec.shape
        (2, 1)

        """
        return self.d2_q.shape[:-1]

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose the vector's batch axes, preserving the feature axis.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> vec = cx.CartesianAccND(Quantity([[[1, 2, 3]],
        ...                                            [[4, 5, 6]]], "m/s2"))
        >>> vec.shape
        (2, 1)
        >>> vec.T.shape
        (1, 2)

        """
        return replace(
            self,
            d2_q=qlax.transpose(self.d2_q, (*range(self.ndim)[::-1], self.ndim)),
        )

    # ===============================================================
    # Further array methods

    def flatten(self) -> "Self":
        """Flatten the vector's batch dimensions, preserving the component axis.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> vec = cx.CartesianAccND(Quantity([[[1, 2, 3]],
        ...                                            [[4, 5, 6]]], "m/s2"))
        >>> vec.shape
        (2, 1)

        >>> vec.flatten().shape
        (2,)

        """
        return replace(
            self, d2_q=jnp.reshape(self.d2_q, (self.size, self.d2_q.shape[-1]), "C")
        )

    def reshape(self, *shape: Any, order: str = "C") -> "Self":
        """Reshape the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> vec = cx.CartesianAccND(Quantity([1, 2, 3], "m/s2"))
        >>> vec.shape
        ()

        >>> vec.reshape(1, 1).shape
        (1, 1)

        """
        return replace(
            self, d2_q=self.d2_q.reshape(*shape, self.d2_q.shape[-1], order=order)
        )
