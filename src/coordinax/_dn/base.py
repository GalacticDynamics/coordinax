"""Representation of coordinates in different systems."""

__all__ = ["AbstractPositionND", "AbstractVelocityND", "AbstractAccelerationND"]


from abc import abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import equinox as eqx

import quaxed.lax as qlax
import quaxed.numpy as qnp

from coordinax._base import AbstractVector
from coordinax._base_acc import AbstractAcceleration
from coordinax._base_pos import AbstractPosition
from coordinax._base_vel import AbstractVelocity
from coordinax._utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Never


class AbstractPositionND(AbstractPosition):
    """Abstract representation of N-D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianPositionND

        return CartesianPositionND

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
        transpose is performed on the last two non-feature axes.
        """
        ndim = self.q.ndim
        if ndim < 2:
            msg = (
                f"x must be at least two-dimensional for matrix_transpose; got {ndim=}"
            )
            raise ValueError(msg)
        axes = (*range(ndim - 2), ndim - 2, ndim - 3)
        return replace(self, q=qlax.transpose(self.q, axes))

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the vector's components.

        When represented as a single array, the vector has an additional
        dimension at the end for the components.

        """
        return self.q.shape[:-1]

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose the vector."""
        return replace(self, q=qlax.transpose(self.q, [*range(self.q.ndim)[1::-1], -1]))

    # ===============================================================
    # Further array methods

    def flatten(self) -> "Self":
        """Flatten the vector."""
        return replace(self, q=qnp.reshape(self.q, (self.size, self.q.shape[-1]), "C"))

    def reshape(self, *hape: Any, order: str = "C") -> "Self":
        """Reshape the vector."""
        return replace(self, q=self.q.reshape(*hape, self.q.shape[-1], order=order))


class AbstractVelocityND(AbstractVelocity):
    """Abstract representation of N-D vector differentials."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianVelocityND

        return CartesianVelocityND

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractPositionND]:
        raise NotImplementedError

    # ===============================================================
    # Array API

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        The last axis is interpreted as the feature axis. The matrix
        transpose is performed on the last two non-feature axes.
        """
        ndim = self.d_q.ndim
        if ndim < 2:
            msg = (
                f"x must be at least two-dimensional for matrix_transpose; got {ndim=}"
            )
            raise ValueError(msg)
        axes = (*range(ndim - 2), ndim - 2, ndim - 3)
        return replace(self, q=qlax.transpose(self.d_q, axes))

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the vector's components.

        When represented as a single array, the vector has an additional
        dimension at the end for the components.

        """
        return self.d_q.shape[:-1]

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose the vector."""
        return replace(
            self, q=qlax.transpose(self.d_q, [*range(self.d_q.ndim)[1::-1], -1])
        )

    # ===============================================================
    # Further array methods

    def flatten(self) -> "Self":
        """Flatten the vector."""
        return replace(
            self, q=qnp.reshape(self.d_q, (self.size, self.d_q.shape[-1]), "C")
        )

    def reshape(self, *hape: Any, order: str = "C") -> "Self":
        """Reshape the vector."""
        return replace(self, q=self.q.reshape(*hape, self.q.shape[-1], order=order))


class AbstractAccelerationND(AbstractAcceleration):
    """Abstract representation of N-D vector differentials."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        """Get the Cartesian acceleration class.

        Examples
        --------
        >>> import coordinax as cx
        >>> cx.CartesianAccelerationND._cartesian_cls
        <class 'coordinax...CartesianAccelerationND'>

        """
        from .cartesian import CartesianAccelerationND

        return CartesianAccelerationND

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVelocityND]:
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Array API

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        The last axis is interpreted as the feature axis. The matrix
        transpose is performed on the last two non-feature axes.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> vec = cx.CartesianAccelerationND(Quantity([[[1, 2, 3]],
        ...                                            [[4, 5, 6]]], "m/s^2"))
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
        >>> vec = cx.CartesianAccelerationND(Quantity([[[1, 2, 3]],
        ...                                            [[4, 5, 6]]], "m/s^2"))
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
        >>> vec = cx.CartesianAccelerationND(Quantity([[[1, 2, 3]],
        ...                                            [[4, 5, 6]]], "m/s^2"))
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
        >>> vec = cx.CartesianAccelerationND(Quantity([[[1, 2, 3]],
        ...                                            [[4, 5, 6]]], "m/s^2"))
        >>> vec.shape
        (2, 1)

        >>> vec.flatten().shape
        (2,)

        """
        return replace(
            self, d2_q=qnp.reshape(self.d2_q, (self.size, self.d2_q.shape[-1]), "C")
        )

    def reshape(self, *shape: Any, order: str = "C") -> "Self":
        """Reshape the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> vec = cx.CartesianAccelerationND(Quantity([1, 2, 3], "m/s^2"))
        >>> vec.shape
        ()

        >>> vec.reshape(1, 1).shape
        (1, 1)

        """
        return replace(
            self, d2_q=self.d2_q.reshape(*shape, self.d2_q.shape[-1], order=order)
        )
