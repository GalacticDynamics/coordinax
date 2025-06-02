"""Representation of coordinates in different systems."""

__all__ = ["AbstractAccND", "AbstractPosND", "AbstractVelND"]


from dataclasses import replace
from typing import TYPE_CHECKING, Any

import equinox as eqx

import quaxed.lax as qlax

import coordinax._src.custom_types as ct
from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

if TYPE_CHECKING:
    from typing import Self

    import coordinax.vecs


class AbstractPosND(AbstractPos):
    """Abstract representation of N-D coordinates in different systems."""

    q: eqx.AbstractVar[ct.BBtLength]

    # ===============================================================
    # Array API

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        The last axis is interpreted as the feature axis. The matrix
        transpose is performed on the last two batch axes.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianPosND(u.Quantity([[[1, 2, 3]],
        ...                                          [[4, 5, 6]]], "m"))
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

        >>> vec = cx.vecs.CartesianPosND.from_([[[1, 2]], [[3, 4]]], "m")
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

        >>> vec = cx.vecs.CartesianPosND.from_([[[1, 2]], [[3, 4]]], "m")
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

        >>> vec = cx.vecs.CartesianPosND.from_([[[1, 2]], [[3, 4]]], "m")
        >>> vec.shape
        (2, 1)

        >>> vec.flatten().shape
        (2,)

        """
        return replace(self, q=self.q.reshape(-1, self.q.shape[-1]))

    def reshape(self, *shape: Any, order: str = "C") -> "Self":
        """Reshape the N-dimensional position.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianPosND.from_([[1, 2], [3, 4]], "m")
        >>> vec.shape
        (2,)

        >>> vec.reshape(1, 2, 1).shape
        (1, 2, 1)

        """
        return replace(self, q=self.q.reshape(*shape, self.q.shape[-1], order=order))


#####################################################################


class AbstractVelND(AbstractVel):
    """Abstract representation of N-D vector differentials."""

    q: eqx.AbstractVar[ct.BBtSpeed]

    # ===============================================================
    # Array API

    @property
    def mT(self) -> "coordinax.vecs.AbstractVelND":  # noqa: N802
        """Transpose the vector.

        The last axis is interpreted as the feature axis. The matrix
        transpose is performed on the last two batch axes.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianVelND.from_([[[1, 2]], [[3, 4]]], "m/s")
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

        >>> vec = cx.vecs.CartesianVelND.from_([[1, 2], [3, 4]], "m/s")
        >>> vec.shape
        (2,)

        """
        return self.q.shape[:-1]

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianVelND.from_([[[1, 2]], [[3, 4]]], "m/s")
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
        """Flatten the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianVelND.from_([[1, 2], [3, 4]], "m/s")
        >>> vec.flatten()
        CartesianVelND(q=Quantity([[1, 2],
                                   [3, 4]], unit='m / s'))

        """
        return replace(self, q=self.q.reshape(-1, self.q.shape[-1]))

    def reshape(self, *shape: Any, order: str = "C") -> "Self":
        """Reshape the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3], "m/s"))
        >>> vec.shape
        ()

        >>> vec.reshape(1, 1).shape
        (1, 1)

        """
        return replace(self, q=self.q.reshape(*shape, self.q.shape[-1], order=order))


#####################################################################


class AbstractAccND(AbstractAcc):
    """Abstract representation of N-D vector differentials."""

    q: eqx.AbstractVar[ct.BBtAcc]

    # ===============================================================
    # Array API

    @property
    def mT(self) -> "coordinax.vecs.AbstractAccND":  # noqa: N802
        """Transpose the vector.

        The last axis is interpreted as the feature axis. The matrix
        transpose is performed on the last two batch axes.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianAccND(u.Quantity([[[1, 2, 3]],
        ...                                          [[4, 5, 6]]], "m/s2"))
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
        >>> import unxt as u
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianAccND(u.Quantity([[[1, 2, 3]],
        ...                                          [[4, 5, 6]]], "m/s2"))
        >>> vec.shape
        (2, 1)

        """
        return self.q.shape[:-1]

    @property
    def T(self) -> "coordinax.vecs.AbstractAccND":  # noqa: N802
        """Transpose the vector's batch axes, preserving the feature axis.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianAccND.from_([[[1, 2, 3]], [[4, 5, 6]]], "m/s2")
        >>> vec.shape
        (2, 1)
        >>> vec.T.shape
        (1, 2)

        """
        return replace(
            self,
            q=qlax.transpose(self.q, (*range(self.ndim)[::-1], self.ndim)),
        )

    # ===============================================================
    # Further array methods

    def flatten(self) -> "coordinax.vecs.AbstractAccND":
        """Flatten the vector's batch dimensions, preserving the component axis.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianAccND.from_([[[1, 2, 3]], [[4, 5, 6]]], "m/s2")
        >>> vec.shape
        (2, 1)

        >>> vec.flatten().shape
        (2,)

        """
        return replace(self, q=self.q.reshape(-1, self.q.shape[-1]))

    def reshape(self, *shape: Any, order: str = "C") -> "coordinax.vecs.AbstractAccND":
        """Reshape the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianAccND.from_([1, 2, 3], "m/s2")
        >>> vec.shape
        ()

        >>> vec.reshape(1, 1).shape
        (1, 1)

        """
        return replace(self, q=self.q.reshape(*shape, self.q.shape[-1], order=order))
