"""Representation of coordinates in different systems."""

__all__ = ["AbstractAcc2D", "AbstractPos2D", "AbstractVel2D"]


from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel


class AbstractPos2D(AbstractPos):
    """Abstract representation of 2D coordinates in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos2D._dimensionality()
        2

        """
        return 2


class AbstractVel2D(AbstractVel):
    """Abstract representation of 2D vector differentials."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianVel2D._dimensionality()
        2

        """
        return 2


class AbstractAcc2D(AbstractAcc):
    """Abstract representation of 2D vector accelerations."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianAcc2D._dimensionality()
        2

        """
        return 2
