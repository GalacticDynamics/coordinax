"""Representation of coordinates in different systems."""

__all__ = ["AbstractAcc3D", "AbstractPos3D", "AbstractVel3D"]

from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel


class AbstractPos3D(AbstractPos):
    """Abstract representation of 3D coordinates in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos3D._dimensionality()
        3

        """
        return 3


class AbstractVel3D(AbstractVel):
    """Abstract representation of 3D vector differentials."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianVel3D._dimensionality()
        3

        """
        return 3


class AbstractAcc3D(AbstractAcc):
    """Abstract representation of 3D vector accelerations."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianAcc3D._dimensionality()
        3

        """
        return 3
