"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos4D"]


from coordinax._src.vectors.base_pos import AbstractPos


class AbstractPos4D(AbstractPos):
    """Abstract representation of 4D coordinates in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.AbstractPos4D._dimensionality()
        4

        """
        return 4
