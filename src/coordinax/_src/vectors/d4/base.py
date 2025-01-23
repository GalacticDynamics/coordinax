"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos4D"]


from typing_extensions import override

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector
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

    @override
    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:  # type: ignore[override]
        raise NotImplementedError  # pragma: no cover
