"""Representation of coordinates in different systems."""

__all__ = ["Abstract4DVector"]


from abc import abstractmethod
from typing import TYPE_CHECKING

from coordinax._base import AbstractVector
from coordinax._base_pos import AbstractPosition
from coordinax._utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Never


class Abstract4DVector(AbstractPosition):
    """Abstract representation of 4D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> "Never":  # type: ignore[override]
        msg = "Not yet implemented"
        raise NotImplementedError(msg)
