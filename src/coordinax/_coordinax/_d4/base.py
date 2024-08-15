"""Representation of coordinates in different systems."""

__all__ = ["AbstractPosition4D"]


from abc import abstractmethod
from typing import TYPE_CHECKING

from coordinax._coordinax.base import AbstractVector
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Never


class AbstractPosition4D(AbstractPosition):
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
