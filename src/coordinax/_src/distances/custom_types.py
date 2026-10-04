"""Type hints for `coordinax.distance`."""

__all__: tuple[str, ...] = ()


from jaxtyping import Shaped

from .base import AbstractDistance
from coordinax._src.custom_types import DimQuantity

BBtLength = Shaped[DimQuantity["length"], "*#batch"]
BatchableDistance = Shaped[AbstractDistance, "*#batch"]
