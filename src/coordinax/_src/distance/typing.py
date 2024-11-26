"""Type hints for `coordinax.distance`."""

__all__: list[str] = []


from jaxtyping import Shaped

import unxt as u

from .base import AbstractDistance

BatchableLength = Shaped[u.Quantity["length"], "*#batch"]
BatchableDistance = Shaped[AbstractDistance, "*#batch"]
