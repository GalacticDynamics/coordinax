"""Type hints for `coordinax.distance`."""

__all__: tuple[str, ...] = ()


from jaxtyping import Shaped

import unxt as u

from .base import AbstractDistance

BBtLength = Shaped[u.Quantity["length"], "*#batch"]
BatchableDistance = Shaped[AbstractDistance, "*#batch"]
