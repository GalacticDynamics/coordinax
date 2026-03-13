"""Type hints for `coordinax.distance`."""

__all__: tuple[str, ...] = ()


from jaxtyping import Shaped

import unxt as u

from .base import AbstractDistance

BBtLength = Shaped[u.Q["length"], "*#batch"]  # type: ignore[type-arg]
BatchableDistance = Shaped[AbstractDistance, "*#batch"]
