"""Type hints for {mod}`coordinax.distances`."""

__all__: tuple[str, ...] = ()


from jaxtyping import Shaped

import unxt as u

from .base import AbstractDistance

BBtLength = Shaped[u.Q["length"], "*#batch"]
BatchableDistance = Shaped[AbstractDistance, "*#batch"]
