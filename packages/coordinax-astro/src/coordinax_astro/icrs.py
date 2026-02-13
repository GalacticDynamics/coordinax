"""Astronomy reference frames."""

__all__ = ("ICRS",)


from jaxtyping import Array, Shaped
from typing import TypeAlias, final

import unxt as u

from .base import AbstractSpaceFrame
from coordinax.distances import Distance

RotationMatrix: TypeAlias = Shaped[Array, "3 3"]
LengthVector: TypeAlias = Shaped[u.Q["length"], "3"] | Shaped[Distance, "3"]  # type: ignore[type-arg]


@final
class ICRS(AbstractSpaceFrame):
    """The International Celestial Reference System (ICRS).

    Examples
    --------
    >>> import coordinax as cx
    >>> frame = cxf.ICRS()
    >>> frame
    ICRS()

    """
