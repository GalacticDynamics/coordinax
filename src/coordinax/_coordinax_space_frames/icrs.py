"""Astronomy reference frames."""

__all__ = ["ICRS"]


from typing import TypeAlias, final

from jaxtyping import Array, Shaped

import unxt as u

from coordinax._src.distances import Distance
from coordinax._src.frames import AbstractReferenceFrame

RotationMatrix: TypeAlias = Shaped[Array, "3 3"]
LengthVector: TypeAlias = Shaped[u.Quantity["length"], "3"] | Shaped[Distance, "3"]


@final
class ICRS(AbstractReferenceFrame):
    """The International Celestial Reference System (ICRS).

    Examples
    --------
    >>> import coordinax as cx
    >>> frame = cx.frames.ICRS()
    >>> frame
    ICRS()

    """
