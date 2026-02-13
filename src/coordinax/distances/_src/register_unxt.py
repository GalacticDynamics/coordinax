"""Register Distance support for `unxt`."""
# pylint: disable=import-error

__all__: tuple[str, ...] = ()


import plum

import unxt as u

from .base import AbstractDistance
from coordinax._src.constants import LENGTH


@plum.dispatch
def dimension_of(obj: type[AbstractDistance], /) -> u.dims.AbstractDimension:
    """Get the dimension of an angle.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> u.dimension_of(cxd.AbstractDistance)
    PhysicalType('length')

    """
    return LENGTH
