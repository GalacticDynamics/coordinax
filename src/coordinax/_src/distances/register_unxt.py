"""Register Distance support for `unxt`."""
# pylint: disable=import-error

__all__: list[str] = []


from plum import dispatch

import unxt as u

from .base import AbstractDistance


@dispatch
def dimension_of(obj: type[AbstractDistance], /) -> u.dims.AbstractDimension:
    """Get the dimension of an angle.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> u.dimension_of(cx.distance.AbstractDistance)
    PhysicalType('length')

    """
    return u.dimension("length")
