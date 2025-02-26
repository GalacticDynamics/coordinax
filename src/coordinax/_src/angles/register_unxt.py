"""Register Angle support for `unxt`."""
# pylint: disable=import-error

__all__: list[str] = []


from plum import dispatch

import unxt as u

from .base import AbstractAngle


@dispatch
def dimension_of(obj: type[AbstractAngle], /) -> u.dims.AbstractDimension:
    """Get the dimension of an angle.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> u.dimension_of(cx.angle.Angle)
    PhysicalType('angle')

    """
    return u.dimension("angle")
