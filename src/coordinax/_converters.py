"""Representation of coordinates in different systems."""

__all__: list[str] = []


from unxt import Quantity

from coordinax._typing import BatchableAngle

_2pid = Quantity(360, "deg")


def converter_azimuth_to_range(phi: BatchableAngle) -> BatchableAngle:
    """Wrap the polar angle to the range [0, 2pi).

    It's safe to do this conversion since this is a phase cut, unlike `theta`,
    which is only on half the sphere.
    """
    return phi % _2pid
