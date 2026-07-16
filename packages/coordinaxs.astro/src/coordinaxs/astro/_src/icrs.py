"""Astronomy reference frames."""

__all__ = ("ICRS", "icrs")


from typing import final

from .base_frame import AbstractSpaceFrame


@final
class ICRS(AbstractSpaceFrame):
    """The International Celestial Reference System (ICRS).

    Examples
    --------
    >>> import coordinaxs.astro as cxastro
    >>> frame = cxastro.ICRS()
    >>> frame
    ICRS()

    """


icrs = ICRS()  # canonical instance for convenience
