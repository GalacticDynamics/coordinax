"""Frames for Astronomy."""

__all__ = ("AbstractSpaceFrame", "ICRS", "Galactocentric")

from .setup_package import install_import_hook

with install_import_hook("coordinax_astro"):
    from .base import AbstractSpaceFrame
    from .frame_transforms import *
    from .galactocentric import Galactocentric
    from .icrs import ICRS
