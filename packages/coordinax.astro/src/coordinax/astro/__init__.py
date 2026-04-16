"""``import coordinax.astro as cxastro`` — Frames for Astronomy."""

__all__ = (
    "Parallax",
    "DistanceModulus",
    "AbstractSpaceFrame",
    "ICRS",
    "icrs",
    "Galactocentric",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.astro"):
    from ._src import (
        ICRS,
        AbstractSpaceFrame,
        DistanceModulus,
        Galactocentric,
        Parallax,
        icrs,
    )


# Populate optional exports into `coordinax.frames` once astro symbols exist.
import coordinax.frames as cxf

cxf._load_optional_frame_exports()

del cxf, install_import_hook
