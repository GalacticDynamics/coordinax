"""``import coordinax.astro as cxastro`` — Frames for Astronomy."""

__all__ = (
    "Parallax",
    "DistanceModulus",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.astro"):
    from ._src import (
        DistanceModulus,
        Parallax,
    )


del install_import_hook
