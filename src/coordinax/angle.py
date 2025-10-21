"""`coordinax.angle` module."""

__all__ = ["AbstractAngle", "Angle", "wrap_to", "Parallax"]

from .setup_package import install_import_hook

with install_import_hook("coordinax.angle"):
    from unxt.quantity import AbstractAngle, Angle, wrap_to

    from ._src.distances import Parallax


del install_import_hook
