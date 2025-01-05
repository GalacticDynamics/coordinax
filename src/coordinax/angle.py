"""`coordinax.angle` module."""

__all__ = ["AbstractAngle", "Angle", "Parallax"]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.angle", RUNTIME_TYPECHECKER):
    from ._src.angles import AbstractAngle, Angle
    from ._src.distances import Parallax


del RUNTIME_TYPECHECKER, install_import_hook
