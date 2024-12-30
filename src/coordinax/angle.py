"""`coordinax.angle` module."""

__all__ = ["AbstractAngle", "Angle", "Parallax"]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.angle", RUNTIME_TYPECHECKER):
    from ._src.angles import AbstractAngle, Angle, Parallax


del RUNTIME_TYPECHECKER, install_import_hook
