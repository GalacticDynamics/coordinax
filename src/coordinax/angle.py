"""`coordinax.angle` module."""

__all__ = ["AbstractAngle", "Angle"]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.angle", RUNTIME_TYPECHECKER):
    from ._src.angles import AbstractAngle, Angle


del RUNTIME_TYPECHECKER, install_import_hook
