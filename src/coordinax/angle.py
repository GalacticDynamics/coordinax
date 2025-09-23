"""`coordinax.angle` module."""
# ruff: noqa: PLC0414

__all__ = ["AbstractAngle", "Angle", "wrap_to", "Parallax"]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.angle", RUNTIME_TYPECHECKER):
    from unxt.quantity import (
        AbstractAngle as AbstractAngle,
        Angle as Angle,
        wrap_to as wrap_to,
    )

    from ._src.distances import Parallax


del RUNTIME_TYPECHECKER, install_import_hook
