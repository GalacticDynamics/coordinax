"""`coordinax.angle` module."""

__all__ = ("AbstractAngle", "Angle", "wrap_to", "Parallax")

from .setup_package import install_import_hook

# TODO: this doesn't actually trigger jaxtyping on these imports; fix it.
with install_import_hook("coordinax.angles"):
    from unxt.quantity import AbstractAngle, Angle, wrap_to

    from ._src.distances import Parallax


del install_import_hook
