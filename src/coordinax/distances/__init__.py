"""`coordinax.distances` module."""

__all__ = ("AbstractDistance", "Distance")

from coordinax._src.setup_package import install_import_hook

with install_import_hook("coordinax.distances"):
    from ._src import AbstractDistance, Distance


del install_import_hook
