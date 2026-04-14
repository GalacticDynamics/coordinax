"""`coordinax.distances` module."""

__all__ = ("AbstractDistance", "Distance")

from ._setup_package import install_import_hook

with install_import_hook("coordinax.distances"):
    from ._src import AbstractDistance, Distance


del install_import_hook
