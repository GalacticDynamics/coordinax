"""Test configuration for coordinax tests."""

__all__ = ("PUBLIC_SUBPACKAGES", "import_public_subpackages")

import importlib

#: The public subpackages. Importing `coordinax` alone does not pull these in,
#: so anything that inspects the dispatch tables must import them first, or it
#: silently inspects whichever ones another test happened to load.
PUBLIC_SUBPACKAGES = (
    "angles",
    "charts",
    "distances",
    "frames",
    "manifolds",
    "representations",
    "transforms",
    "vectors",
)


def import_public_subpackages() -> None:
    """Import every public subpackage, for the dispatches they register."""
    for name in PUBLIC_SUBPACKAGES:
        importlib.import_module(f"coordinax.{name}")
