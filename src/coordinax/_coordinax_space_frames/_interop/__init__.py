"""Interoperability."""

__all__: list[str] = []

from .optional_deps import OptDeps

if OptDeps.ASTROPY.installed:
    from . import interop_astropy  # noqa: F401
