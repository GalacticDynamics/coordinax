"""Interoperability."""

__all__: tuple[str, ...] = ()

from .optional_deps import OptDeps

if OptDeps.ASTROPY.installed:
    from . import coordinax_interop_astropy  # noqa: F401
