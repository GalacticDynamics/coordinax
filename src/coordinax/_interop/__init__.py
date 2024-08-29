"""Interoperability."""

__all__: list[str] = []

from . import optional_deps

if optional_deps.HAS_ASTROPY:
    from . import coordinax_interop_astropy  # noqa: F401
