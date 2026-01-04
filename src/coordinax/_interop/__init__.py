"""Interoperability."""

__all__: tuple[str, ...] = ()

import contextlib

from .optional_deps import OptDeps


def _register_interop_packages() -> None:
    """Register interoperability packages after coordinax is fully loaded.

    This function is called after the coordinax module finishes loading
    to avoid circular import issues. Interop packages just register
    plum dispatches and don't need to be imported during coordinax init.
    """
    if OptDeps.COORDINAX_INTEROP_ASTROPY.installed:
        with contextlib.suppress(Exception):
            import coordinax_interop_astropy  # noqa: F401, PLC0415  # pylint: disable=E0401,W0611
