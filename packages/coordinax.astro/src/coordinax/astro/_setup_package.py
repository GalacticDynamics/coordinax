"""Package setup information.

Note that this module is NOT public API nor are any of its contents.
Stability is NOT guaranteed.
This module exposes package setup information for the `unxt` package.

"""

__all__: tuple[str, ...] = ()

import contextlib
import os

from collections.abc import Sequence
from jaxtyping import install_import_hook as _install_import_hook
from typing import Any, Final, Literal

_RUNTIME_TYPECHECKER: str | None | Literal[False]
match os.getenv("COORDINAX_ENABLE_RUNTIME_TYPECHECKING", "False"):
    case "False":
        _RUNTIME_TYPECHECKER = False
    case "None":
        _RUNTIME_TYPECHECKER = None
    case str() as _name:
        _RUNTIME_TYPECHECKER = _name

RUNTIME_TYPECHECKER: Final[str | None | Literal[False]] = _RUNTIME_TYPECHECKER
"""Runtime type checking variable "COORDINAX_ENABLE_RUNTIME_TYPECHECKING".

Set to "False" to disable runtime typechecking (default).
Set to "None" to only enable typechecking for `@jaxtyped`-decorated functions.
Set to "beartype.beartype" to enable runtime typechecking.

See https://docs.kidger.site/jaxtyping/api/runtime-type-checking for more
information on options.


"""


def install_import_hook(
    modules: str | Sequence[str], /
) -> contextlib.AbstractContextManager[Any, None]:
    """Install the jaxtyping import hook for the given modules.

    Parameters
    ----------
    modules
        Module name or sequence of module names to install the import hook for.

    Returns
    -------
    contextlib.AbstractContextManager
        Context manager that installs the import hook on entry and removes it on exit.

    """
    return (
        _install_import_hook(modules, RUNTIME_TYPECHECKER)
        if RUNTIME_TYPECHECKER is not False
        else contextlib.nullcontext()
    )


def coordinax_frames_exports() -> dict[str, object]:
    """Return frame symbols exported to ``coordinax.frames`` via entry-point."""
    try:
        from ._src.base_frame import AbstractSpaceFrame  # noqa: PLC0415
        from ._src.galactocentric import Galactocentric  # noqa: PLC0415
        from ._src.icrs import ICRS, icrs  # noqa: PLC0415
    except ImportError as exc:
        # During `import coordinax.astro`, this provider may run while
        # `coordinax.astro._src.base_frame` is still being initialized.
        # Returning no exports here avoids failing the import cycle; exports
        # are loaded again once `coordinax.astro` initialization completes.
        if "partially initialized module" in str(exc):
            return {}
        raise

    return {
        "AbstractSpaceFrame": AbstractSpaceFrame,
        "ICRS": ICRS,
        "icrs": icrs,
        "Galactocentric": Galactocentric,
    }
