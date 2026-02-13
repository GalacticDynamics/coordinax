"""Package setup information.

Note that this module is NOT public API nor are any of its contents.
Stability is NOT guaranteed.
This module exposes package setup information for the `unxt` package.

"""

__all__: tuple[str, ...] = ("RUNTIME_TYPECHECKER", "install_import_hook", "OptDeps")

import contextlib
import os
from importlib.metadata import entry_points

from collections.abc import Sequence
from jaxtyping import install_import_hook as _install_import_hook
from typing import Any, Final, Literal, final

from optional_dependencies import OptionalDependencyEnum, auto

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


@final
class OptDeps(OptionalDependencyEnum):  # pylint: disable=invalid-enum-extension
    """Optional dependencies for ``coordinax``."""

    COORDINAX_INTEROP_ASTROPY = auto()


def register_interop_packages() -> None:
    """Register interoperability packages after coordinax is fully loaded.

    This function discovers and imports all installed ``coordinax.interop.*``
    packages via the ``coordinax.interop`` entry-point group.  Each interop
    package (e.g. ``coordinax-interop-astropy``) declares an entry point
    in its ``pyproject.toml``::

        [project.entry-points."coordinax.interop"]
        astropy = "coordinax.interop.astropy"

    Importing these modules registers their plum dispatches.
    """
    for ep in entry_points(group="coordinax.interop"):
        with contextlib.suppress(Exception):
            ep.load()
