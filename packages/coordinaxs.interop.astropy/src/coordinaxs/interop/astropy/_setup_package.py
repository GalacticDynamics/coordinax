"""Runtime type-checking import hook for ``coordinaxs.interop.astropy``.

Not public API. Mirrors ``coordinaxs.astro._setup_package``; see
``coordinax._src.setup_package`` for the meaning of
``COORDINAX_ENABLE_RUNTIME_TYPECHECKING``.

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


def install_import_hook(
    modules: str | Sequence[str], /
) -> contextlib.AbstractContextManager[Any, None]:
    """Install the jaxtyping import hook for ``modules``; a no-op when disabled."""
    return (
        _install_import_hook(modules, RUNTIME_TYPECHECKER)
        if RUNTIME_TYPECHECKER is not False
        else contextlib.nullcontext()
    )
