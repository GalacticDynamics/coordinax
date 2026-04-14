"""Package setup: runtime type-checking import hook for ``coordinax.curveframes``.

This module configures `jaxtyping <https://docs.kidger.site/jaxtyping/>`_’s
import hook, which optionally checks array shape/dtype annotations at
runtime.  The behaviour is controlled by the environment variable
``COORDINAX_ENABLE_RUNTIME_TYPECHECKING``:

* ``"False"`` (default) — no runtime checks; ``install_import_hook``
  returns a no-op context manager.
* ``"None"`` — install the hook with ``typechecker=None`` (shape-only
  checks, no beartype).
* Any other string (e.g. ``"beartype.beartype"``) — use that callable as
  the runtime type checker.
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
    """Install the jaxtyping import hook for the given modules.

    When the runtime type checker is enabled, this wraps the given
    module names with jaxtyping's import hook so that array shape
    and dtype annotations are checked at import time.  When disabled
    (the default), returns a no-op context manager.

    Parameters
    ----------
    modules : str or sequence of str
        Module name(s) to instrument.

    Returns
    -------
    contextlib.AbstractContextManager
        A context manager that installs (and later removes) the import
        hook, or a ``nullcontext`` if runtime checking is off.
    """
    return (
        _install_import_hook(modules, RUNTIME_TYPECHECKER)
        if RUNTIME_TYPECHECKER is not False
        else contextlib.nullcontext()
    )
