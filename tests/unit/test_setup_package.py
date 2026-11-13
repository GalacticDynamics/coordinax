"""How each package reads ``COORDINAX_ENABLE_RUNTIME_TYPECHECKING``.

The suite always runs with the variable set to a checker name, so the other
two spellings are only exercised here, by re-executing each setup module under
a patched environment. The subprocess guard in
``tests/repo/test_runtime_typechecking_hook.py`` checks the hook itself.
"""

import contextlib
import importlib

import pytest

SETUP_MODULES = (
    "coordinax._src.setup_package",
    "coordinaxs.astro._setup_package",
    "coordinaxs.curveframes._setup_package",
    "coordinaxs.interop.astropy._setup_package",
)


@pytest.mark.parametrize("modname", SETUP_MODULES)
@pytest.mark.parametrize(
    ("value", "expected"),
    [("False", False), ("None", None), ("beartype.beartype", "beartype.beartype")],
)
def test_env_var_is_parsed(
    monkeypatch: pytest.MonkeyPatch, modname: str, value: str, expected: object
) -> None:
    mod = importlib.import_module(modname)
    monkeypatch.setenv("COORDINAX_ENABLE_RUNTIME_TYPECHECKING", value)
    try:
        importlib.reload(mod)
        assert expected == mod.RUNTIME_TYPECHECKER
        # Disabled means a no-op context manager, not jaxtyping's hook.
        hook = mod.install_import_hook("coordinax_no_such_module")
        assert isinstance(hook, contextlib.nullcontext) is (expected is False)
    finally:
        monkeypatch.undo()
        importlib.reload(mod)
