"""Interop registration must not depend on import order.

`coordinax` is a regular package whose ``__init__`` loads the
``coordinaxs.interop`` entry-point group, and the astropy interop references
`coordinaxs.astro` types while astro imports `coordinax.frames`. Importing
astro first therefore runs core's interop loader while astro is only partially
initialized, and the registration has to be completed afterwards rather than
dropped.

These tests must run in a *subprocess*: the session `conftest.py` preloads
`coordinax` before any `coordinaxs.*` package, so an in-process test can never
observe the astro-first ordering that this guards.
"""

import subprocess
import sys

import pytest

import coordinax as cx

pytest.importorskip("coordinaxs.astro")
pytest.importorskip("coordinaxs.interop.astropy")

# Each case imports a different module *first*, then asserts that an
# astropy->coordinax conversion registered by the interop package works.
_FIRST_IMPORTS = [
    "import coordinaxs.astro",
    "import coordinaxs.hypothesis.astro",
    "import coordinax",
    "import coordinax.angles",
    "import coordinax.frames",
    "import coordinaxs.interop.astropy",
]

_CHECK = """
import sys
from plum import convert
import astropy.units as apyu
import coordinaxs.astro as cxastro

assert "coordinaxs.interop.astropy" in sys.modules, "interop was never registered"
out = convert(apyu.Quantity(1.0, "mas"), cxastro.Parallax)
assert isinstance(out, cxastro.Parallax), out
print("OK")
"""


@pytest.mark.parametrize("first_import", _FIRST_IMPORTS)
def test_interop_registers_regardless_of_import_order(first_import: str) -> None:
    """Astropy conversions register no matter which module is imported first."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", f"{first_import}\n{_CHECK}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"`{first_import}` first left interop unregistered:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_interop_loader_is_idempotent() -> None:
    """Repeated loader calls neither duplicate work nor raise."""
    loaded = cx._OPTIONAL_INTEROP_STATE["loaded"]
    assert "astropy" in loaded, "astropy interop entry point should be loaded"

    before = set(loaded)
    cx._load_optional_interop()
    cx._load_optional_interop()
    assert set(cx._OPTIONAL_INTEROP_STATE["loaded"]) == before
