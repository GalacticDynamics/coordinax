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

import importlib.util
import os
import subprocess
import sys

import pytest

import coordinax as cx

# Each case below spawns a child that cold-imports astropy, coordinax and the
# whole JAX stack. Beside xdist workers doing the same on a 4-core runner the
# children starve, and `subprocess.run` without a timeout then blocks for as
# long as the job lives: CI showed this file stalling silently at 99% until the
# runner was shut down. So it runs in the dedicated serial job instead, the
# same one the x32 dtype test uses.
pytestmark = pytest.mark.subprocess_heavy

#: Six children at ~2s each when idle; this only has to be far enough above
#: that to distinguish a slow runner from a stalled one, while still failing
#: long before the job is killed.
_IMPORT_TIMEOUT_S = 300

#: Trim what the child sets up: it only imports, never computes, so JAX's
#: threads and accelerator discovery are pure contention. See `_X32_ENV` in
#: `packages/coordinaxs.hypothesis/.../test_jax_dtypes.py` for the measurements.
_CHILD_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

# This is the single most important behavioural test in the packaging overhaul,
# and a module-level `importorskip` would let a CI job that is *supposed* to
# verify order-independence report green with zero signal if the extra ever
# stopped being installed. So the sanctioned test session (the nox `test`
# session, which installs `--extra workspace`) sets
# ``COORDINAX_REQUIRE_INTEROP_TESTS=1``: when set, a missing extra is a hard
# error rather than a silent skip. Ad-hoc minimal-install runs (without the var)
# still skip gracefully.
_REQUIRE_INTEROP = os.environ.get("COORDINAX_REQUIRE_INTEROP_TESTS") == "1"

for _pkg in ("coordinaxs.astro", "coordinaxs.interop.astropy"):
    # `find_spec` *raises* ModuleNotFoundError (rather than returning None) when
    # a parent namespace is absent — e.g. `coordinaxs.interop` when only the
    # astro extra is installed — so treat that as "not installed" too. Any other
    # missing module is a genuine packaging/runtime failure: let it propagate.
    try:
        _spec = importlib.util.find_spec(_pkg)
    except ModuleNotFoundError as exc:
        if not f"{_pkg}.".startswith(f"{exc.name}."):
            raise
        _spec = None
    if _spec is None:
        if _REQUIRE_INTEROP:
            msg = (
                f"{_pkg} is not installed, but COORDINAX_REQUIRE_INTEROP_TESTS=1 "
                "requires the interop order-independence tests to run. Install "
                "the `workspace` extra."
            )
            raise RuntimeError(msg)
        pytest.skip(f"{_pkg} not installed", allow_module_level=True)

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
    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", f"{first_import}\n{_CHECK}"],
            env={**os.environ, **_CHILD_ENV},
            capture_output=True,
            text=True,
            timeout=_IMPORT_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # Without this the call blocks until the job is killed, and the failure
        # reads as a dead runner rather than as this test.
        msg = (
            f"`{first_import}` first: the child did not finish importing in "
            f"{_IMPORT_TIMEOUT_S}s. It costs ~2s idle, so this is a stall."
        )
        raise AssertionError(msg) from exc
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
