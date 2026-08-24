"""Dtype strategies must agree with the active JAX x64 setting."""

__all__: tuple[str, ...] = ()

import os
import subprocess
import sys
import time

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import find, settings
from hypothesis.errors import NoSuchExample

from coordinaxs.hypothesis.utils._src.annotations.dtypes import honoured_dtypes

#: Whether this interpreter honours 64-bit dtypes. The suite sets
#: ``JAX_ENABLE_X64=1``; downstream users of these strategies often do not.
X64_ENABLED = jnp.empty(0, dtype=jnp.float64).dtype == jnp.dtype(jnp.float64)


def _is_findable(dtypes: st.SearchStrategy[Any], target: Any, /) -> bool:
    """Whether *target* can be drawn from *dtypes* at all."""
    try:
        find(
            dtypes,
            lambda dt: jnp.dtype(dt) == jnp.dtype(target),
            settings=settings(database=None),
        )
    except NoSuchExample:
        return False
    return True


def test_float64_offered_iff_x64() -> None:
    """``float64`` is reachable exactly when the runtime honours it.

    Under x64 the filter must not over-prune; without it float64 must be gone.
    """
    dtypes = honoured_dtypes(st.sampled_from([jnp.float32, jnp.float64]))
    assert _is_findable(dtypes, jnp.float32)
    assert _is_findable(dtypes, jnp.float64) is X64_ENABLED


#: Draws `charts()` in a fresh interpreter, printing the count of hard errors.
#: Out-of-process: the x64 setting is read once, at JAX import.
#:
#: This is the one place `deadline=None` is still written by hand. The
#: subprocess runs `python -c`, which never loads the root ``conftest.py``, so
#: the profile registered there -- the reason no other test needs it -- does
#: not reach this interpreter.
_X32_SCRIPT = """
import warnings
warnings.filterwarnings("ignore")
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.errors import InvalidArgument
import jax.numpy as jnp
assert jnp.empty(0, dtype=jnp.float64).dtype == jnp.dtype(jnp.float32), "x64 leaked in"
import coordinaxs.hypothesis.main as cxst

bad = []

@given(d=st.data())
@settings(max_examples=150, deadline=None, database=None,
          suppress_health_check=list(HealthCheck))
def run(d):
    try:
        d.draw(cxst.charts())
    except InvalidArgument as exc:
        bad.append(str(exc)[:120])

import sys as _sys
print("imported", file=_sys.stderr, flush=True)
run()
print(len(bad))
print("\\n".join(bad[:3]))
"""


#: Wall-clock ceiling for the subprocess below.
#:
#: The work costs ~3.4s on an idle machine and ~4.4s with all ten logical cores
#: saturated, so this is a ~130x margin, not a tight bound. It has nonetheless
#: been hit on CI -- on several PRs at once, including ones touching neither
#: `charts()` nor anything it imports, which is what marks it as a flake rather
#: than a regression.
#:
#: Ruled out, each by measurement rather than reasoning:
#:
#: - CPU contention from `-n logical` (#788). Saturating every core costs ~30%,
#:   not the ~130x a timeout would need.
#: - JAX persistent-compilation-cache locking between workers. No cache is
#:   configured, so there is no lock to contend for.
#: - Memory exhaustion. The child peaks at ~290MB and a pytest worker at
#:   ~297MB, so a four-worker runner sits near 1.5GB against 7GB+ available.
#:
#: The cause is still unknown. Raising the ceiling further would only make the
#: eventual failure slower to arrive, so it stays where it is and the handler
#: below makes the next occurrence say something useful instead.
_X32_TIMEOUT_S = 600


def test_charts_draws_without_x64() -> None:
    """``charts()`` must not raise `InvalidArgument` when x64 is off.

    Guards the wiring rather than the helper: dropping `honoured_dtypes` from the
    dtype defaults leaves the unit test above passing while this fails.
    """
    started = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603  # fixed literal script, this interpreter
            [sys.executable, "-c", _X32_SCRIPT],
            env={**os.environ, "JAX_ENABLE_X64": "0"},
            capture_output=True,
            text=True,
            timeout=_X32_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover  - CI-only flake
        # `subprocess.run` raises before the assert below, so a timeout used to
        # surface as a bare `TimeoutExpired` carrying a 600-line repr of the
        # script and nothing about what the child was doing. Re-raise as a
        # failure that says how far it got.
        out = (
            (exc.stdout or b"").decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        err = (
            (exc.stderr or b"").decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        stage = (
            "while drawing charts()"
            if "imported" in err
            else "before importing jax/coordinaxs"
        )
        msg = (
            f"the x32 subprocess exceeded {_X32_TIMEOUT_S}s, {stage}.\n"
            f"It costs ~3.4s idle and ~4.4s with every core saturated, so a "
            f"timeout here is a stall, not slow work -- see the note above "
            f"`_X32_TIMEOUT_S` for what has already been ruled out.\n"
            f"elapsed={time.monotonic() - started:.1f}s\n"
            f"child stdout: {out[-1000:]!r}\n"
            f"child stderr: {err[-2000:]!r}"
        )
        raise AssertionError(msg) from exc
    assert proc.returncode == 0, proc.stderr[-3000:]
    n_bad, _, detail = proc.stdout.strip().partition("\n")
    assert int(n_bad) == 0, f"{n_bad} InvalidArgument draws:\n{detail}"
