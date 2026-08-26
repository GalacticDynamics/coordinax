"""Dtype strategies must agree with the active JAX x64 setting."""

__all__: tuple[str, ...] = ()

import os
import subprocess
import sys
import time

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp
import pytest
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
#: The subprocess is load-bearing, not incidental -- do not replace it with
#: `jax.config.update("jax_enable_x64", False)` in-process. That toggle does
#: change `jnp.empty(0, dtype=float64).dtype`, so an in-process version looks
#: like it works; but `SCALAR_DTYPES` is built from `xps.scalar_dtypes()` at
#: import, when x64 was still on, so the float64 draws this guards never
#: appear. Verified by deleting `honoured_dtypes` from `strategy.py` and
#: re-running: the subprocess catches the regression, the in-process form
#: reports 0 and passes.
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
#: The work costs ~3.4s idle. It used to be 600s, sized for a child starved by
#: xdist workers cold-importing the same JAX stack; that is no longer the
#: shape of the run, because the test has a serial job to itself and takes
#: ~22s there end to end. A ceiling far above the real cost only makes a stall
#: slower to report, so this is sized against the job it actually runs in.
_X32_TIMEOUT_S = 120


#: Environment for the child, beyond turning x64 off.
#:
#: The child only *draws dtypes* -- it never computes -- so every thread and
#: backend JAX would set up for it is waste that competes with the rest of the
#: suite. Pinning the platform also skips accelerator discovery, which is pure
#: cost on a CPU runner. Measured on this script: 3.1s -> 1.8s idle, and
#: 3.6-5.1s -> 2.3-2.6s with every core saturated, the spread mattering as much
#: as the mean.
_X32_ENV = {
    "JAX_ENABLE_X64": "0",
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def _text(stream: "bytes | str | None") -> str:
    """Decode a `TimeoutExpired` stream, which is bytes even under `text=True`."""
    return (
        stream.decode(errors="replace") if isinstance(stream, bytes) else stream or ""
    )


def _run_x32() -> "subprocess.CompletedProcess[str]":
    """Draw `charts()` in a fresh x32 interpreter."""
    return subprocess.run(  # noqa: S603  # fixed literal script, this interpreter
        [sys.executable, "-c", _X32_SCRIPT],
        env={**os.environ, **_X32_ENV},
        capture_output=True,
        text=True,
        timeout=_X32_TIMEOUT_S,
        check=False,
    )


@pytest.mark.subprocess_heavy
def test_charts_draws_without_x64() -> None:
    """``charts()`` must not raise `InvalidArgument` when x64 is off.

    Guards the wiring rather than the helper: dropping `honoured_dtypes` from the
    dtype defaults leaves the unit test above passing while this fails.
    """
    started = time.monotonic()
    try:
        proc = _run_x32()
    except subprocess.TimeoutExpired as exc:  # pragma: no cover  - CI-only flake
        # `subprocess.run` raises before the assert below, so a timeout used to
        # surface as a bare `TimeoutExpired` carrying a 600-line repr of the
        # script and nothing about what the child was doing. Re-raise as a
        # failure that says how far it got.
        out, err = _text(exc.stdout), _text(exc.stderr)
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
