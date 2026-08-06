"""Dtype strategies must agree with the active JAX x64 setting.

``jax.numpy`` declares the 64-bit dtypes unconditionally, but with
``jax_enable_x64`` off it narrows them: ``jnp.asarray(x, dtype=float64)`` returns
a float32 array. Hypothesis then finds the built array does not match the dtype
it asked for and raises `InvalidArgument` -- a hard error, not a discard.
"""

__all__: tuple[str, ...] = ()

import os
import subprocess
import sys

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis.extra.array_api import make_strategies_namespace

from coordinaxs.hypothesis.utils._src.annotations.dtypes import jax_honoured

xps = make_strategies_namespace(jnp)

#: Whether this interpreter honours 64-bit dtypes. The suite sets
#: ``JAX_ENABLE_X64=1``; downstream users of these strategies often do not.
X64 = jnp.empty(0, dtype=jnp.float64).dtype == jnp.dtype(jnp.float64)


def test_filter_tracks_the_runtime_not_the_declaration() -> None:
    """``float64`` survives the filter exactly when the runtime honours it.

    Meaningful under either setting: with x64 on, float64 must still be drawn
    (the filter must not over-prune); with it off, it must be dropped.
    """
    drawn: set[Any] = set()

    @given(dtype=jax_honoured(st.sampled_from([jnp.float32, jnp.float64])))
    @settings(max_examples=50, deadline=None, database=None)
    def collect(dtype: Any) -> None:
        drawn.add(jnp.dtype(dtype))

    collect()
    assert jnp.dtype(jnp.float32) in drawn
    assert (jnp.dtype(jnp.float64) in drawn) is X64


@given(dtype=jax_honoured(xps.scalar_dtypes()))
@settings(max_examples=100, deadline=None)
def test_every_offered_dtype_round_trips(dtype: Any) -> None:
    """An array built with a drawn dtype comes back with that same dtype.

    This is the property whose violation hypothesis reports as
    ``Could not create array via xp.asarray(..., dtype=...)``.
    """
    assert jnp.empty(0, dtype=dtype).dtype == jnp.dtype(dtype)


#: Draws `charts()` in a fresh interpreter and reports how many draws died.
#:
#: Run out-of-process because the setting is read once, at JAX import, and this
#: suite has already imported JAX with x64 on.
_X32_DRAW = """
import warnings
warnings.filterwarnings("ignore")
from hypothesis import given, settings, HealthCheck, strategies as st
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
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ == "InvalidArgument":
            bad.append(str(exc)[:120])

run()
print(len(bad))
print("\\n".join(bad[:3]))
"""


def test_charts_draws_without_x64() -> None:
    """``charts()`` must not raise `InvalidArgument` when x64 is off.

    Guards the wiring, not just the helper: dropping `jax_honoured` from the
    dtype defaults would leave the unit tests above passing while this fails.
    Measured 43 failures in 200 draws before the filter was applied.
    """
    proc = subprocess.run(  # noqa: S603  # fixed literal script, this interpreter
        [sys.executable, "-c", _X32_DRAW],
        env={**os.environ, "JAX_ENABLE_X64": "0"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    n_bad, _, detail = proc.stdout.strip().partition("\n")
    assert int(n_bad) == 0, f"{n_bad} InvalidArgument draws:\n{detail}"
