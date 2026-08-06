"""Dtype strategies must agree with the active JAX x64 setting."""

__all__: tuple[str, ...] = ()

import os
import subprocess
import sys

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import find, settings
from hypothesis.errors import NoSuchExample

from coordinaxs.hypothesis.utils._src.annotations.dtypes import jax_honoured

#: Whether this interpreter honours 64-bit dtypes. The suite sets
#: ``JAX_ENABLE_X64=1``; downstream users of these strategies often do not.
X64 = jnp.empty(0, dtype=jnp.float64).dtype == jnp.dtype(jnp.float64)


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


def test_filter_tracks_the_runtime_not_the_declaration() -> None:
    """``float64`` survives the filter exactly when the runtime honours it.

    Meaningful under either setting: with x64 on, float64 must still be
    reachable (the filter must not over-prune); with it off, it must be gone.

    `hypothesis.find` rather than counting what `@given` happened to draw --
    the question is whether a dtype is reachable *at all*, and `find` answers
    that by searching rather than by sampling and hoping.
    """
    dtypes = jax_honoured(st.sampled_from([jnp.float32, jnp.float64]))
    assert _is_findable(dtypes, jnp.float32)
    assert _is_findable(dtypes, jnp.float64) is X64


#: Draws `charts()` in a fresh interpreter and prints the count of hard errors.
#:
#: Out-of-process because the x64 setting is read once, at JAX import, and this
#: suite has already imported JAX with it on.
_X32_DRAW = """
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

run()
print(len(bad))
print("\\n".join(bad[:3]))
"""


def test_charts_draws_without_x64() -> None:
    """``charts()`` must not raise `InvalidArgument` when x64 is off.

    Guards the wiring rather than the helper: dropping `jax_honoured` from the
    dtype defaults leaves the unit test above passing while this fails.
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
