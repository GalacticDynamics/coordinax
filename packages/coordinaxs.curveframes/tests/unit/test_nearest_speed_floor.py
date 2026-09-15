"""The residual's speed floor must scale with the data's dtype, not the global one.

`jnp.finfo(jnp.zeros(()).dtype)` reads the *default* float. Under this repo's
``JAX_ENABLE_X64=1`` that is f64 even when the curve data is f32, which floored
the residual's divisor ~2.3e4x too low to clamp f32 rounding noise.

Tested on the helper rather than through a solve: `safe_speed` is positive under
either floor, so the residual's sign -- and therefore the root every bracket
test and the bisection converge on -- is identical. The floor moves conditioning,
never the answer, so no `nearest_tau` query distinguishes the two.
"""

import jax
import jax.numpy as jnp
import pytest

from coordinaxs.curveframes._src.nearest import _relative_speed_floor


def test_the_tests_run_in_the_regime_the_bug_needs() -> None:
    """Without x64 the default float *is* f32 and the bug cannot appear."""
    assert jax.config.jax_enable_x64


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_the_floor_is_read_from_the_data_not_the_default_float(dtype) -> None:
    fallback = jnp.asarray(2.0, dtype=dtype)
    floor = _relative_speed_floor(fallback)

    assert floor.dtype == dtype
    expected = float(jnp.sqrt(jnp.finfo(dtype).eps)) * 2.0
    assert float(floor) == pytest.approx(expected, rel=1e-6)


def test_the_f32_floor_is_big_enough_to_clamp_f32_noise() -> None:
    """The regression guard: reading f64's eps here gives 2.98e-08, not 6.9e-04.

    A speed of 2e-06 is pure f32 rounding noise and must not survive the floor
    and reach the division.
    """
    floor = _relative_speed_floor(jnp.asarray(2.0, dtype=jnp.float32))
    assert float(floor) > 1e-4
    assert float(jnp.maximum(jnp.asarray(2e-06, dtype=jnp.float32), floor)) == float(
        floor
    )
