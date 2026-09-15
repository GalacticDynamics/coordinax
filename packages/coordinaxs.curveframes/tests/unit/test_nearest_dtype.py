"""`nearest_tau` must read the dtype it is *working in*, not the default float.

Two constants were derived from ``jnp.finfo(jnp.zeros(()).dtype)``, which names
the global default. Under this repo's ``JAX_ENABLE_X64=1`` that is f64 even when
the curve data is f32, so both came out sized for a precision the data does not
have: the residual's speed floor (below) and the solver tolerance (bottom).

The two differ in how much rides on them, which is why they are tested at
different levels. The **floor** is latent: ``safe_speed`` is positive under
either value, so the residual's sign -- and therefore the root every bracket
test and the bisection converge on -- is identical, and no query distinguishes
them. It is tested on the helper directly. The **tolerance** is not latent: a
tolerance below the working dtype's resolution can never be met, and the solve
refuses instead of returning its already-correct answer -- measured, 15 of 19
f32 circle queries. That one is tested through `nearest_tau` itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc
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


# --------------------------------------------------------------------------
# The solver tolerance.


def _circle(dtype):
    def curve(tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = jnp.asarray(tau.ustrip("s"), dtype=dtype)
        return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    return curve


def _solve(dtype, theta: float) -> float:
    builder = cxfc.FrenetSerretBuilder(_circle(dtype), "s")
    bounds = (u.Q(jnp.asarray(0.0, dtype), "s"), u.Q(jnp.asarray(1.0, dtype), "s"))
    x = u.Q(
        jnp.asarray([1.5 * np.cos(theta), 1.5 * np.sin(theta), 0.0], dtype=dtype), "km"
    )
    return float(cxfc.nearest_tau(builder, x, bounds=bounds).ustrip("s"))


#: A probe at radius 1.5 and angle ``theta`` is nearest the unit circle at
#: ``tau = theta``, so the truth needs no solve of its own.
THETAS = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]


@pytest.mark.parametrize("theta", THETAS)
def test_f32_queries_are_not_refused(theta: float) -> None:
    """A tolerance below f32's own resolution can never be met.

    With ``tol`` read from the global (f64) default, 15 of 19 such queries
    raised rather than returning an already-correct answer.
    """
    got = _solve(jnp.float32, theta)
    # The solve is only asked to be good to its own `sqrt(eps)`; measured 1.8e-4.
    assert got == pytest.approx(theta, abs=float(jnp.sqrt(jnp.finfo(jnp.float32).eps)))


@pytest.mark.parametrize("theta", THETAS)
def test_f64_keeps_its_full_precision(theta: float) -> None:
    """The f32 fix must not loosen f64: measured identical before and after."""
    assert _solve(jnp.float64, theta) == pytest.approx(theta, abs=1e-8)
