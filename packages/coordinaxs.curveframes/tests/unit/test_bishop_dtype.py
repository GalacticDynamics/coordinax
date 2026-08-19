"""`BishopBuilder` promotes integers to float without overriding f32.

``dtype=float`` names the *default* float, so under ``jax_enable_x64`` it
widened an f32 input to f64, discarding a precision the caller chose. Asserted
as the promotion contract rather than by toggling x64, so it holds either way
round -- and ``jax.experimental.enable_x64`` no longer exists.

Tested at `_float` rather than through `BishopBuilder`: the builder's output
dtype is set by the curve and the ODE solver, which promote to f64 under x64
whatever the ``initial_normal`` was, so an end-to-end assertion would be
testing diffrax rather than this contract.
"""

import jax.numpy as jnp
import pytest

from coordinaxs.curveframes._src.bishop import _float


@pytest.mark.parametrize(
    "given",
    [
        jnp.asarray([0, 1, 0]),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, 0.0]),
    ],
    ids=["int", "f32", "default-float"],
)
def test_float_follows_result_type(given):
    assert _float(given).dtype == jnp.result_type(given, float)


def test_an_integer_input_becomes_floating():
    """Integer arithmetic in the transport ODE would be a trap."""
    assert jnp.issubdtype(_float(jnp.asarray([0, 1, 0])).dtype, jnp.floating)


def test_a_python_list_still_works():
    """Regression: `result_type` cannot read a list, `asarray` must run first."""
    assert _float([0.0, 1.0, 0.0]).shape == (3,)
