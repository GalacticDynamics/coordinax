"""Test :mod:`coordinax._src.vectors.d4`."""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx


@pytest.mark.parametrize(
    ("t", "q"),
    [
        (u.Quantity(1.0, "s"), u.Quantity([1.0, 2, 3], "m")),
        (u.Quantity(jnp.ones(2), "s"), u.Quantity(jnp.ones((2, 3)), "m")),
    ],
    ids=["scalar", "batch"],
)
def test_fourvector_time(t: u.Quantity, q: u.Quantity) -> None:
    """``t`` takes a scalar or batched time."""
    w = cx.vecs.FourVector(t=t, q=q)
    assert jnp.array_equal(w.t.value, t.value)
    assert w.dimensions["t"] == u.dimension("time")


def test_fourvector_time_wrong_dimension() -> None:
    """``t`` must be a time."""
    with pytest.raises(ValueError, match="Physical type mismatch"):
        cx.vecs.FourVector(t=u.Quantity(1.0, "m"), q=u.Quantity([1.0, 2, 3], "m"))
