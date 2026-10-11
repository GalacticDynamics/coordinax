"""Test :mod:`coordinax._src.vectors.d4`."""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx


@pytest.mark.parametrize(
    ("t", "q"),
    [
        (u.Q(1.0, "s"), u.Q([1.0, 2, 3], "m")),
        (u.Q(jnp.ones(2), "s"), u.Q(jnp.ones((2, 3)), "m")),
    ],
    ids=["scalar", "batch"],
)
def test_fourvector_time(t: u.Q, q: u.Q) -> None:
    """``t`` takes a scalar or batched time."""
    w = cx.vecs.FourVector(t=t, q=q)
    assert jnp.array_equal(w.t.value, t.value)


def test_fourvector_time_wrong_dimension() -> None:
    """``t`` must be a time."""
    with pytest.raises(ValueError, match=r"FourVector\.t must have dimension 'time'"):
        cx.vecs.FourVector(t=u.Q(1.0, "m"), q=u.Q([1.0, 2, 3], "m"))
