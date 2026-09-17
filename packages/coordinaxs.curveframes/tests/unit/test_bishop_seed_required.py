"""`BishopBuilder` must not choose the n-plane gauge for you.

The seed is arbitrary *and* load-bearing: a chart's `(n1, n2)` depend on it
wherever the curvature and the offset are both non-zero, so a caller who never
chose one still depended on it. That silence is the defect behind #870, where
a rigid rotation -- an isometry -- reported `|K| = 0.1178` instead of zero.

These pin the breaking half directly, so it cannot drift back to a default.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def test_omitting_the_seed_raises() -> None:
    """Silence is no longer a choice, and the message names both ways out."""
    with pytest.raises(ValueError, match="`normal_0` is required") as excinfo:
        cxfc.BishopBuilder(circle, "s")
    msg = str(excinfo.value)

    assert "3-vector" in msg
    assert "auto" in msg
    assert "#870" in msg  # where the caveats are written down


def test_auto_reproduces_the_old_default_exactly() -> None:
    """``"auto"`` seeds from `_auto_normal_0` at ``tau_0``, exactly.

    That wiring is what the old `None` default did, so every site that opted
    into ``"auto"`` kept its numbers. Asserted exactly, not within a tolerance:
    any drift is a silent change of frame.
    """
    b = cxfc.BishopBuilder(circle, "s", normal_0="auto")
    r = np.asarray(b.rotation_matrix(u.Q(0.37, "s")))

    # Gram--Schmidt of the least-aligned world axis against T(tau_0), which is
    # what `_auto_normal_0` computes and what the default used to be.
    from coordinaxs.curveframes._src.bishop import _auto_normal_0

    t0 = np.asarray(b.tangent(u.Q(0.0, "s")).ustrip(""))
    expected_seed = np.asarray(_auto_normal_0(jnp.asarray(t0)))
    explicit = cxfc.BishopBuilder(circle, "s", normal_0=jnp.asarray(expected_seed))

    assert np.array_equal(r, np.asarray(explicit.rotation_matrix(u.Q(0.37, "s"))))


def test_an_explicit_vector_is_honoured() -> None:
    """The other way out, and it must actually change the frame."""
    auto = cxfc.BishopBuilder(circle, "s", normal_0="auto")
    chosen = cxfc.BishopBuilder(circle, "s", normal_0=jnp.asarray([0.0, 0.0, 1.0]))

    assert not np.allclose(
        np.asarray(auto.rotation_matrix(u.Q(0.37, "s"))),
        np.asarray(chosen.rotation_matrix(u.Q(0.37, "s"))),
    )


def test_the_sentinel_is_not_a_pytree_leaf() -> None:
    """`'auto'` in a dynamic field would ride along in every trace."""
    b = cxfc.BishopBuilder(circle, "s", normal_0="auto")

    assert b.auto_seed is True
    assert not any(isinstance(leaf, str) for leaf in jax.tree.leaves(b))


@pytest.mark.parametrize("typo", ["atuo", "AUTO", "Auto", "auto "])
def test_a_near_miss_string_is_refused(typo: str) -> None:
    """A stray string would ride along as a pytree leaf, failing later in JAX."""
    with pytest.raises(ValueError, match="is not a seed"):
        cxfc.BishopBuilder(circle, "s", normal_0=typo)
