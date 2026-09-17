"""`BishopBuilder` must not choose the n-plane gauge for you.

The seed is arbitrary *and* load-bearing: a chart's `(n1, n2)` depend on it
wherever the curvature and the offset are both non-zero, so a caller who never
chose one still depended on it. That silence is the defect behind #870, where
a rigid rotation -- an isometry -- reported `|K| = 0.1178` instead of zero.

These pin the breaking half directly, so it cannot drift back to a default.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def test_omitting_the_seed_raises() -> None:
    """The whole point of the change: silence is no longer a choice."""
    with pytest.raises(ValueError, match="`initial_normal` is required"):
        cxfc.BishopBuilder(circle, "s")


def test_the_message_names_both_ways_out() -> None:
    """A vector, or the world-axis rule by name -- and the caveats of the latter."""
    with pytest.raises(ValueError, match="`initial_normal` is required") as excinfo:
        cxfc.BishopBuilder(circle, "s")
    msg = str(excinfo.value)

    assert "auto" in msg
    assert "equivariant" in msg  # why the auto rule is not free
    assert "#870" in msg


def test_auto_reproduces_the_old_default_exactly() -> None:
    """Bit-identical, which is why ~170 existing sites kept their numbers.

    If this ever drifts, every call site that opted into ``"auto"`` silently
    changed frame -- so it is asserted exactly rather than within a tolerance.
    """
    b = cxfc.BishopBuilder(circle, "s", initial_normal="auto")
    r = np.asarray(b.rotation_matrix(u.Q(0.37, "s")))

    # Gram--Schmidt of the least-aligned world axis against T(tau_0), which is
    # what `_auto_initial_normal` computes and what the default used to be.
    from coordinaxs.curveframes._src.bishop import _auto_initial_normal

    t0 = np.asarray(b.tangent(u.Q(0.0, "s")).ustrip(""))
    expected_seed = np.asarray(_auto_initial_normal(jnp.asarray(t0)))
    explicit = cxfc.BishopBuilder(
        circle, "s", initial_normal=jnp.asarray(expected_seed)
    )

    assert np.array_equal(r, np.asarray(explicit.rotation_matrix(u.Q(0.37, "s"))))


def test_an_explicit_vector_is_honoured() -> None:
    """The other way out, and it must actually change the frame."""
    auto = cxfc.BishopBuilder(circle, "s", initial_normal="auto")
    chosen = cxfc.BishopBuilder(
        circle, "s", initial_normal=jnp.asarray([0.0, 0.0, 1.0])
    )

    assert not np.allclose(
        np.asarray(auto.rotation_matrix(u.Q(0.37, "s"))),
        np.asarray(chosen.rotation_matrix(u.Q(0.37, "s"))),
    )


def test_the_sentinel_is_not_a_pytree_leaf() -> None:
    """`'auto'` in a dynamic field would ride along in every trace.

    Measured when it did: 103 failures across the package, because
    `jax.tree.leaves` returned a `str` beside the arrays.
    """
    import jax

    leaves = jax.tree.leaves(cxfc.BishopBuilder(circle, "s", initial_normal="auto"))

    assert not any(isinstance(leaf, str) for leaf in leaves)


@pytest.mark.parametrize("typo", ["atuo", "AUTO", "Auto", "auto "])
def test_a_near_miss_string_is_refused(typo: str) -> None:
    """A stray string would ride along as a pytree leaf, failing later in JAX."""
    with pytest.raises(ValueError, match="is not a seed"):
        cxfc.BishopBuilder(circle, "s", initial_normal=typo)


def test_only_the_exact_sentinel_is_accepted() -> None:
    """And it leaves nothing behind in the tree."""
    import jax

    b = cxfc.BishopBuilder(circle, "s", initial_normal="auto")

    assert b.auto_seed is True
    assert not any(isinstance(leaf, str) for leaf in jax.tree.leaves(b))
