"""The two `curveframes` guards reject what they cannot handle.

Each covers the degenerate case it was written for and the non-finite ones it
used to admit before being rewritten as a negated positive test.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc
from coordinaxs.curveframes._src.bishop import _orthonormalize


def helix(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


T0 = jnp.array([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "v",
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, jnp.nan, 0.0], [1.0, jnp.inf, 0.0]],
    ids=["parallel", "zero", "nan", "inf"],
)
def test_an_unusable_initial_normal_is_rejected(v: list[float]) -> None:
    """`parallel` and `zero` are the original cases; the rest returned a NaN triad."""
    with pytest.raises(eqx.EquinoxRuntimeError, match="parallel"):
        _orthonormalize(jnp.array(v), T0)


def test_a_well_conditioned_normal_is_untouched() -> None:
    """The guard costs the valid case nothing."""
    out = _orthonormalize(jnp.array([0.0, 2.0, 0.0]), T0)
    assert jnp.allclose(jnp.asarray(out), jnp.array([0.0, 1.0, 0.0]))


@pytest.mark.parametrize("s", [jnp.nan, 99.0], ids=["nan", "outside"])
def test_an_out_of_domain_arc_length_is_rejected(s: float) -> None:
    """A NaN fell through to `diffrax`, which happily interpolated it."""
    fast = cxfc.ArcLength(helix, "s", s_max=u.Q(5.0, "km"))
    with pytest.raises(eqx.EquinoxRuntimeError, match="solved domain"):
        fast(u.Q(s, "km"))


def test_an_in_domain_arc_length_is_untouched() -> None:
    """The guard costs the valid case nothing."""
    fast = cxfc.ArcLength(helix, "s", s_max=u.Q(5.0, "km"))
    assert jnp.isfinite(fast(u.Q(2.0, "km")).ustrip("km")).all()
