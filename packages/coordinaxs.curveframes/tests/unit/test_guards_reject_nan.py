"""A NaN must not walk through a guard that claims to reject bad input.

`x <= tol` and `x > hi` are both False for a NaN, so a guard written as a
direct comparison admits it and returns a NaN result with nothing raised --
worse than the case the guard was written for, which at least errors.

`TubularChart`'s reach guard already avoids this by testing `~(f > 0)`; these
pin the same property for the two guards that were still written the direct
way.
"""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc
from coordinaxs.curveframes._src.bishop import _orthonormalize


def helix(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf], ids=["nan", "inf"])
def test_a_non_finite_initial_normal_is_rejected(bad: float) -> None:
    """`_orthonormalize` returned a NaN triad instead of raising.

    The rejection was `norm <= 1e-12 * |v|`; with a NaN `v` both sides are NaN
    and the comparison is False. Fails if it goes back to the direct form.
    """
    T0 = jnp.array([1.0, 0.0, 0.0])
    with pytest.raises(Exception, match="parallel"):
        _orthonormalize(jnp.array([1.0, bad, 0.0]), T0)


def test_the_legitimate_degenerate_cases_still_raise() -> None:
    """The case the guard was written for keeps working."""
    T0 = jnp.array([1.0, 0.0, 0.0])
    for v in (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0])):
        with pytest.raises(Exception, match="parallel"):
            _orthonormalize(v, T0)

    # and a well-conditioned normal is untouched
    out = _orthonormalize(jnp.array([0.0, 2.0, 0.0]), T0)
    assert jnp.allclose(jnp.asarray(out), jnp.array([0.0, 1.0, 0.0]))


def test_a_nan_arc_length_is_rejected() -> None:
    """The domain guard returned a NaN position instead of raising.

    The test was `(s < -margin) | (s > s_max + margin)`; a NaN is False for
    both, so it fell through to `diffrax`, which happily interpolated a NaN.
    """
    fast = cxfc.ArcLength(helix, "s", s_max=u.Q(5.0, "km"))
    with pytest.raises(Exception, match="solved domain"):
        fast(u.Q(jnp.nan, "km"))

    # in-domain and genuinely-outside both behave as before
    assert jnp.isfinite(fast(u.Q(2.0, "km")).ustrip("km")).all()
    with pytest.raises(Exception, match="solved domain"):
        fast(u.Q(99.0, "km"))


# --------------------------------------------------------------------------
# The same class in core: range guards written as a direct comparison.


def test_a_nan_boost_velocity_is_rejected() -> None:
    """`LorentzBoost` returned a non-finite gamma instead of raising.

    The guard was `beta_sq >= 1.0`, and its own comment says it exists so that
    no derived quantity "leaks a non-finite value" -- which a NaN did, being
    False for that comparison.
    """
    import coordinax.transforms as cxfm

    bad = cxfm.LorentzBoost(u.Q(jnp.array([jnp.nan, 0.0, 0.0]), ""))
    with pytest.raises(Exception, match="subluminal"):
        _ = bad.gamma

    # subluminal is unaffected
    ok = cxfm.LorentzBoost(u.Q(jnp.array([0.5, 0.0, 0.0]), ""))
    assert jnp.isfinite(ok.gamma)


def test_nan_fails_the_coordinate_bounds_checks() -> None:
    """`leq`/`geq` admitted a NaN, so an out-of-range coordinate slipped by."""
    from coordinax._src.charts.checks import geq, leq

    with pytest.raises(Exception, match="less than or equal"):
        leq(u.Q(jnp.nan, "m"), u.Q(2, "m"))
    with pytest.raises(Exception, match="greater than or equal"):
        geq(u.Q(jnp.nan, "m"), u.Q(2, "m"))

    # in-range values still pass
    leq(u.Q(1.0, "m"), u.Q(2, "m"))
    geq(u.Q(3.0, "m"), u.Q(2, "m"))
