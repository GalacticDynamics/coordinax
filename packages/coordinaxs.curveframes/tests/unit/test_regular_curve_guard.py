"""Every curve frame needs a *regular* curve, and says so when it does not.

A parameter where ``gamma' = 0`` has no tangent direction. That is a property
of the parametrisation, and no builder repairs it -- so the refusal must name
the parametrisation and must not send the caller to a different builder.

Before this guard, `tangent` returned an all-NaN vector on `FrenetSerretBuilder`
and `BishopBuilder` with no error at all -- the same silent-NaN class that #856
removed from `normal` and `rotation_matrix`, left behind on the tangent path.
The messages that did fire blamed the wrong cause: vanishing *curvature* on
Frenet, leaving the *plane* on signed planar, neither of which is what went
wrong. See GalacticDynamics/coordinax#889.
"""

from typing import Any

import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def stationary(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """``(t^2, 0, 0)`` km: ``gamma'(0) = 0``, so ``tau = 0`` is not regular.

    Regular everywhere else, so this isolates the degeneracy at one parameter
    rather than testing a curve that is bad throughout.
    """
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t**2, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")


def builders() -> list[Any]:
    """One of each, with `BishopBuilder` given the seed it now requires."""
    return [
        cxfc.FrenetSerretBuilder(stationary, "s"),
        cxfc.BishopBuilder(stationary, "s", normal_0="auto"),
        cxfc.SignedPlanarBuilder(stationary, "s"),
    ]


IDS = ["frenet-serret", "bishop", "signed-planar"]
AT_DEGENERACY = u.Q(0.0, "s")


@pytest.mark.parametrize("builder", builders(), ids=IDS)
class TestDegenerateParameterIsRefused:
    """All three builders refuse, and all three blame the right thing."""

    def test_tangent_raises_rather_than_returning_nan(self, builder: Any) -> None:
        """The silent-NaN half of #889, and the more serious one."""
        with pytest.raises(Exception, match="regular"):
            builder.tangent(AT_DEGENERACY)

    def test_rotation_matrix_raises(self, builder: Any) -> None:
        with pytest.raises(Exception, match="regular"):
            builder.rotation_matrix(AT_DEGENERACY)

    def test_the_message_names_the_parametrisation(self, builder: Any) -> None:
        """Not the curvature, and not the plane -- neither is what failed."""
        with pytest.raises(Exception, match="regular") as exc:
            builder.tangent(AT_DEGENERACY)
        msg = " ".join(str(exc.value).split())
        assert "gamma'" in msg or "derivative" in msg

    def test_the_message_does_not_recommend_another_builder(self, builder: Any) -> None:
        """No builder repairs a degenerate parametrisation, so none is offered."""
        with pytest.raises(Exception, match="regular") as exc:
            builder.tangent(AT_DEGENERACY)
        msg = " ".join(str(exc.value).split())
        assert "BishopBuilder" not in msg
        assert "SignedPlanarBuilder" not in msg


@pytest.mark.parametrize("builder", builders(), ids=IDS)
class TestRegularParametersAreUnaffected:
    """The guard is exact-zero: a merely *small* speed is still a direction."""

    def test_away_from_the_degeneracy(self, builder: Any) -> None:
        T = builder.tangent(u.Q(1.0, "s"))
        assert jnp.all(jnp.isfinite(T.value))
        assert jnp.allclose(jnp.linalg.norm(T.value), 1.0, atol=1e-10)

    def test_a_very_small_speed_is_accepted(self, builder: Any) -> None:
        """At tau=1e-8 the speed is 2e-8 km/s -- tiny, but a real direction."""
        T = builder.tangent(u.Q(1e-8, "s"))
        assert jnp.all(jnp.isfinite(T.value))
        assert jnp.allclose(jnp.linalg.norm(T.value), 1.0, atol=1e-10)
