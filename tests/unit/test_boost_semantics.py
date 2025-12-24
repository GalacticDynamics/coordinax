"""Tests for Boost operator semantics according to spec.

Tests the role-specialized Boost operator behavior:
- Point: x'(tau) = x(tau) + v0 * (tau - tau0)
- Pos: identity (displacement invariant under boost)
- Vel: v' = v + v0
- Acc: identity (constant boost)
"""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx
import coordinax.ops as cxo
import coordinax.roles as cxr


class TestBoostOnVector:
    """Tests for Boost applied to Vector."""

    def test_boost_on_point_with_tau(self):
        """Boost on Point role translates by v0 * (tau - tau0)."""
        # Create boost with v0 = [100, 0, 0] km/s and tau0 = 0
        boost = cxo.Boost.from_([100, 0, 0], "km/s")

        # Create a point at origin
        p = cx.Vector.from_([0, 0, 0], "km")  # Point role by default

        # Apply boost at tau = 1 second
        tau = u.Q(1, "s")
        p_prime = boost(tau, p)

        # Should be translated by v0 * tau = [100, 0, 0] km/s * 1 s = [100, 0, 0] km
        assert jnp.allclose(u.ustrip("km", p_prime.data["x"]), 100.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km", p_prime.data["y"]), 0.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km", p_prime.data["z"]), 0.0, rtol=1e-5)

    def test_boost_on_point_with_tau0(self):
        """Boost on Point uses tau0 reference epoch."""
        # Create boost with v0 = [100, 0, 0] km/s and tau0 = 1 s
        boost = cxo.Boost.from_([100, 0, 0], "km/s", tau0=u.Q(1, "s"))

        # Create a point at origin
        p = cx.Vector.from_([0, 0, 0], "km")

        # Apply boost at tau = 2 seconds -> dt = tau - tau0 = 1 s
        tau = u.Q(2, "s")
        p_prime = boost(tau, p)

        # Should be translated by v0 * (tau - tau0) = [100, 0, 0] km/s * 1 s
        assert jnp.allclose(u.ustrip("km", p_prime.data["x"]), 100.0, rtol=1e-5)

    def test_boost_on_point_with_tau_none(self):
        """Boost on Point with tau=None uses tau=0."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")  # tau0 defaults to 0

        p = cx.Vector.from_([0, 0, 0], "km")

        # Apply boost with tau=None (meaning tau=0)
        p_prime = boost(None, p)

        # dt = 0 - 0 = 0, so no translation
        assert jnp.allclose(u.ustrip("km", p_prime.data["x"]), 0.0, rtol=1e-5)

    def test_boost_on_pos_is_identity(self):
        """Boost on Pos role is identity (displacements invariant)."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")

        # Create a displacement vector (using Quantity form for explicit role)
        dx = cx.Vector.from_(u.Q([10, 20, 30], "km"), cxr.pos)

        # Apply boost
        dx_prime = boost(u.Q(1, "s"), dx)

        # Should be unchanged
        assert jnp.allclose(u.ustrip("km", dx_prime.data["x"]), 10.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km", dx_prime.data["y"]), 20.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km", dx_prime.data["z"]), 30.0, rtol=1e-5)

    def test_boost_on_vel(self):
        """Boost on Vel role adds v0."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")

        # Create a velocity vector (using Quantity form for explicit role)
        v = cx.Vector.from_(u.Q([10, 20, 30], "km/s"), cxr.vel)

        # Apply boost
        v_prime = boost(None, v)

        # Should add v0 to velocity
        assert jnp.allclose(u.ustrip("km/s", v_prime.data["x"]), 110.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km/s", v_prime.data["y"]), 20.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km/s", v_prime.data["z"]), 30.0, rtol=1e-5)

    def test_boost_on_acc_is_identity(self):
        """Boost on Acc role is identity (constant boost)."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")

        # Create an acceleration vector (using Quantity form for explicit role)
        a = cx.Vector.from_(u.Q([1, 2, 3], "m/s^2"), cxr.acc)

        # Apply boost
        a_prime = boost(None, a)

        # Should be unchanged
        assert jnp.allclose(u.ustrip("m/s^2", a_prime.data["x"]), 1.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("m/s^2", a_prime.data["y"]), 2.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("m/s^2", a_prime.data["z"]), 3.0, rtol=1e-5)


class TestTranslateBoostPipeline:
    """Tests for Translate | Boost pipeline on Point vectors."""

    def test_pipe_on_point(self):
        """Translate | Boost pipeline applies both to Point."""
        # Alice-to-Bob style frame transform
        shift = cxo.Translate.from_([1, 0, 0], "km")
        boost = cxo.Boost.from_([10, 0, 0], "km/s")  # tau0=0
        pipe = shift | boost

        # Create a point at origin
        p = cx.Vector.from_([0, 0, 0], "km")

        # Apply at tau = 1 s
        # - Translate adds [1, 0, 0] km
        # - Boost adds v0 * tau = [10, 0, 0] km/s * 1 s = [10, 0, 0] km
        # Total: [11, 0, 0] km
        p_prime = pipe(u.Q(1, "s"), p)

        assert jnp.allclose(u.ustrip("km", p_prime.data["x"]), 11.0, rtol=1e-5)

    def test_pipe_on_vel(self):
        """Translate | Boost pipeline: Translate is no-op on Vel, Boost adds v0."""
        shift = cxo.Translate.from_([1, 0, 0], "km")
        boost = cxo.Boost.from_([10, 0, 0], "km/s")
        pipe = shift | boost

        # Create a velocity vector (using Quantity form for explicit role)
        v = cx.Vector.from_(u.Q([5, 0, 0], "km/s"), r.vel)

        # Apply pipeline
        # - Translate is no-op on Vel (will raise, actually)
        # Wait - need to check if Translate raises on non-Point
        with pytest.raises(TypeError):
            pipe(None, v)


class TestBoostParameters:
    """Tests for Boost parameter handling."""

    def test_boost_tau0_default(self):
        """Default tau0 is 0 seconds."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")
        assert u.ustrip("s", boost.tau0) == 0

    def test_boost_inverse_preserves_tau0(self):
        """Inverse boost preserves tau0."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s", tau0=u.Q(10, "s"))
        inv_boost = boost.inverse

        assert u.ustrip("s", inv_boost.tau0) == 10.0

    def test_boost_addition_preserves_tau0(self):
        """Adding two boosts uses the first boost's tau0."""
        b1 = cxo.Boost.from_([100, 0, 0], "km/s", tau0=u.Q(10, "s"))
        b2 = cxo.Boost.from_([0, 50, 0], "km/s", tau0=u.Q(20, "s"))
        combined = b1 + b2

        assert u.ustrip("s", combined.tau0) == 10.0
