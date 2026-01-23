"""Tests for Boost operator semantics according to spec.

Tests the role-specialized Boost operator behavior:
- Point: TypeError (boost does not act on Point role)
- Pos: identity (displacement invariant under boost)
- Vel: v' = v + dv
- PhysAcc: identity for constant boost, a' = a + d(dv)/dt for time-dependent boost
"""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx
import coordinax.ops as cxo
import coordinax.roles as cxr


class TestBoostOnVector:
    """Tests for Boost applied to Vector."""

    def test_boost_on_point_raises(self):
        """Boost on Point role raises TypeError."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")

        # Create a point at origin
        p = cx.Vector.from_([0, 0, 0], "km")  # Point role by default

        # Boost should raise TypeError on Point role
        with pytest.raises(TypeError, match="Boost does not act on Point role"):
            boost(None, p)

    def test_boost_on_pos_is_identity(self):
        """Boost on Pos role is identity (displacements invariant)."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")

        # Create a displacement vector (using Quantity form for explicit role)
        dx = cx.Vector.from_(u.Q([10, 20, 30], "km"), cxr.phys_disp)

        # Apply boost
        dx_prime = boost(u.Q(1, "s"), dx)

        # Should be unchanged
        assert jnp.allclose(u.ustrip("km", dx_prime.data["x"]), 10.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km", dx_prime.data["y"]), 20.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km", dx_prime.data["z"]), 30.0, rtol=1e-5)

    def test_boost_on_vel(self):
        """Boost on PhysVel role adds dv."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")

        # Create a velocity vector (using Quantity form for explicit role)
        v = cx.Vector.from_(u.Q([10, 20, 30], "km/s"), cxr.phys_vel)

        # Apply boost
        v_prime = boost(None, v)

        # Should add dv to velocity
        assert jnp.allclose(u.ustrip("km/s", v_prime.data["x"]), 110.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km/s", v_prime.data["y"]), 20.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km/s", v_prime.data["z"]), 30.0, rtol=1e-5)

    def test_boost_on_acc_is_identity(self):
        """Boost on PhysAcc role is identity (constant boost)."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")

        # Create an acceleration vector (using Quantity form for explicit role)
        a = cx.Vector.from_(u.Q([1, 2, 3], "m/s^2"), cxr.phys_acc)

        # Apply boost
        a_prime = boost(None, a)

        # Should be unchanged
        assert jnp.allclose(u.ustrip("m/s^2", a_prime.data["x"]), 1.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("m/s^2", a_prime.data["y"]), 2.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("m/s^2", a_prime.data["z"]), 3.0, rtol=1e-5)

    def test_boost_time_dependent_on_acc(self):
        """Time-dependent boost on PhysAcc adds time derivative."""

        # Create time-dependent boost: dv(t) = [t * 10, 0, 0] m/s
        def dv_fn(t):
            return {
                "x": u.Q(t.ustrip("s") * 10, "m/s"),
                "y": u.Q(0, "m/s"),
                "z": u.Q(0, "m/s"),
            }

        boost = cxo.Boost(dv_fn, chart=cx.cart3d)

        # Create an acceleration vector
        a = cx.Vector.from_(u.Q([1, 2, 3], "m/s^2"), cxr.phys_acc)

        # Apply boost at t = 5s
        # d/dt dv = 10 m/s^2 in x direction
        a_prime = boost(u.Q(5, "s"), a)

        # Should add derivative: [1 + 10, 2, 3] = [11, 2, 3] m/s^2
        assert jnp.allclose(u.ustrip("m/s^2", a_prime.data["x"]), 11.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("m/s^2", a_prime.data["y"]), 2.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("m/s^2", a_prime.data["z"]), 3.0, rtol=1e-5)


class TestBoostPipeline:
    """Tests for Boost in pipelines."""

    def test_boost_addition(self):
        """Adding two boosts combines the velocity offsets."""
        b1 = cxo.Boost.from_([100, 0, 0], "km/s")
        b2 = cxo.Boost.from_([0, 50, 0], "km/s")
        combined = b1 + b2

        # Apply to velocity
        v = cx.Vector.from_(u.Q([10, 20, 30], "km/s"), cxr.phys_vel)
        v_prime = combined(None, v)

        # Should add both boosts
        assert jnp.allclose(u.ustrip("km/s", v_prime.data["x"]), 110.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km/s", v_prime.data["y"]), 70.0, rtol=1e-5)

    def test_boost_inverse(self):
        """Inverse boost negates the velocity offset."""
        boost = cxo.Boost.from_([100, 0, 0], "km/s")
        inv_boost = boost.inverse

        # Apply original boost
        v = cx.Vector.from_(u.Q([10, 20, 30], "km/s"), cxr.phys_vel)
        v_boosted = boost(None, v)

        # Apply inverse boost
        v_back = inv_boost(None, v_boosted)

        # Should return to original
        assert jnp.allclose(u.ustrip("km/s", v_back.data["x"]), 10.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km/s", v_back.data["y"]), 20.0, rtol=1e-5)
        assert jnp.allclose(u.ustrip("km/s", v_back.data["z"]), 30.0, rtol=1e-5)
