"""Tests for the Boost operator."""

import jax
import jax.numpy as jnp
import pytest

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm


@pytest.fixture
def delta_v() -> dict:
    return {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}


@pytest.fixture
def boost(delta_v) -> cxfm.Boost:
    return cxfm.Boost(delta_v, chart=cxc.cart3d)


@pytest.fixture
def vel_cdict() -> dict:
    return {"x": jnp.array(2.0), "y": jnp.array(3.0), "z": jnp.array(4.0)}


# ============================================================================


class TestBoostConstruction:
    """Tests for the construction and basic properties of the Boost operator."""

    def test_delta_field(self, boost, delta_v):
        for k in delta_v:
            assert jnp.allclose(boost.delta[k], delta_v[k])

    def test_chart_field(self, boost):
        assert boost.chart is cxc.cart3d

    def test_repr(self, boost):
        r = repr(boost)
        assert "Boost(" in r
        assert "chart=Cart3D(M=Rn(3))" in r

    def test_inverse_is_boost(self, boost):
        inv = boost.inverse
        assert isinstance(inv, cxfm.Boost)

    def test_inverse_negates_delta(self, boost):
        inv = boost.inverse
        for k in boost.delta:
            assert jnp.allclose(inv.delta[k], -boost.delta[k])

    def test_add_same_type(self, boost):
        combined = boost + boost
        assert isinstance(combined, cxfm.Boost)
        for k in boost.delta:
            assert jnp.allclose(combined.delta[k], 2 * boost.delta[k])

    def test_groups_is_diffeomorphism_only(self, boost):
        assert boost.groups() == frozenset((cxfm.DiffeomorphismGroup,))

    def test_composed_groups_with_translate(self, boost):
        shift = cxfm.Translate.from_([1, 2, 3], "m")
        op = cxfm.Composed((shift, boost))
        assert op.groups() == frozenset((cxfm.DiffeomorphismGroup,))


# ============================================================================


class TestBoostOnPoint:
    """Boost should not affect points."""

    def test_identity(self, boost):
        p = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(3.0)}
        result = cxfm.act(boost, None, p, cxc.cart3d, cxr.point)
        for k, v in p.items():
            assert jnp.allclose(result[k], v)


class TestBoostOnDisplacement:
    """Boost should not affect displacements."""

    def test_identity_coord_disp(self, boost):
        d = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(3.0)}
        result = cxfm.act(boost, None, d, cxc.cart3d, cxr.coord_disp)
        for k, v in d.items():
            assert jnp.allclose(result[k], v)

    def test_identity_phys_disp(self, boost):
        d = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(3.0)}
        result = cxfm.act(boost, None, d, cxc.cart3d, cxr.phys_disp)
        for k, v in d.items():
            assert jnp.allclose(result[k], v)


class TestBoostOnVelocity:
    """Boost should shift velocities by the boost delta."""

    def test_shifts_coord_vel(self, boost, vel_cdict, delta_v):
        result = cxfm.act(boost, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k] + delta_v[k])

    def test_shifts_phys_vel(self, boost, vel_cdict, delta_v):
        result = cxfm.act(boost, None, vel_cdict, cxc.cart3d, cxr.phys_vel)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k] + delta_v[k])

    def test_right_add_true(self, delta_v, vel_cdict):
        boost_r = cxfm.Boost(delta_v, chart=cxc.cart3d, right_add=True)
        result = cxfm.act(boost_r, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k] + delta_v[k])

    def test_right_add_false(self, delta_v, vel_cdict):
        boost_l = cxfm.Boost(delta_v, chart=cxc.cart3d, right_add=False)
        result = cxfm.act(boost_l, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        for k in vel_cdict:
            # right_add=False: delta + x
            assert jnp.allclose(result[k], delta_v[k] + vel_cdict[k])

    def test_raises_on_chart_mismatch(self, boost):
        v = {"rho": jnp.array(2.0), "phi": jnp.array(3.0), "z": jnp.array(4.0)}
        with pytest.raises(ValueError, match="input chart to match the boost chart"):
            cxfm.act(boost, None, v, cxc.cyl3d, cxr.coord_vel)


class TestBoostOnAcceleration:
    """Boost should not affect accelerations."""

    def test_identity_coord_acc(self, boost):
        a = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(3.0)}
        result = cxfm.act(boost, None, a, cxc.cart3d, cxr.coord_acc)
        for k, v in a.items():
            assert jnp.allclose(result[k], v)


# ============================================================================


class TestBoostJAXCompatibility:
    """Tests Boost can be used with JAX transformations like jit and vmap."""

    def test_jit(self, boost, vel_cdict):
        result = jax.jit(lambda v: cxfm.act(boost, None, v, cxc.cart3d, cxr.coord_vel))(
            vel_cdict
        )
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k] + boost.delta[k])

    def test_vmap(self, boost):
        batch_vel = {
            "x": jnp.ones(3) * 2.0,
            "y": jnp.ones(3) * 3.0,
            "z": jnp.ones(3) * 4.0,
        }
        result = jax.vmap(
            lambda v: cxfm.act(boost, None, v, cxc.cart3d, cxr.coord_vel)
        )(batch_vel)
        assert result["x"].shape == (3,)
        assert jnp.allclose(result["x"], batch_vel["x"] + boost.delta["x"])


# ============================================================================


class TestBoostRoundTrip:
    """Tests that applying a boost and then its inverse restores the original vector."""

    def test_round_trip_velocity(self, boost, vel_cdict):
        boosted = cxfm.act(boost, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        restored = cxfm.act(boost.inverse, None, boosted, cxc.cart3d, cxr.coord_vel)
        for k in vel_cdict:
            assert jnp.allclose(restored[k], vel_cdict[k], atol=1e-6)

    def test_add_inverse_is_identity(self, boost, vel_cdict):
        identity_op = boost + boost.inverse
        result = cxfm.act(identity_op, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k], atol=1e-6)
