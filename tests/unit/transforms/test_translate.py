"""Tests for Translate operator with semantic_kind field."""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from coordinax.internal import CDict


@pytest.fixture
def cart_delta() -> CDict:
    """A sample delta vector in Cartesian coordinates."""
    return {"x": jnp.array(1), "y": jnp.array(2), "z": jnp.array(3)}


@pytest.fixture
def translate_dpl(cart_delta) -> cxfm.Translate:
    """A Translate with default semantic_kind (displacement)."""
    return cxfm.Translate(cart_delta, chart=cxc.cart3d)


@pytest.fixture
def translate_vel(cart_delta) -> cxfm.Translate:
    """A Translate with semantic_kind=velocity."""
    return cxfm.Translate(cart_delta, chart=cxc.cart3d, semantic_kind=cxr.vel)


@pytest.fixture
def translate_acc(cart_delta) -> cxfm.Translate:
    """A Translate with semantic_kind=acceleration."""
    return cxfm.Translate(cart_delta, chart=cxc.cart3d, semantic_kind=cxr.acc)


@pytest.fixture
def point_cdict() -> CDict:
    """A sample point in Cartesian coordinates."""
    return {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}


@pytest.fixture
def vel_cdict() -> CDict:
    """A sample velocity vector in Cartesian coordinates."""
    return {"x": jnp.array(10), "y": jnp.array(20), "z": jnp.array(30)}


@pytest.fixture
def disp_cdict() -> CDict:
    """A sample displacement vector in Cartesian coordinates."""
    return {"x": jnp.array(5), "y": jnp.array(6), "z": jnp.array(7)}


@pytest.fixture
def acc_cdict() -> CDict:
    """A sample acceleration vector in Cartesian coordinates."""
    return {"x": jnp.array(0.1), "y": jnp.array(0.2), "z": jnp.array(0.3)}


# ============================================================================


class TestTranslateSemanticKindField:
    """Tests for the semantic_kind field of Translate."""

    def test_default_is_displacement(self, translate_dpl):
        assert isinstance(translate_dpl.semantic_kind, cxr.Displacement)
        assert translate_dpl.semantic_kind == cxr.dpl

    def test_velocity_semantic_kind(self, translate_vel):
        assert isinstance(translate_vel.semantic_kind, cxr.Velocity)
        assert translate_vel.semantic_kind == cxr.vel

    def test_acceleration_semantic_kind(self, translate_acc):
        assert isinstance(translate_acc.semantic_kind, cxr.Acceleration)
        assert translate_acc.semantic_kind == cxr.acc

    def test_default_repr_hides_semantic_kind(self, translate_dpl):
        r = repr(translate_dpl)
        # Default semantic_kind=dpl should NOT appear (it equals the default)
        assert "semantic_kind" not in r

    def test_non_default_repr_shows_semantic_kind(self, translate_vel):
        r = repr(translate_vel)
        assert "semantic_kind" in r

    def test_inverse_preserves_semantic_kind(self, translate_vel):
        inv = translate_vel.inverse
        assert isinstance(inv, cxfm.Translate)
        assert inv.semantic_kind == cxr.vel


# ============================================================================


class TestTranslateDisplacementSemantic:
    """Translate with semantic_kind=dpl (default) shifts points only.

    Per spec: a spatial Translate is identity for all tangent representations
    (displacements, velocities, accelerations).
    """

    def test_shifts_point(self, translate_dpl, point_cdict, cart_delta):
        result = cxfm.act(translate_dpl, None, point_cdict, cxc.cart3d, cxr.point)
        for k in point_cdict:
            assert jnp.allclose(result[k], point_cdict[k] + cart_delta[k])

    def test_identity_for_coord_disp(self, translate_dpl, disp_cdict):
        result = cxfm.act(translate_dpl, None, disp_cdict, cxc.cart3d, cxr.coord_disp)
        for k in disp_cdict:
            assert jnp.allclose(result[k], disp_cdict[k])

    def test_identity_for_phys_disp(self, translate_dpl, disp_cdict):
        result = cxfm.act(translate_dpl, None, disp_cdict, cxc.cart3d, cxr.phys_disp)
        for k in disp_cdict:
            assert jnp.allclose(result[k], disp_cdict[k])

    def test_identity_for_velocity(self, translate_dpl, vel_cdict):
        result = cxfm.act(translate_dpl, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k])

    def test_identity_for_acceleration(self, translate_dpl, acc_cdict):
        result = cxfm.act(translate_dpl, None, acc_cdict, cxc.cart3d, cxr.coord_acc)
        for k in acc_cdict:
            assert jnp.allclose(result[k], acc_cdict[k])


# ============================================================================


class TestTranslateVelocitySemantic:
    """Translate with semantic_kind=vel acts only on velocity vectors."""

    def test_identity_for_point(self, translate_vel, point_cdict):
        result = cxfm.act(translate_vel, None, point_cdict, cxc.cart3d, cxr.point)
        for k in point_cdict:
            assert jnp.allclose(result[k], point_cdict[k])

    def test_identity_for_displacement(self, translate_vel, disp_cdict):
        result = cxfm.act(translate_vel, None, disp_cdict, cxc.cart3d, cxr.coord_disp)
        for k in disp_cdict:
            assert jnp.allclose(result[k], disp_cdict[k])

    def test_shifts_coord_vel(self, translate_vel, vel_cdict, cart_delta):
        result = cxfm.act(translate_vel, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k] + cart_delta[k])

    def test_shifts_phys_vel(self, translate_vel, vel_cdict, cart_delta):
        result = cxfm.act(translate_vel, None, vel_cdict, cxc.cart3d, cxr.phys_vel)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k] + cart_delta[k])

    def test_identity_for_acceleration(self, translate_vel, acc_cdict):
        result = cxfm.act(translate_vel, None, acc_cdict, cxc.cart3d, cxr.coord_acc)
        for k in acc_cdict:
            assert jnp.allclose(result[k], acc_cdict[k])


# ============================================================================


class TestTranslateAccelerationSemantic:
    """Translate with semantic_kind=acc acts only on acceleration vectors."""

    def test_identity_for_point(self, translate_acc, point_cdict):
        result = cxfm.act(translate_acc, None, point_cdict, cxc.cart3d, cxr.point)
        for k in point_cdict:
            assert jnp.allclose(result[k], point_cdict[k])

    def test_identity_for_displacement(self, translate_acc, disp_cdict):
        result = cxfm.act(translate_acc, None, disp_cdict, cxc.cart3d, cxr.coord_disp)
        for k in disp_cdict:
            assert jnp.allclose(result[k], disp_cdict[k])

    def test_identity_for_velocity(self, translate_acc, vel_cdict):
        result = cxfm.act(translate_acc, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k])

    def test_shifts_coord_acc(self, translate_acc, acc_cdict, cart_delta):
        result = cxfm.act(translate_acc, None, acc_cdict, cxc.cart3d, cxr.coord_acc)
        for k in acc_cdict:
            assert jnp.allclose(result[k], acc_cdict[k] + cart_delta[k])


# ============================================================================


class TestTranslateVelJAXCompatibility:
    """Tests Translate(kind=vel) is compatible with JAX transformations."""

    def test_jit_velocity_semantic(self, translate_vel, vel_cdict):
        result = jax.jit(
            lambda v: cxfm.act(translate_vel, None, v, cxc.cart3d, cxr.coord_vel)
        )(vel_cdict)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k] + translate_vel.delta[k])

    def test_vmap_velocity_semantic(self, translate_vel):
        batch = {"x": jnp.ones(4) * 10, "y": jnp.ones(4) * 20, "z": jnp.ones(4) * 30}
        result = jax.vmap(
            lambda v: cxfm.act(translate_vel, None, v, cxc.cart3d, cxr.coord_vel)
        )(batch)
        assert result["x"].shape == (4,)
        assert jnp.allclose(result["x"], batch["x"] + translate_vel.delta["x"])


# ============================================================================


class TestTranslateVelRoundtrip:
    """Tests roundtripping translate velocity vector."""

    def test_vel_roundtrip(self, translate_vel, vel_cdict):
        shifted = cxfm.act(translate_vel, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        restored = cxfm.act(
            translate_vel.inverse, None, shifted, cxc.cart3d, cxr.coord_vel
        )
        for k in vel_cdict:
            assert jnp.allclose(restored[k], vel_cdict[k], atol=1e-6)

    def test_point_roundtrip_with_quantity(self):
        shift = cxfm.Translate.from_([1, 2, 3], "km")
        x = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
        shifted = cxfm.act(shift, None, x, cxc.cart3d, cxr.point)
        restored = cxfm.act(shift.inverse, None, shifted, cxc.cart3d, cxr.point)
        for k, v in x.items():
            assert jnp.allclose(
                u.ustrip(u.unit("km"), restored[k]),
                u.ustrip(u.unit("km"), v),
                atol=1e-6,
            )


# ============================================================================


class TestTranslateAddPreservesSemanticKind:
    """Tests adding Translates with the same semantic_kind gives a Translate."""

    def test_add_dpl_dpl(self, translate_dpl):
        result = translate_dpl + translate_dpl
        assert isinstance(result, cxfm.Translate)
        assert result.semantic_kind == cxr.dpl

    def test_add_vel_vel(self, translate_vel):
        result = translate_vel + translate_vel
        assert isinstance(result, cxfm.Translate)
        assert result.semantic_kind == cxr.vel

    def test_add_acc_acc(self, translate_acc):
        result = translate_acc + translate_acc
        assert isinstance(result, cxfm.Translate)
        assert result.semantic_kind == cxr.acc

    def test_add_different_types_gives_composed(self, translate_dpl, translate_vel):
        result = translate_dpl + translate_vel
        assert isinstance(result, cxfm.Composed)

    def test_vel_add_combined_delta(self, translate_vel, cart_delta, vel_cdict):
        combined = translate_vel + translate_vel
        result = cxfm.act(combined, None, vel_cdict, cxc.cart3d, cxr.coord_vel)
        for k in vel_cdict:
            assert jnp.allclose(result[k], vel_cdict[k] + 2 * cart_delta[k])


# ============================================================================


class TestTranslateDisplacementNonCartesianDelta:
    """Translate with delta in non-Cartesian chart uses tangent_map Jacobian."""

    def test_non_cartesian_delta_with_usys(self):
        """Translate with delta in spherical chart works when usys is provided."""
        usys = u.unitsystems.si

        # delta expressed in spherical 3d chart (with units)
        sph_delta = {"r": u.Q(1, "km"), "theta": u.Q(0, "rad"), "phi": u.Q(0, "rad")}
        t = cxfm.Translate(sph_delta, chart=cxc.sph3d)

        # Apply to a Cartesian point (with units)
        x = {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
        # Previously raised NotImplementedError; now uses tangent_map Jacobian
        result = cxfm.act(t, None, x, cxc.cart3d, cxr.point, usys=usys)
        assert "x" in result
        assert "y" in result
        assert "z" in result
