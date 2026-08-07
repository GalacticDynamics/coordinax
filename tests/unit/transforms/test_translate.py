"""Tests for Translate operator with semantic_kind field."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm

# Sample CDicts (Cartesian, unitless) and the Translates built from them.
CART_DELTA = {"x": jnp.array(1), "y": jnp.array(2), "z": jnp.array(3)}
POINT = {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}
DISP = {"x": jnp.array(5), "y": jnp.array(6), "z": jnp.array(7)}
VEL = {"x": jnp.array(10), "y": jnp.array(20), "z": jnp.array(30)}
ACC = {"x": jnp.array(0.1), "y": jnp.array(0.2), "z": jnp.array(0.3)}

TR_DPL = cxfm.Translate(CART_DELTA, chart=cxc.cart3d)
TR_VEL = cxfm.Translate(CART_DELTA, chart=cxc.cart3d, semantic_kind=cxr.vel)
TR_ACC = cxfm.Translate(CART_DELTA, chart=cxc.cart3d, semantic_kind=cxr.acc)


# ============================================================================


class TestTranslateSemanticKindField:
    """Tests for the semantic_kind field of Translate."""

    def test_default_is_displacement(self):
        assert isinstance(TR_DPL.semantic_kind, cxr.Displacement)
        assert TR_DPL.semantic_kind == cxr.dpl

    def test_velocity_semantic_kind(self):
        assert isinstance(TR_VEL.semantic_kind, cxr.Velocity)
        assert TR_VEL.semantic_kind == cxr.vel

    def test_acceleration_semantic_kind(self):
        assert isinstance(TR_ACC.semantic_kind, cxr.Acceleration)
        assert TR_ACC.semantic_kind == cxr.acc

    def test_default_repr_hides_semantic_kind(self):
        # Default semantic_kind=dpl should NOT appear (it equals the default)
        assert "semantic_kind" not in repr(TR_DPL)

    def test_non_default_repr_shows_semantic_kind(self):
        assert "semantic_kind" in repr(TR_VEL)

    def test_inverse_preserves_semantic_kind(self):
        inv = TR_VEL.inverse
        assert isinstance(inv, cxfm.Translate)
        assert inv.semantic_kind == cxr.vel


# ============================================================================


class TestCallableDeltaRejected:
    """A callable ``delta`` (the deleted API) must fail LOUDLY, never silently.

    ``delta`` is a plain `dict` field with no coercion, so without an explicit
    check a function sails through ``__init__``. Two layers reject it: runtime
    type-checking (enabled only by this project's pytest config) and
    `AbstractAdd.__check_init__` (always on -- the only guard library users
    get). Under pytest the type-checker fires FIRST and raises
    `jaxtyping.TypeCheckError`, which is not a `TypeError`, so the
    ``__check_init__`` guard is unreachable via normal construction here and
    is exercised directly below. Deleting ``__check_init__`` must fail
    ``test_check_init_guard_rejects_callable_delta``.
    """

    @staticmethod
    def _delta_fn(t):
        return {"x": u.Q(3.0, "km/s") * t, "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}

    @pytest.mark.parametrize(
        "cls", [cxfm.Translate, cxfm.Boost], ids=lambda c: c.__name__
    )
    def test_check_init_guard_rejects_callable_delta(self, cls):
        """THE guard: `__check_init__` itself, called directly.

        ``__check_init__`` reads only ``self.delta``, so a stand-in carrying
        one is enough to reach it without the ``__init__`` typecheck. The
        message must both reject the callable and name the replacement.
        """
        with pytest.raises(
            TypeError, match=r"component dict, not a callable[\s\S]*TimeDep"
        ):
            cls.__check_init__(SimpleNamespace(delta=self._delta_fn))


# ============================================================================


@pytest.mark.parametrize(
    ("op", "rep", "data", "shifts"),
    [
        # Displacement (the default): shifts points only. Per spec, a spatial
        # Translate is identity for every tangent representation.
        (TR_DPL, cxr.point, POINT, True),
        (TR_DPL, cxr.coord_disp, DISP, False),
        (TR_DPL, cxr.phys_disp, DISP, False),
        (TR_DPL, cxr.coord_vel, VEL, False),
        (TR_DPL, cxr.coord_acc, ACC, False),
        # semantic_kind=vel: acts only on velocity vectors.
        (TR_VEL, cxr.point, POINT, False),
        (TR_VEL, cxr.coord_disp, DISP, False),
        (TR_VEL, cxr.coord_vel, VEL, True),
        (TR_VEL, cxr.phys_vel, VEL, True),
        (TR_VEL, cxr.coord_acc, ACC, False),
        # semantic_kind=acc: acts only on acceleration vectors.
        (TR_ACC, cxr.point, POINT, False),
        (TR_ACC, cxr.coord_disp, DISP, False),
        (TR_ACC, cxr.coord_vel, VEL, False),
        (TR_ACC, cxr.coord_acc, ACC, True),
    ],
)
def test_semantic_kind_picks_the_one_slot_that_moves(op, rep, data, shifts):
    """A Translate shifts the representation matching its ``semantic_kind``."""
    result = cxfm.act(op, None, data, cxc.cart3d, rep)
    for k, v in data.items():
        assert jnp.allclose(result[k], v + CART_DELTA[k] if shifts else v)


# ============================================================================


class TestTranslateVelJAXCompatibility:
    """Tests Translate(kind=vel) is compatible with JAX transformations."""

    def test_jit_velocity_semantic(self):
        result = jax.jit(
            lambda v: cxfm.act(TR_VEL, None, v, cxc.cart3d, cxr.coord_vel)
        )(VEL)
        for k, v in VEL.items():
            assert jnp.allclose(result[k], v + TR_VEL.delta[k])

    def test_vmap_velocity_semantic(self):
        batch = {"x": jnp.ones(4) * 10, "y": jnp.ones(4) * 20, "z": jnp.ones(4) * 30}
        result = jax.vmap(
            lambda v: cxfm.act(TR_VEL, None, v, cxc.cart3d, cxr.coord_vel)
        )(batch)
        assert result["x"].shape == (4,)
        assert jnp.allclose(result["x"], batch["x"] + TR_VEL.delta["x"])


# ============================================================================


class TestTranslateVelRoundtrip:
    """Tests roundtripping translate velocity vector."""

    def test_vel_roundtrip(self):
        shifted = cxfm.act(TR_VEL, None, VEL, cxc.cart3d, cxr.coord_vel)
        restored = cxfm.act(TR_VEL.inverse, None, shifted, cxc.cart3d, cxr.coord_vel)
        for k, v in VEL.items():
            assert jnp.allclose(restored[k], v, atol=1e-6)

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

    @pytest.mark.parametrize(
        ("op", "kind"), [(TR_DPL, cxr.dpl), (TR_VEL, cxr.vel), (TR_ACC, cxr.acc)]
    )
    def test_add_same_kind(self, op, kind):
        result = op + op
        assert isinstance(result, cxfm.Translate)
        assert result.semantic_kind == kind

    def test_add_different_types_gives_composed(self):
        assert isinstance(TR_DPL + TR_VEL, cxfm.Composed)

    def test_vel_add_combined_delta(self):
        combined = TR_VEL + TR_VEL
        result = cxfm.act(combined, None, VEL, cxc.cart3d, cxr.coord_vel)
        for k, v in VEL.items():
            assert jnp.allclose(result[k], v + 2 * CART_DELTA[k])


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


# ============================================================================


def test_static_fibre_offset_lands_only_on_the_matching_slot():
    """`act_jet` on a *static* fibre offset prolongs slot-wise, ladder-free.

    This is the `ladder=None` arm of ``_prolong_slotwise``: with no
    materialized ``(op0, k)`` to reuse, every slot goes through the ordinary
    `act`, which applies the static offset's own ladder rule. The behaviour
    that pins it is that the offset lands on exactly ONE slot -- the one whose
    order matches its ``semantic_kind`` -- because a static delta has zero
    tau-derivative, so every higher slot gains nothing.
    """
    jet = {0: POINT, 1: VEL, 2: ACC}
    out = cxfm.act_jet(TR_VEL, None, jet, cxc.cart3d)

    for k, p0 in POINT.items():  # identity point action
        assert jnp.allclose(out[0][k], p0)
        # k=1 offset hits the velocity slot ...
        assert jnp.allclose(out[1][k], VEL[k] + CART_DELTA[k])
        # ... and nothing else: d(delta)/dtau = 0 for a static offset.
        assert jnp.allclose(out[2][k], ACC[k])
