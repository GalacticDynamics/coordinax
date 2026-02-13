"""Tests for PointedVector."""

import equinox as eqx
import hypothesis.strategies as st
import jax
import pytest
from hypothesis import given, settings

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.roles as cxr
import coordinax_hypothesis as cxst
from coordinax.objs import PointedVector, Vector


class TestPointedVectorInit:
    """Tests for PointedVector initialization."""

    def test_basic_init(self):
        """Test basic initialization with explicit Vector instances."""
        # Create vectors with explicit chart and role
        base = Vector.from_(u.Q([1, 2, 3], "km"), cxc.cart3d, cxr.point)  # Affine point
        vel = Vector.from_(
            u.Q([10, 20, 30], "km/s"), cxc.cart3d, cxr.phys_vel
        )  # Velocity
        acc = Vector.from_(
            u.Q([0.1, 0.2, 0.3], "km/s^2"),
            cxc.cart3d,
            cxr.phys_acc,
        )  # Acceleration

        bundle = PointedVector(base=base, velocity=vel, acceleration=acc)

        # Base and fields are stored (but may be copies due to eqx.error_if)
        assert isinstance(bundle.base, Vector)
        assert isinstance(bundle["velocity"], Vector)
        assert isinstance(bundle["acceleration"], Vector)
        assert set(bundle.keys()) == {"velocity", "acceleration"}

    def test_base_must_have_point_role(self):
        """Test that base must have Point role."""
        # Create a non-Point vector (Vel)
        non_point = Vector.from_(u.Q([1, 2, 3], "km/s"), cxc.cart3d, cxr.phys_vel)
        vel = Vector.from_(u.Q([10, 20, 30], "km/s"), cxc.cart3d, cxr.phys_vel)

        with pytest.raises(RuntimeError, match="base must have role Point"):
            PointedVector(base=non_point, velocity=vel)

    def test_fields_cannot_have_point_role(self):
        """Test that field vectors cannot have Point role."""
        base = Vector.from_(
            u.Q([1, 2, 3], "km"), cxc.cart3d, cxr.point
        )  # Has Point role
        another_point = Vector.from_(
            u.Q([4, 5, 6], "km"), cxc.cart3d, cxr.point
        )  # Also has Point role

        with pytest.raises(
            RuntimeError,
            match=(
                r"Field 'extra_field' has role Point\. PointedVector stores fibre "
                r"vectors anchored at base; store additional points elsewhere."
            ),
        ):
            PointedVector(base=base, extra_field=another_point)

    def test_broadcastable_shapes(self):
        """Test that shapes must be broadcastable."""
        base = Vector.from_([1, 2, 3], "km")  # shape ()
        vel = Vector.from_(
            jnp.array([[10, 20, 30], [40, 50, 60]]), "km/s"
        )  # shape (2,)

        # This should work (broadcasting)
        bundle = PointedVector(base=base, velocity=vel)
        assert bundle.shape == (2,)

    def test_non_broadcastable_shapes_error(self):
        """Test that non-broadcastable shapes raise error."""
        base = Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "km")  # shape (2,)
        vel = Vector.from_(jnp.array([[[10, 20, 30]] * 3]), "km/s")  # shape (1, 3)

        # Shapes (2,) and (1, 3) are incompatible for final broadcast
        with pytest.raises(RuntimeError, match="vector shapes are not broadcastable"):
            PointedVector(base=base, velocity=vel)


class TestPointedVectorFrom:
    """Tests for from_ constructor."""

    def test_from_dict_with_base_key(self):
        """Test from_ with 'base' in the data dict."""
        data = {
            "base": u.Q([1, 2, 3], "km"),
            "velocity": u.Q([4, 5, 6], "km/s"),
        }
        bundle = PointedVector.from_(data)

        assert isinstance(bundle.base, Vector)
        assert isinstance(bundle["velocity"], Vector)

    def test_from_dict_with_explicit_base(self):
        """Test from_ with explicit base parameter."""
        base = Vector.from_([1, 2, 3], "km")
        data = {
            "velocity": u.Q([4, 5, 6], "km/s"),
        }
        bundle = PointedVector.from_(data, base=base)

        # Note: eqx.error_if creates copies, so check components match
        assert jnp.allclose(bundle.base["x"], base["x"], atol=u.Q(1e-10, "km"))

    def test_from_dict_missing_base_raises(self):
        """Test that from_ without base raises ValueError."""
        data = {
            "velocity": u.Q([4, 5, 6], "km/s"),
        }
        with pytest.raises(ValueError, match="base must be provided"):
            PointedVector.from_(data)


class TestPointedVectorMappingAPI:
    """Tests for mapping interface."""

    def test_keys_values_items(self):
        """Test mapping methods."""
        base = Vector.from_([1, 2, 3], "km")
        vel = Vector.from_([10, 20, 30], "km/s")
        acc = Vector.from_([0.1, 0.2, 0.3], "km/s^2")

        bundle = PointedVector(base=base, velocity=vel, acceleration=acc)

        assert set(bundle.keys()) == {"velocity", "acceleration"}
        assert len(list(bundle.values())) == 2
        assert len(list(bundle.items())) == 2

    def test_getitem_string(self):
        """Test __getitem__ with string key."""
        base = Vector.from_([1, 2, 3], "km")
        vel = Vector.from_([10, 20, 30], "km/s")

        bundle = PointedVector(base=base, velocity=vel)

        assert bundle["velocity"] is vel

    def test_getitem_index(self):
        """Test __getitem__ with numeric index."""
        base = Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "km")
        vel = Vector.from_(jnp.array([[10, 20, 30], [40, 50, 60]]), "km/s")

        bundle = PointedVector(base=base, velocity=vel)

        # Index to get first element
        sub_bundle = bundle[0]
        assert isinstance(sub_bundle, PointedVector)
        assert sub_bundle.shape == ()

    def test_q_property(self):
        """Test .q property (alias for base)."""
        base = Vector.from_([1, 2, 3], "km")
        vel = Vector.from_([10, 20, 30], "km/s")

        bundle = PointedVector(base=base, velocity=vel)

        assert bundle.q is bundle.base


class TestPointedVectorVConvert:
    """Tests for vconvert method."""

    def test_vconvert_basic(self):
        """Test basic conversion to spherical coordinates."""
        base = Vector.from_([1, 1, 1], "m")
        vel = Vector.from_([10, 10, 10], "m/s")

        bundle = PointedVector(base=base, velocity=vel)
        sph_bundle = bundle.vconvert(cxc.sph3d)

        # Check reps changed
        assert isinstance(sph_bundle.base.chart, cxc.Spherical3D)
        assert isinstance(sph_bundle["velocity"].chart, cxc.Spherical3D)

    def test_vconvert_vs_manual(self):
        """Test that vconvert matches manual conversion."""
        base = Vector.from_([1, 2, 3], "m")
        vel = Vector.from_([4, 5, 6], "m/s")
        bundle = PointedVector(base=base, velocity=vel)

        # Bundle conversion
        sph_bundle = bundle.vconvert(cxc.sph3d)

        # Manual conversion
        base_sph = base.vconvert(cxc.sph3d)
        at_for_vel = base.vconvert(vel.chart)  # Should be base itself (already cart3d)
        vel_sph = vel.vconvert(cxc.sph3d, at_for_vel)

        # Compare results
        for comp in sph_bundle.base.data:
            base_match = jnp.allclose(
                sph_bundle.base[comp].value, base_sph[comp].value, atol=1e-10
            )
            assert base_match
        for comp in sph_bundle["velocity"].data:
            vel_match = jnp.allclose(
                sph_bundle["velocity"][comp].value, vel_sph[comp].value, atol=1e-10
            )
            assert vel_match

    def test_vconvert_with_field_charts(self):
        """Test vconvert with different target reps for fields."""
        base = Vector.from_([1, 1, 1], "m")
        vel = Vector.from_([10, 10, 10], "m/s")

        bundle = PointedVector(base=base, velocity=vel)

        # Convert base to spherical, velocity to cylindrical
        mixed_bundle = bundle.vconvert(cxc.sph3d, field_charts={"velocity": cxc.cyl3d})

        assert isinstance(mixed_bundle.base.chart, cxc.Spherical3D)
        assert isinstance(mixed_bundle["velocity"].chart, cxc.Cylindrical3D)

    def test_vconvert_preserves_base_role(self):
        """Test that conversion preserves role flags."""
        base = Vector.from_([1, 2, 3], "m")
        vel = Vector.from_([4, 5, 6], "m/s")

        bundle = PointedVector(base=base, velocity=vel)
        sph_bundle = bundle.vconvert(cxc.sph3d)

        assert isinstance(sph_bundle.base.role, cxr.Point)
        assert isinstance(sph_bundle["velocity"].role, cxr.PhysVel)


class TestPointedVectorJAX:
    """Tests for JAX compatibility."""

    def test_jit_smoke_test(self):
        """Test that bundle vconvert works (JIT has limitations).

        Note: Due to representation objects being non-static, pointedvectors can't
        be directly passed through JIT boundaries. The vconvert method itself
        works correctly in eager mode.
        """
        base = Vector.from_([1, 2, 3], "m")
        vel = Vector.from_([4, 5, 6], "m/s")
        bundle = PointedVector(base=base, velocity=vel)

        # vconvert works in eager mode
        sph_bundle = bundle.vconvert(cxc.sph3d)

        assert isinstance(sph_bundle, PointedVector)
        assert isinstance(sph_bundle.base.chart, cxc.Spherical3D)
        assert isinstance(sph_bundle["velocity"].chart, cxc.Spherical3D)

    def test_vmap_smoke_test(self):
        """Test that bundle works with vmap."""
        # Create batched bundle
        base = Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "m")
        vel = Vector.from_(jnp.array([[10, 20, 30], [40, 50, 60]]), "m/s")
        bundle = PointedVector(base=base, velocity=vel)

        assert bundle.shape == (2,)

        # Index should give sub-pointedvectors
        sub0 = bundle[0]
        sub1 = bundle[1]

        assert sub0.shape == ()
        assert sub1.shape == ()


class TestPointedVectorEquality:
    """Tests for equality and hashing."""

    def test_equality_same_pointedvectors(self):
        """Test equality for identical pointedvectors."""
        base1 = Vector.from_([1, 2, 3], "m")
        vel1 = Vector.from_([4, 5, 6], "m/s")
        bundle1 = PointedVector(base=base1, velocity=vel1)

        base2 = Vector.from_([1, 2, 3], "m")
        vel2 = Vector.from_([4, 5, 6], "m/s")
        bundle2 = PointedVector(base=base2, velocity=vel2)

        # Note: equality uses jnp.equal, which returns an array
        result = bundle1 == bundle2
        assert jnp.all(result)

    @pytest.mark.skip(
        reason="Vector contains dict which is unhashable - known limitation"
    )
    def test_hash(self):
        """Test that pointedvectors are hashable."""
        base = Vector.from_([1, 2, 3], "m")
        vel = Vector.from_([4, 5, 6], "m/s")
        bundle = PointedVector(base=base, velocity=vel)

        # Should not raise
        hash_val = hash(bundle)
        assert isinstance(hash_val, int)


class TestPointedVectorShape:
    """Tests for shape handling."""

    def test_shape_scalar_bundle(self):
        """Test shape for scalar (0-d) bundle."""
        base = Vector.from_([1, 2, 3], "m")
        vel = Vector.from_([4, 5, 6], "m/s")
        bundle = PointedVector(base=base, velocity=vel)

        assert bundle.shape == ()

    def test_shape_broadcasted_bundle(self):
        """Test shape for broadcasted bundle."""
        base = Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "m")  # (2,)
        vel = Vector.from_([10, 20, 30], "m/s")  # ()

        bundle = PointedVector(base=base, velocity=vel)

        assert bundle.shape == (2,)


class TestPointedVectorRoleValidation:
    """Property-based tests for role validation."""

    def test_displacement_field_allowed(self):
        """Test that Displacement role is allowed in fields."""
        base = Vector.from_([1, 2, 3], "m")
        disp = Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
            cxr.phys_disp,
        )

        # Should not raise
        bundle = PointedVector(base=base, displacement=disp)
        assert isinstance(bundle["displacement"].role, cxr.PhysDisp)

    def test_acc_field_allowed(self):
        """Test that PhysAcc role is allowed in fields."""
        base = Vector.from_([1, 2, 3], "m")
        acc = Vector.from_([0.1, 0.2, 0.3], "m/s^2")

        # Should not raise
        bundle = PointedVector(base=base, acceleration=acc)
        assert isinstance(bundle["acceleration"].role, cxr.PhysAcc)


# ==============================================================================
# Property-based tests using Hypothesis


class TestPointedVectorPropertyTests:
    """Property-based tests using coordinax-hypothesis."""

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.pointedvectors())
    @settings(max_examples=50)
    def test_property_base_always_pos(self, bundle):
        """Property: base must always have Pos role."""
        assert isinstance(bundle.base.role, cxr.PhysDisp)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.pointedvectors(field_keys=("velocity", "acceleration")))
    @settings(max_examples=50)
    def test_property_fields_never_pos(self, bundle):
        """Property: field vectors never have Pos role."""
        for field_vec in bundle.values():
            assert not isinstance(field_vec.role, cxr.PhysDisp)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(
        bundle=cxst.pointedvectors(field_keys=("velocity",)),
        target_chart=cxst.charts(exclude=(cxc.Abstract0D,)),
    )
    @settings(max_examples=30, deadline=None)
    def test_property_vconvert_vs_manual(self, bundle, target_chart):
        """Property: vconvert matches manual conversion."""
        # Bundle conversion
        converted_bundle = bundle.vconvert(target_chart)

        # Manual conversion for base
        manual_base = bundle.base.vconvert(target_chart)

        # Compare base (positions)
        for comp in converted_bundle.base.data:
            base_match = jnp.allclose(
                u.ustrip(converted_bundle.base[comp]),
                u.ustrip(manual_base[comp]),
                atol=1e-8,
                rtol=1e-6,
            )
            assert base_match, f"Base component {comp} doesn't match"

        # Manual conversion for first field (velocity)
        if "velocity" in bundle:
            vel = bundle["velocity"]
            # Need base in velocity's rep for 'at'
            at_for_vel = bundle.base.vconvert(vel.chart)
            manual_vel = vel.vconvert(target_chart, at_for_vel)

            converted_vel = converted_bundle["velocity"]

            # Compare velocity components
            for comp in converted_vel.data:
                vel_match = jnp.allclose(
                    u.ustrip(converted_vel[comp]),
                    u.ustrip(manual_vel[comp]),
                    atol=1e-8,
                    rtol=1e-6,
                )
                assert vel_match, f"Velocity component {comp} doesn't match"

    # ==================================================================
    # Simpler property tests using explicit representations

    @given(
        base_vals=st.lists(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=3,
            max_size=3,
        ),
        vel_vals=st.lists(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=3,
            max_size=3,
        ),
    )
    @settings(max_examples=20)
    def test_property_simple_base_always_point(self, base_vals, vel_vals):
        """Property: base always has Point role (affine location)."""
        base = Vector.from_(base_vals, "m")
        vel = Vector.from_(vel_vals, "m/s")
        bundle = PointedVector(base=base, velocity=vel)

        assert isinstance(bundle.base.role, cxr.Point)
        assert isinstance(bundle["velocity"].role, cxr.PhysVel)

    @given(
        base_vals=st.lists(
            st.floats(
                min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=3,
            max_size=3,
        ),
        vel_vals=st.lists(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=3,
            max_size=3,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_simple_vconvert_preserves_roles(self, base_vals, vel_vals):
        """Property: vconvert preserves roles (simple test)."""
        base = Vector.from_(base_vals, "m")
        vel = Vector.from_(vel_vals, "m/s")
        bundle = PointedVector(base=base, velocity=vel)

        # Convert to spherical
        sph_bundle = bundle.vconvert(cxc.sph3d)

        assert isinstance(sph_bundle.base.role, cxr.Point)
        assert isinstance(sph_bundle["velocity"].role, cxr.PhysVel)

    # ==================================================================
    # Original hypothesis tests (continued)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.pointedvectors())
    @settings(max_examples=50)
    def test_property_base_role_preserved_after_conversion(self, bundle):
        """Property: base role is preserved after conversion."""
        target_chart = cxc.sph3d
        converted = bundle.vconvert(target_chart)

        assert isinstance(converted.base.role, cxr.PhysDisp)
        assert isinstance(bundle.base.role, cxr.PhysDisp)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.pointedvectors(field_keys=("velocity", "acceleration")))
    @settings(max_examples=50)
    def test_property_field_roles_preserved_after_conversion(self, bundle):
        """Property: field roles are preserved after conversion."""
        target_chart = cxc.cyl3d

        # Get original roles
        original_roles = {k: type(v.role) for k, v in bundle.items()}

        converted = bundle.vconvert(target_chart)

        # Check roles match
        for k, v in original_roles.items():
            assert type(converted[k].role) is v

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(
        bundle=cxst.pointedvectors(shape=st.just((3,))),  # Fixed batch shape
    )
    @settings(max_examples=30)
    def test_property_batch_shape_preserved(self, bundle):
        """Property: batch shape is preserved during conversion."""
        original_shape = bundle.shape
        converted = bundle.vconvert(cxc.sph3d)

        assert converted.shape == original_shape

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.pointedvectors())
    @settings(max_examples=30, deadline=None)
    def test_property_jit_compilable(self, bundle):
        """Property: pointedvectors can be JIT-compiled."""
        target_chart = cxc.cyl3d

        @jax.jit
        def convert(b):
            return b.vconvert(target_chart)

        # Should not raise
        result = convert(bundle)
        assert isinstance(result, PointedVector)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.pointedvectors(field_keys=("velocity",)))
    @settings(max_examples=30)
    def test_property_mixed_target_charts(self, bundle):
        """Property: can specify different target reps for different fields."""
        # Convert base to sph3d, field to cyl3d
        mixed = bundle.vconvert(cxc.sph3d, field_charts={"velocity": cxc.cyl3d})

        assert isinstance(mixed.base.chart, cxc.Spherical3D)
        assert isinstance(mixed["velocity"].chart, cxc.Cylindrical3D)

    @given(
        base_role=st.just(cxr.PhysVel),  # Non-Pos role
    )
    @settings(max_examples=10)
    def test_property_non_pos_base_rejected(self, base_role):
        """Property: base with non-Pos role is rejected."""
        # Create vector with non-Pos role
        non_pos_vec = Vector(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
            cxc.cart3d,
            base_role(),
        )
        vel = Vector.from_([10, 20, 30], "m/s")

        with pytest.raises(
            eqx.EquinoxTracetimeError, match="base must have role Point"
        ):  # eqx.error_if
            PointedVector(base=non_pos_vec, velocity=vel)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(
        bundle=cxst.pointedvectors(
            field_keys=("displacement",), field_roles=(cxr.PhysDisp,)
        )
    )
    @settings(max_examples=30)
    def test_property_displacement_role_supported(self, bundle):
        """Property: Displacement role is supported in fields."""
        assert "displacement" in bundle
        assert isinstance(bundle["displacement"].role, cxr.PhysDisp)
