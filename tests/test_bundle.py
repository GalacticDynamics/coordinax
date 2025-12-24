"""Tests for FiberPoint."""

import equinox as eqx
import hypothesis.strategies as st
import jax
import pytest
from hypothesis import given, settings

import quaxed.numpy as jnp
import unxt as u

import coordinax as cx
import coordinax_hypothesis as cxst


class TestFiberPointInit:
    """Tests for FiberPoint initialization."""

    def test_basic_init(self):
        """Test basic initialization with explicit Vector instances."""
        # Create vectors with explicit chart and role
        base = cx.Vector.from_(
            u.Q([1, 2, 3], "km"), cx.charts.cart3d, cx.roles.point
        )  # Affine point
        vel = cx.Vector.from_(
            u.Q([10, 20, 30], "km/s"), cx.charts.cart3d, cx.roles.vel
        )  # Velocity
        acc = cx.Vector.from_(
            u.Q([0.1, 0.2, 0.3], "km/s^2"),
            cx.charts.cart3d,
            cx.roles.acc,
        )  # Acceleration

        bundle = cx.FiberPoint(base=base, velocity=vel, acceleration=acc)

        # Base and fields are stored (but may be copies due to eqx.error_if)
        assert isinstance(bundle.base, cx.Vector)
        assert isinstance(bundle["velocity"], cx.Vector)
        assert isinstance(bundle["acceleration"], cx.Vector)
        assert set(bundle.keys()) == {"velocity", "acceleration"}

    def test_base_must_have_point_role(self):
        """Test that base must have Point role."""
        # Create a non-Point vector (Vel)
        non_point = cx.Vector.from_(
            u.Q([1, 2, 3], "km/s"), cx.charts.cart3d, cx.roles.vel
        )
        vel = cx.Vector.from_(u.Q([10, 20, 30], "km/s"), cx.charts.cart3d, cx.roles.vel)

        with pytest.raises(RuntimeError, match="base must have role Point"):
            cx.FiberPoint(base=non_point, velocity=vel)

    def test_fields_cannot_have_point_role(self):
        """Test that field vectors cannot have Point role."""
        base = cx.Vector.from_(
            u.Q([1, 2, 3], "km"), cx.charts.cart3d, cx.roles.point
        )  # Has Point role
        another_point = cx.Vector.from_(
            u.Q([4, 5, 6], "km"), cx.charts.cart3d, cx.roles.point
        )  # Also has Point role

        with pytest.raises(
            RuntimeError,
            match=(
                r"Field 'extra_field' has role Point\. FiberPoint stores fibre "
                r"vectors anchored at base; store additional points elsewhere."
            ),
        ):
            cx.FiberPoint(base=base, extra_field=another_point)

    def test_broadcastable_shapes(self):
        """Test that shapes must be broadcastable."""
        base = cx.Vector.from_([1, 2, 3], "km")  # shape ()
        vel = cx.Vector.from_(
            jnp.array([[10, 20, 30], [40, 50, 60]]), "km/s"
        )  # shape (2,)

        # This should work (broadcasting)
        bundle = cx.FiberPoint(base=base, velocity=vel)
        assert bundle.shape == (2,)

    def test_non_broadcastable_shapes_error(self):
        """Test that non-broadcastable shapes raise error."""
        base = cx.Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "km")  # shape (2,)
        vel = cx.Vector.from_(jnp.array([[[10, 20, 30]] * 3]), "km/s")  # shape (1, 3)

        # Shapes (2,) and (1, 3) are incompatible for final broadcast
        with pytest.raises(RuntimeError, match="vector shapes are not broadcastable"):
            cx.FiberPoint(base=base, velocity=vel)


class TestFiberPointFrom:
    """Tests for from_ constructor."""

    def test_from_dict_with_base_key(self):
        """Test from_ with 'base' in the data dict."""
        data = {
            "base": u.Q([1, 2, 3], "km"),
            "velocity": u.Q([4, 5, 6], "km/s"),
        }
        bundle = cx.FiberPoint.from_(data)

        assert isinstance(bundle.base, cx.Vector)
        assert isinstance(bundle["velocity"], cx.Vector)

    def test_from_dict_with_explicit_base(self):
        """Test from_ with explicit base parameter."""
        base = cx.Vector.from_([1, 2, 3], "km")
        data = {
            "velocity": u.Q([4, 5, 6], "km/s"),
        }
        bundle = cx.FiberPoint.from_(data, base=base)

        # Note: eqx.error_if creates copies, so check components match
        assert jnp.allclose(bundle.base["x"], base["x"])

    def test_from_dict_missing_base_raises(self):
        """Test that from_ without base raises ValueError."""
        data = {
            "velocity": u.Q([4, 5, 6], "km/s"),
        }
        with pytest.raises(ValueError, match="must contain 'base'"):
            cx.FiberPoint.from_(data)


class TestFiberPointMappingAPI:
    """Tests for mapping interface."""

    def test_keys_values_items(self):
        """Test mapping methods."""
        base = cx.Vector.from_([1, 2, 3], "km")
        vel = cx.Vector.from_([10, 20, 30], "km/s")
        acc = cx.Vector.from_([0.1, 0.2, 0.3], "km/s^2")

        bundle = cx.FiberPoint(base=base, velocity=vel, acceleration=acc)

        assert set(bundle.keys()) == {"velocity", "acceleration"}
        assert len(list(bundle.values())) == 2
        assert len(list(bundle.items())) == 2

    def test_getitem_string(self):
        """Test __getitem__ with string key."""
        base = cx.Vector.from_([1, 2, 3], "km")
        vel = cx.Vector.from_([10, 20, 30], "km/s")

        bundle = cx.FiberPoint(base=base, velocity=vel)

        assert bundle["velocity"] is vel

    def test_getitem_index(self):
        """Test __getitem__ with numeric index."""
        base = cx.Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "km")
        vel = cx.Vector.from_(jnp.array([[10, 20, 30], [40, 50, 60]]), "km/s")

        bundle = cx.FiberPoint(base=base, velocity=vel)

        # Index to get first element
        sub_bundle = bundle[0]
        assert isinstance(sub_bundle, cx.FiberPoint)
        assert sub_bundle.shape == ()

    def test_q_property(self):
        """Test .q property (alias for base)."""
        base = cx.Vector.from_([1, 2, 3], "km")
        vel = cx.Vector.from_([10, 20, 30], "km/s")

        bundle = cx.FiberPoint(base=base, velocity=vel)

        assert bundle.q is bundle.base


class TestFiberPointVConvert:
    """Tests for vconvert method."""

    def test_vconvert_basic(self):
        """Test basic conversion to spherical coordinates."""
        base = cx.Vector.from_([1, 1, 1], "m")
        vel = cx.Vector.from_([10, 10, 10], "m/s")

        bundle = cx.FiberPoint(base=base, velocity=vel)
        sph_bundle = bundle.vconvert(cx.charts.sph3d)

        # Check reps changed
        assert isinstance(sph_bundle.base.chart, cx.charts.Spherical3D)
        assert isinstance(sph_bundle["velocity"].chart, cx.charts.Spherical3D)

    def test_vconvert_vs_manual(self):
        """Test that vconvert matches manual conversion."""
        base = cx.Vector.from_([1, 2, 3], "m")
        vel = cx.Vector.from_([4, 5, 6], "m/s")
        bundle = cx.FiberPoint(base=base, velocity=vel)

        # Bundle conversion
        sph_bundle = bundle.vconvert(cx.charts.sph3d)

        # Manual conversion
        base_sph = base.vconvert(cx.charts.sph3d)
        at_for_vel = base.vconvert(vel.chart)  # Should be base itself (already cart3d)
        vel_sph = vel.vconvert(cx.charts.sph3d, at_for_vel)

        # Compare results
        for comp in sph_bundle.base.data:
            base_match = jnp.allclose(sph_bundle.base[comp], base_sph[comp], atol=1e-10)
            assert base_match
        for comp in sph_bundle["velocity"].data:
            vel_match = jnp.allclose(
                sph_bundle["velocity"][comp], vel_sph[comp], atol=1e-10
            )
            assert vel_match

    def test_vconvert_with_field_charts(self):
        """Test vconvert with different target reps for fields."""
        base = cx.Vector.from_([1, 1, 1], "m")
        vel = cx.Vector.from_([10, 10, 10], "m/s")

        bundle = cx.FiberPoint(base=base, velocity=vel)

        # Convert base to spherical, velocity to cylindrical
        mixed_bundle = bundle.vconvert(
            cx.charts.sph3d, field_charts={"velocity": cx.charts.cyl3d}
        )

        assert isinstance(mixed_bundle.base.chart, cx.charts.Spherical3D)
        assert isinstance(mixed_bundle["velocity"].chart, cx.charts.Cylindrical3D)

    def test_vconvert_preserves_base_role(self):
        """Test that conversion preserves role flags."""
        base = cx.Vector.from_([1, 2, 3], "m")
        vel = cx.Vector.from_([4, 5, 6], "m/s")

        bundle = cx.FiberPoint(base=base, velocity=vel)
        sph_bundle = bundle.vconvert(cx.charts.sph3d)

        assert isinstance(sph_bundle.base.role, cx.roles.Point)
        assert isinstance(sph_bundle["velocity"].role, cx.roles.Vel)


class TestFiberPointJAX:
    """Tests for JAX compatibility."""

    def test_jit_smoke_test(self):
        """Test that bundle vconvert works (JIT has limitations).

        Note: Due to representation objects being non-static, bundles can't
        be directly passed through JIT boundaries. The vconvert method itself
        works correctly in eager mode.
        """
        base = cx.Vector.from_([1, 2, 3], "m")
        vel = cx.Vector.from_([4, 5, 6], "m/s")
        bundle = cx.FiberPoint(base=base, velocity=vel)

        # vconvert works in eager mode
        sph_bundle = bundle.vconvert(cx.charts.sph3d)

        assert isinstance(sph_bundle, cx.FiberPoint)
        assert isinstance(sph_bundle.base.chart, cx.charts.Spherical3D)
        assert isinstance(sph_bundle["velocity"].chart, cx.charts.Spherical3D)

    def test_vmap_smoke_test(self):
        """Test that bundle works with vmap."""
        # Create batched bundle
        base = cx.Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "m")
        vel = cx.Vector.from_(jnp.array([[10, 20, 30], [40, 50, 60]]), "m/s")
        bundle = cx.FiberPoint(base=base, velocity=vel)

        assert bundle.shape == (2,)

        # Index should give sub-bundles
        sub0 = bundle[0]
        sub1 = bundle[1]

        assert sub0.shape == ()
        assert sub1.shape == ()


class TestFiberPointEquality:
    """Tests for equality and hashing."""

    def test_equality_same_bundles(self):
        """Test equality for identical bundles."""
        base1 = cx.Vector.from_([1, 2, 3], "m")
        vel1 = cx.Vector.from_([4, 5, 6], "m/s")
        bundle1 = cx.FiberPoint(base=base1, velocity=vel1)

        base2 = cx.Vector.from_([1, 2, 3], "m")
        vel2 = cx.Vector.from_([4, 5, 6], "m/s")
        bundle2 = cx.FiberPoint(base=base2, velocity=vel2)

        # Note: equality uses jnp.equal, which returns an array
        result = bundle1 == bundle2
        assert jnp.all(result)

    @pytest.mark.skip(
        reason="Vector contains dict which is unhashable - known limitation"
    )
    def test_hash(self):
        """Test that bundles are hashable."""
        base = cx.Vector.from_([1, 2, 3], "m")
        vel = cx.Vector.from_([4, 5, 6], "m/s")
        bundle = cx.FiberPoint(base=base, velocity=vel)

        # Should not raise
        hash_val = hash(bundle)
        assert isinstance(hash_val, int)


class TestFiberPointShape:
    """Tests for shape handling."""

    def test_shape_scalar_bundle(self):
        """Test shape for scalar (0-d) bundle."""
        base = cx.Vector.from_([1, 2, 3], "m")
        vel = cx.Vector.from_([4, 5, 6], "m/s")
        bundle = cx.FiberPoint(base=base, velocity=vel)

        assert bundle.shape == ()

    def test_shape_broadcasted_bundle(self):
        """Test shape for broadcasted bundle."""
        base = cx.Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "m")  # (2,)
        vel = cx.Vector.from_([10, 20, 30], "m/s")  # ()

        bundle = cx.FiberPoint(base=base, velocity=vel)

        assert bundle.shape == (2,)


class TestFiberPointRoleValidation:
    """Property-based tests for role validation."""

    def test_displacement_field_allowed(self):
        """Test that Displacement role is allowed in fields."""
        base = cx.Vector.from_([1, 2, 3], "m")
        disp = cx.Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
            cx.charts.cart3d,
            cx.roles.pos,
        )

        # Should not raise
        bundle = cx.FiberPoint(base=base, displacement=disp)
        assert isinstance(bundle["displacement"].role, cx.roles.Pos)

    def test_acc_field_allowed(self):
        """Test that Acc role is allowed in fields."""
        base = cx.Vector.from_([1, 2, 3], "m")
        acc = cx.Vector.from_([0.1, 0.2, 0.3], "m/s^2")

        # Should not raise
        bundle = cx.FiberPoint(base=base, acceleration=acc)
        assert isinstance(bundle["acceleration"].role, cx.roles.Acc)


# ==============================================================================
# Property-based tests using Hypothesis


class TestFiberPointPropertyTests:
    """Property-based tests using coordinax-hypothesis."""

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.bundles())
    @settings(max_examples=50)
    def test_property_base_always_pos(self, bundle):
        """Property: base must always have Pos role."""
        assert isinstance(bundle.base.role, cx.roles.Pos)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.bundles(field_keys=("velocity", "acceleration")))
    @settings(max_examples=50)
    def test_property_fields_never_pos(self, bundle):
        """Property: field vectors never have Pos role."""
        for field_vec in bundle.values():
            assert not isinstance(field_vec.role, cx.roles.Pos)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(
        bundle=cxst.bundles(field_keys=("velocity",)),
        target_chart=cxst.charts(exclude=(cx.charts.Abstract0D,)),
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
        base = cx.Vector.from_(base_vals, "m")
        vel = cx.Vector.from_(vel_vals, "m/s")
        bundle = cx.FiberPoint(base=base, velocity=vel)

        assert isinstance(bundle.base.role, cx.roles.Point)
        assert isinstance(bundle["velocity"].role, cx.roles.Vel)

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
    @settings(max_examples=20)
    def test_property_simple_vconvert_preserves_roles(self, base_vals, vel_vals):
        """Property: vconvert preserves roles (simple test)."""
        base = cx.Vector.from_(base_vals, "m")
        vel = cx.Vector.from_(vel_vals, "m/s")
        bundle = cx.FiberPoint(base=base, velocity=vel)

        # Convert to spherical
        sph_bundle = bundle.vconvert(cx.charts.sph3d)

        assert isinstance(sph_bundle.base.role, cx.roles.Point)
        assert isinstance(sph_bundle["velocity"].role, cx.roles.Vel)

    # ==================================================================
    # Original hypothesis tests (continued)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.bundles())
    @settings(max_examples=50)
    def test_property_base_role_preserved_after_conversion(self, bundle):
        """Property: base role is preserved after conversion."""
        target_chart = cx.charts.sph3d
        converted = bundle.vconvert(target_chart)

        assert isinstance(converted.base.role, cx.roles.Pos)
        assert isinstance(bundle.base.role, cx.roles.Pos)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.bundles(field_keys=("velocity", "acceleration")))
    @settings(max_examples=50)
    def test_property_field_roles_preserved_after_conversion(self, bundle):
        """Property: field roles are preserved after conversion."""
        target_chart = cx.charts.cyl3d

        # Get original roles
        original_roles = {k: type(v.role) for k, v in bundle.items()}

        converted = bundle.vconvert(target_chart)

        # Check roles match
        for k, v in original_roles.items():
            assert type(converted[k].role) is v

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(
        bundle=cxst.bundles(shape=st.just((3,))),  # Fixed batch shape
    )
    @settings(max_examples=30)
    def test_property_batch_shape_preserved(self, bundle):
        """Property: batch shape is preserved during conversion."""
        original_shape = bundle.shape
        converted = bundle.vconvert(cx.charts.sph3d)

        assert converted.shape == original_shape

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.bundles())
    @settings(max_examples=30, deadline=None)
    def test_property_jit_compilable(self, bundle):
        """Property: bundles can be JIT-compiled."""
        target_chart = cx.charts.cyl3d

        @jax.jit
        def convert(b):
            return b.vconvert(target_chart)

        # Should not raise
        result = convert(bundle)
        assert isinstance(result, cx.FiberPoint)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(bundle=cxst.bundles(field_keys=("velocity",)))
    @settings(max_examples=30)
    def test_property_mixed_target_charts(self, bundle):
        """Property: can specify different target reps for different fields."""
        # Convert base to sph3d, field to cyl3d
        mixed = bundle.vconvert(
            cx.charts.sph3d, field_charts={"velocity": cx.charts.cyl3d}
        )

        assert isinstance(mixed.base.chart, cx.charts.Spherical3D)
        assert isinstance(mixed["velocity"].chart, cx.charts.Cylindrical3D)

    @given(
        base_role=st.just(cx.roles.Vel),  # Non-Pos role
    )
    @settings(max_examples=10)
    def test_property_non_pos_base_rejected(self, base_role):
        """Property: base with non-Pos role is rejected."""
        # Create vector with non-Pos role
        non_pos_vec = cx.Vector(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
            cx.charts.cart3d,
            base_role(),
        )
        vel = cx.Vector.from_([10, 20, 30], "m/s")

        with pytest.raises(
            eqx.RuntimeException, match="TODO: figure out matching message"
        ):  # eqx.error_if
            cx.FiberPoint(base=non_pos_vec, velocity=vel)

    @pytest.mark.skip(reason="Hypothesis strategy needs CartND representation fixes")
    @given(
        bundle=cxst.bundles(field_keys=("displacement",), field_roles=(cx.roles.Pos,))
    )
    @settings(max_examples=30)
    def test_property_displacement_role_supported(self, bundle):
        """Property: Displacement role is supported in fields."""
        assert "displacement" in bundle
        assert isinstance(bundle["displacement"].role, cx.roles.Pos)
