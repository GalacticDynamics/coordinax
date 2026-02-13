"""Test vectors."""

import jax.tree as jtu
import pytest
from hypothesis import given, settings

import quaxed.numpy as jnp
import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.roles as cxr
import coordinax_hypothesis as cxst


@given(
    vec_and_chart=cxst.vectors_with_target_chart(chart=cxst.charts(), role=cxr.Point)
)
def test_position_vector_vconvert_roundtrip(vec_and_chart):
    """Test vconvert roundtrip for position vectors."""
    vec, target_chain = vec_and_chart

    for chart in target_chain:
        converted = vec.vconvert(chart)
        assert isinstance(converted, cx.Vector)
        assert converted.chart == chart

        roundtripped = converted.vconvert(vec.chart)
        assert isinstance(roundtripped, cx.Vector)
        assert roundtripped.chart == vec.chart
        assert roundtripped.role == vec.role
        # Compare data values (not the vectors themselves)
        assert all(
            jtu.map(
                jnp.allclose,
                list(vec.data.values()),
                list(roundtripped.data.values()),
            )
        )


@settings(max_examples=10, deadline=None)
@given(
    vec_and_chart=cxst.vectors_with_target_chart(chart=cxst.charts(), role=cxr.Point)
)
def test_position_represent_as_matches_vconvert(vec_and_chart):
    """Test that represent_as matches vconvert for position vectors."""
    vec, target_chain = vec_and_chart

    for chart in target_chain:
        # Test that represent_as produces same result as vconvert
        result_represent_as = vec.represent_as(chart)
        result_vconvert = vec.vconvert(chart)

        assert isinstance(result_represent_as, cx.Vector)
        assert isinstance(result_vconvert, cx.Vector)
        assert result_represent_as.chart == result_vconvert.chart
        assert result_represent_as.role == result_vconvert.role
        # Compare data values only, not the entire pytree
        assert all(
            jtu.map(
                jnp.allclose,
                list(result_represent_as.data.values()),
                list(result_vconvert.data.values()),
            )
        )


@settings(max_examples=10, deadline=None)
@given(
    vec_and_chart=cxst.vectors_with_target_chart(chart=cxst.charts(), role=cxr.PhysVel)
)
def test_velocity_represent_as_matches_vconvert(vec_and_chart):
    """Test that represent_as matches vconvert for velocity vectors."""
    vec, target_chain = vec_and_chart

    # Create a position vector for the differential transformation
    pos = cx.Vector.from_([1.0, 2.0, 3.0], "m")
    pos_in_vec_chart = pos.vconvert(vec.chart)

    for chart in target_chain:
        # Test that represent_as produces same result as vconvert
        result_represent_as = vec.represent_as(chart, pos_in_vec_chart)
        result_vconvert = vec.vconvert(chart, pos_in_vec_chart)

        assert isinstance(result_represent_as, cx.Vector)
        assert isinstance(result_vconvert, cx.Vector)
        assert result_represent_as.chart == result_vconvert.chart
        assert result_represent_as.role == result_vconvert.role
        # Compare data values only, not the entire pytree
        assert all(
            jtu.map(
                jnp.allclose,
                list(result_represent_as.data.values()),
                list(result_vconvert.data.values()),
            )
        )


@settings(max_examples=10, deadline=None)
@given(
    vec_and_chart=cxst.vectors_with_target_chart(chart=cxst.charts(), role=cxr.PhysAcc)
)
def test_acceleration_represent_as_matches_vconvert(vec_and_chart):
    """Test that represent_as matches vconvert for acceleration vectors."""
    vec, target_chain = vec_and_chart

    # Create a position vector for the differential transformation
    pos = cx.Vector.from_([1.0, 2.0, 3.0], "m")
    pos_in_vec_chart = pos.vconvert(vec.chart)

    for chart in target_chain:
        # Test that represent_as produces same result as vconvert
        result_represent_as = vec.represent_as(chart, pos_in_vec_chart)
        result_vconvert = cx.vconvert(chart, vec, pos_in_vec_chart)

        assert isinstance(result_represent_as, cx.Vector)
        assert isinstance(result_vconvert, cx.Vector)
        assert result_represent_as.chart == result_vconvert.chart
        assert result_represent_as.role == result_vconvert.role
        assert all(
            jtu.map(
                jnp.allclose,
                list(result_represent_as.data.values()),
                list(result_vconvert.data.values()),
            )
        )


# =============================================================================
# Test vector addition with role semantics


class TestPos:
    """Tests for the Pos role (position-difference / physical displacement)."""

    def test_disp_role_exists(self):
        """Test that Pos role is accessible."""
        assert hasattr(cxc.cart3d, "PhysDisp")
        assert hasattr(cxc.cart3d, "phys_disp")
        assert isinstance(cxr.phys_disp, cxr.PhysDisp)

    def test_disp_order(self):
        """Test that Pos has order 0 (same as Point)."""
        assert cxr.PhysDisp.order == 0
        assert cxr.Point.order == 0

    def test_disp_derivative(self):
        """Test that derivative of Pos is PhysVel."""
        assert isinstance(cxr.phys_disp.derivative(), cxr.PhysVel)

    def test_create_pos_vector(self):
        """Test creating a vector with Pos role."""
        disp = cx.Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
            cxr.phys_disp,
        )
        assert isinstance(disp.role, cxr.PhysDisp)


class TestIsEuclideanRep:
    """Tests for is_euclidean property."""

    def test_cartesian_is_euclidean(self):
        """Test that Cartesian representations are Euclidean."""
        assert cxc.cart3d.is_euclidean is True
        assert cxc.cart2d.is_euclidean is True

    def test_spherical_is_euclidean(self):
        """Test that spherical coordinates in Euclidean space are Euclidean."""
        # Spherical coordinates on R^3 are still Euclidean space
        assert cxc.sph3d.is_euclidean is True

    def test_twosphere_is_not_euclidean(self):
        """Test that TwoSphere (intrinsic manifold) is not Euclidean."""
        assert cxc.twosphere.is_euclidean is False


class TestVectorAddition:
    """Tests for vector addition with role semantics."""

    def test_disp_plus_pos(self):
        """Test that Pos + Pos = PhysDisp."""
        d1 = cx.Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
            cxr.phys_disp,
        )
        d2 = cx.Vector(
            {"x": u.Q(0.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
            cxr.phys_disp,
        )
        result = d1.add(d2)
        assert isinstance(result.role, cxr.PhysDisp)
        assert jnp.allclose(u.ustrip("m", result["x"]), 1.0)
        assert jnp.allclose(u.ustrip("m", result["y"]), 2.0)
        assert jnp.allclose(u.ustrip("m", result["z"]), 0.0)

    def test_point_plus_pos(self):
        """Test that Point + Pos = Point."""
        point = cx.Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(1.0, "m")},
            cxc.cart3d,
            cxr.point,
        )
        disp = cx.Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")},
            cxc.cart3d,
            cxr.phys_disp,
        )
        result = point.add(disp)
        assert isinstance(result.role, cxr.Point)
        assert jnp.allclose(u.ustrip("m", result["x"]), 2.0)
        assert jnp.allclose(u.ustrip("m", result["y"]), 3.0)
        assert jnp.allclose(u.ustrip("m", result["z"]), 4.0)

    def test_point_plus_point_raises(self):
        """Test that Point + Point raises TypeError."""
        point1 = cx.Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
            cxr.point,
        )
        point2 = cx.Vector(
            {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
            cxr.point,
        )
        with pytest.raises(TypeError, match="Cannot add Point \\+ Point"):
            point1.add(point2)

    def test_disp_plus_point_raises(self):
        """Test that Pos + Point raises TypeError."""
        disp = cx.Vector(
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
            cxr.phys_disp,
        )
        point = cx.Vector(
            {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
            cxr.point,
        )
        with pytest.raises(TypeError, match="Cannot add Pos \\+ Point"):
            disp.add(point)


class TestAsDisplacement:
    """Tests for as_disp function."""

    def test_as_disp_no_origin(self):
        """Test as_disp with no origin (from coordinate origin)."""
        pos = cx.Vector.from_([1, 2, 3], "m")
        disp = cx.as_disp(pos)
        assert isinstance(disp.role, cxr.PhysDisp)
        assert jnp.allclose(u.ustrip("m", disp["x"]), 1.0)
        assert jnp.allclose(u.ustrip("m", disp["y"]), 2.0)
        assert jnp.allclose(u.ustrip("m", disp["z"]), 3.0)

    def test_as_disp_with_origin(self):
        """Test as_disp with explicit origin."""
        pos = cx.Vector.from_([3, 4, 5], "m")
        origin = cx.Vector.from_([1, 2, 3], "m")
        disp = cx.as_disp(pos, origin)
        assert isinstance(disp.role, cxr.PhysDisp)
        assert jnp.allclose(u.ustrip("m", disp["x"]), 2.0)
        assert jnp.allclose(u.ustrip("m", disp["y"]), 2.0)
        assert jnp.allclose(u.ustrip("m", disp["z"]), 2.0)

    def test_as_disp_requires_pos_role(self):
        """Test that as_disp requires Pos role."""
        vel = cx.Vector(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
            cxc.cart3d,
            cxr.phys_vel,
        )
        with pytest.raises(TypeError, match="Cannot convert vector with role"):
            cx.as_disp(vel)

    def test_as_disp_with_chart_parameter(self):
        """Test as_disp with target chart conversion."""
        # Create a position in Cartesian
        point = cx.Vector.from_([1, 0, 0], "m")
        origin = cx.Vector.from_([0, 0, 0], "m")

        # Request displacement in spherical chart
        disp_sph = cx.as_disp(point, origin, chart=cxc.sph3d, at=point)

        # Should be Displacement role in spherical rep
        assert isinstance(disp_sph.role, cxr.PhysDisp)
        assert disp_sph.chart == cxc.sph3d

        # The displacement should still represent the same physical vector
        # Convert back to Cartesian to verify (using positional from_pos)
        disp_cart = disp_sph.vconvert(cxc.cart3d, point)
        assert jnp.allclose(u.ustrip("m", disp_cart["x"]), 1.0)
        assert jnp.allclose(u.ustrip("m", disp_cart["y"]), 0.0)

    def test_as_disp_from_non_cartesian(self):
        """Test as_disp from a non-Cartesian position without explicit origin.

        This verifies that when origin=None, the position is converted to Cartesian
        first before being interpreted as a displacement.
        """
        # Create a position in spherical coordinates
        pos_sph = cx.Vector(
            {
                "r": u.Q(2.0, "m"),
                "theta": u.Q(jnp.pi / 2, "rad"),  # 90 degrees
                "phi": u.Q(0.0, "rad"),  # 0 degrees
            },
            cxc.sph3d,
            cxr.phys_disp,
        )

        # Convert to displacement (should convert to Cartesian first)
        disp = cx.as_disp(pos_sph)

        # Should be in Cartesian representation (the default for displacement
        # from origin)
        assert disp.chart == cxc.cart3d
        assert isinstance(disp.role, cxr.PhysDisp)

        # Verify the values match the Cartesian conversion of the spherical position
        # r=2, theta=pi/2, phi=0 => x=2, y=0, z=0
        assert jnp.allclose(u.ustrip("m", disp["x"]), 2.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("m", disp["y"]), 0.0, atol=1e-6)
        assert jnp.allclose(u.ustrip("m", disp["z"]), 0.0, atol=1e-6)


class TestVelVelAddition:
    """Tests for velocity + velocity addition (same role)."""

    def test_vel_plus_vel(self):
        """Test that PhysVel + PhysVel = PhysVel (same role addition)."""
        v1 = cx.Vector(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
            cxc.cart3d,
            cxr.phys_vel,
        )
        v2 = cx.Vector(
            {"x": u.Q(0.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(0.0, "m/s")},
            cxc.cart3d,
            cxr.phys_vel,
        )
        result = v1.add(v2)
        assert isinstance(result.role, cxr.PhysVel)
        assert jnp.allclose(u.ustrip("m/s", result["x"]), 1.0)
        assert jnp.allclose(u.ustrip("m/s", result["y"]), 2.0)


def test_displacement_addition_commutative():
    """Test that PhysDisp + PhysDisp is commutative."""
    d1 = cx.Vector(
        {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")},
        cxc.cart3d,
        cxr.phys_disp,
    )
    d2 = cx.Vector(
        {"x": u.Q(4.0, "m"), "y": u.Q(5.0, "m"), "z": u.Q(6.0, "m")},
        cxc.cart3d,
        cxr.phys_disp,
    )

    result1 = d1.add(d2)
    result2 = d2.add(d1)

    assert isinstance(result1.role, cxr.PhysDisp)
    assert isinstance(result2.role, cxr.PhysDisp)

    # Check values are close - need to strip units for comparison
    for key in result1.data:
        v1 = u.ustrip("m", result1[key])
        v2 = u.ustrip("m", result2[key])
        assert jnp.allclose(v1, v2, rtol=1e-5)


def test_displacement_transformation_via_tangent_transform():
    """Test that Displacement transforms via tangent_transform, not coord_map.

    This test verifies the geometric distinction:
    - Positions transform via point_transform: p_new = f(p_old)
    - Displacements transform via tangent_transform: v_new = J(p) · v_old

    CRITICAL: Displacement uses PHYSICAL components (uniform length units),
    not coordinate increments. In cylindrical coords:
    - Physical: (rho[m], phi[m], z[m]) where phi is tangential length
    - NOT: (Δrho[m], Δphi[rad], Δz[m]) ✗

    In cylindrical coords at position φ, the orthonormal frame is:
    - ê_ρ = (cos φ, sin φ, 0)
    - ê_φ = (-sin φ, cos φ, 0)
    - ê_z = (0, 0, 1)

    A physical displacement (rho=1m, phi=2m, z=0m) transforms to Cartesian as:
    d_cart = 1m * ê_ρ + 2m * ê_φ + 0m * ê_z
    """
    # Create a Position to define the base point
    pos_at_phi0 = cx.Vector(
        {"rho": u.Q(1.0, "m"), "phi": u.Q(0.0, "rad"), "z": u.Q(0.0, "m")},
        cxc.cyl3d,
        cxr.phys_disp,
    )

    # Create a physical Displacement: (rho=1m, phi=0m, z=0m)
    # At φ=0: ê_ρ = (1,0,0), ê_φ = (0,1,0)
    # Expected Cartesian: 1m*(1,0,0) + 0m*(0,1,0) = (1,0,0)m
    disp_at_phi0 = cx.Vector(
        {"rho": u.Q(1.0, "m"), "phi": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        cxc.cyl3d,
        cxr.phys_disp,
    )

    # Convert displacement to Cartesian using the base point
    disp_cart = disp_at_phi0.vconvert(cxc.cart3d, pos_at_phi0)

    # At φ=0, ê_ρ = (cos 0, sin 0) = (1, 0)
    # So rho=1m should give dx=1m, dy=0m
    assert isinstance(disp_cart.role, cxr.PhysDisp)
    assert jnp.allclose(u.ustrip("m", disp_cart["x"]), 1.0, atol=1e-10)
    assert jnp.allclose(u.ustrip("m", disp_cart["y"]), 0.0, atol=1e-10)
    assert jnp.allclose(u.ustrip("m", disp_cart["z"]), 0.0, atol=1e-10)

    # Now test at φ=90°: ê_ρ = (0, 1), ê_φ = (-1, 0)
    pos_at_phi90 = cx.Vector(
        {"rho": u.Q(1.0, "m"), "phi": u.Q(jnp.pi / 2, "rad"), "z": u.Q(0.0, "m")},
        cxc.cyl3d,
        cxr.phys_disp,
    )

    # Same physical displacement components (rho=1m, phi=0m, z=0m) at different position
    disp_at_phi90 = cx.Vector(
        {"rho": u.Q(1.0, "m"), "phi": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        cxc.cyl3d,
        cxr.phys_disp,
    )

    # Convert using the new base point
    disp_cart_90 = disp_at_phi90.vconvert(cxc.cart3d, pos_at_phi90)

    # At φ=90°, ê_ρ = (cos 90°, sin 90°) = (0, 1)
    # So rho=1m should give dx=0m, dy=1m
    assert isinstance(disp_cart_90.role, cxr.PhysDisp)
    assert jnp.allclose(u.ustrip("m", disp_cart_90["x"]), 0.0, atol=1e-10)
    assert jnp.allclose(u.ustrip("m", disp_cart_90["y"]), 1.0, atol=1e-10)
    assert jnp.allclose(u.ustrip("m", disp_cart_90["z"]), 0.0, atol=1e-10)

    # This demonstrates: same physical components, different base point
    # → different Cartesian result (basis vectors depend on position)


def test_disp_plus_displacement_with_coordinate_conversion():
    """Test Point + PhysDisp when representations differ.

    When adding Point (in one rep) + PhysDisp (in another rep), the
    displacement must be converted using the position as the base point.

    IMPORTANT: PhysDisp uses physical components (uniform length units).
    """
    # Position in Cartesian
    pos_cart = cx.Vector.from_([1.0, 0.0, 0.0], "m")  # At (x=1, y=0)

    # Physical displacement in cylindrical: (rho=0m, phi=1m, z=0m)
    # This is "1 meter in the tangential direction"
    disp_cyl = cx.Vector(
        {"rho": u.Q(0.0, "m"), "phi": u.Q(1.0, "m"), "z": u.Q(0.0, "m")},
        cxc.cyl3d,
        cxr.phys_disp,
    )

    # Add them: should convert displacement to Cartesian using pos_cart as base
    result = pos_cart.add(disp_cyl)

    # The position (1, 0, 0) in Cartesian is (ρ=1, φ=0, z=0) in cylindrical
    # At φ=0, ê_φ = (-sin 0, cos 0) = (0, 1)
    # Physical displacement: 0m*ê_ρ + 1m*ê_φ = (0, 1, 0)m in Cartesian
    # Result: (1, 0, 0) + (0, 1, 0) = (1, 1, 0)
    assert isinstance(result.role, cxr.PhysDisp)
    assert jnp.allclose(u.ustrip("m", result["x"]), 1.0, atol=1e-10)
    assert jnp.allclose(u.ustrip("m", result["y"]), 1.0, atol=1e-10)
    assert jnp.allclose(u.ustrip("m", result["z"]), 0.0, atol=1e-10)


def test_physdisp_plus_physdisp_different_charts_raises():
    """Test that PhysDisp + PhysDisp in different charts requires base point.

    Since displacements transform via tangent_transform (which needs a base point),
    adding displacements in different representations is ambiguous without
    specifying where the transformation should be evaluated.
    """
    disp_cart = cx.Vector(
        {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        cxc.cart3d,
        cxr.phys_disp,
    )
    # NOTE: Using angle unit (rad) here - this will be caught by validation when
    # we add it
    disp_cyl = cx.Vector(
        {"rho": u.Q(1.0, "m"), "phi": u.Q(0.0, "rad"), "z": u.Q(0.0, "m")},
        cxc.cyl3d,
        cxr.phys_disp,
    )

    with pytest.raises(
        ValueError, match="Cannot add displacements in different representations"
    ):
        disp_cart.add(disp_cyl)


@settings(max_examples=10, deadline=None)
@given(role=cxst.roles())
def test_roles_strategy(role):
    """Test that roles strategy generates valid roles."""
    assert isinstance(role, cxr.AbstractRole)
    assert role.order in {0, 1, 2}
