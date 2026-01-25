"""Tests for metric resolution."""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.metrics as cxm
import coordinax.transforms as cxt


class TestMetricOf:
    """Tests for metric_of function."""

    def test_euclidean_3d(self) -> None:
        """Test metric_of for 3D Euclidean chart."""
        metric = cxm.metric_of(cxc.cart3d)
        assert isinstance(metric, cxm.EuclideanMetric)
        assert metric.signature == (1, 1, 1)

    def test_cylindrical_3d(self) -> None:
        """Test metric_of for cylindrical 3D chart."""
        metric = cxm.metric_of(cxc.cyl3d)
        assert isinstance(metric, cxm.EuclideanMetric)
        assert metric.signature == (1, 1, 1)

    def test_spacetime_euclidean(self) -> None:
        """Test metric_of for SpaceTimeEuclidean chart."""
        chart = cxc.SpaceTimeEuclidean(cxc.cyl3d)
        metric = cxm.metric_of(chart)
        assert isinstance(metric, cxm.EuclideanMetric)
        assert metric.signature == (1, 1, 1, 1)

    def test_spacetime_ct(self) -> None:
        """Test metric_of for SpaceTimeCT chart."""
        chart = cxc.SpaceTimeCT(cxc.cyl3d)
        metric = cxm.metric_of(chart)
        assert isinstance(metric, cxm.MinkowskiMetric)
        assert metric.signature == (-1, 1, 1, 1)

    def test_twosphere(self) -> None:
        """Test metric_of for 2-sphere chart."""
        metric = cxm.metric_of(cxc.twosphere)
        assert isinstance(metric, cxm.SphereMetric)
        assert metric.signature == (1, 1)


class TestNorm:
    """Tests for norm function."""

    # -----------------------------------------------------------------
    # Euclidean norm tests

    def test_euclidean_3d_pythagorean(self) -> None:
        """Test classic 3-4-5 Pythagorean triple."""
        metric = cxm.metric_of(cxc.cart3d)
        v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
        result = cxm.norm(metric, cxc.cart3d, v)
        assert jnp.allclose(u.ustrip("m", result), 5.0)

    def test_euclidean_3d_unit_vector(self) -> None:
        """Test unit vector has norm 1."""
        metric = cxm.metric_of(cxc.cart3d)
        v = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
        result = cxm.norm(metric, cxc.cart3d, v)
        assert jnp.allclose(u.ustrip("m", result), 1.0)

    def test_euclidean_3d_diagonal(self) -> None:
        """Test norm of (1, 1, 1) is sqrt(3)."""
        metric = cxm.metric_of(cxc.cart3d)
        v = {"x": u.Q(1, "km"), "y": u.Q(1, "km"), "z": u.Q(1, "km")}
        result = cxm.norm(metric, cxc.cart3d, v)
        assert jnp.allclose(u.ustrip("km", result), jnp.sqrt(3.0))

    def test_euclidean_2d(self) -> None:
        """Test 2D Euclidean norm."""
        metric = cxm.metric_of(cxc.cart2d)
        v = {"x": u.Q(3, "m"), "y": u.Q(4, "m")}
        result = cxm.norm(metric, cxc.cart2d, v)
        assert jnp.allclose(u.ustrip("m", result), 5.0)

    def test_euclidean_1d(self) -> None:
        """Test 1D Euclidean norm is absolute value."""
        metric = cxm.metric_of(cxc.cart1d)
        v = {"x": u.Q(-5, "m")}
        result = cxm.norm(metric, cxc.cart1d, v)
        assert jnp.allclose(u.ustrip("m", result), 5.0)

    def test_euclidean_with_at_parameter(self) -> None:
        """Test that at parameter is accepted but ignored for Euclidean."""
        metric = cxm.metric_of(cxc.cart3d)
        v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
        p = {"x": u.Q(100, "km"), "y": u.Q(200, "km"), "z": u.Q(300, "km")}
        result = cxm.norm(metric, cxc.cart3d, v, at=p)
        # Result should be same regardless of position
        assert jnp.allclose(u.ustrip("m", result), 5.0)

    def test_euclidean_velocity_units(self) -> None:
        """Test that velocity units are preserved."""
        metric = cxm.metric_of(cxc.cart3d)
        v = {"x": u.Q(3, "m/s"), "y": u.Q(4, "m/s"), "z": u.Q(0, "m/s")}
        result = cxm.norm(metric, cxc.cart3d, v)
        assert jnp.allclose(u.ustrip("m/s", result), 5.0)

    def test_euclidean_zero_vector(self) -> None:
        """Test norm of zero vector is zero."""
        metric = cxm.metric_of(cxc.cart3d)
        v = {"x": u.Q(0, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
        result = cxm.norm(metric, cxc.cart3d, v)
        assert jnp.allclose(u.ustrip("m", result), 0.0)

    # -----------------------------------------------------------------
    # Sphere metric tests

    def test_sphere_equator_equal_components(self) -> None:
        """At equator, theta and phi components contribute equally."""
        metric = cxm.metric_of(cxc.twosphere)
        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")}
        v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(1, "rad/s")}
        result = cxm.norm(metric, cxc.twosphere, v, at=p)
        # At equator sin(theta)=1, so g_phiphi=1, norm = sqrt(1+1) = sqrt(2)
        assert jnp.allclose(u.ustrip("rad/s", result), jnp.sqrt(2.0), atol=1e-5)

    def test_sphere_near_pole_phi_vanishes(self) -> None:
        """Near pole, phi component contribution vanishes."""
        metric = cxm.metric_of(cxc.twosphere)
        # Very close to north pole
        p = {"theta": u.Angle(0.001, "rad"), "phi": u.Angle(0, "rad")}
        v = {"theta": u.Q(0, "rad/s"), "phi": u.Q(1, "rad/s")}
        result = cxm.norm(metric, cxc.twosphere, v, at=p)
        # sin(0.001) ≈ 0.001, so g_phiphi ≈ 0, norm ≈ 0
        assert u.ustrip("rad/s", result) < 0.01

    def test_sphere_theta_only(self) -> None:
        """Pure theta motion has norm 1 everywhere."""
        metric = cxm.metric_of(cxc.twosphere)
        # At various latitudes
        for theta_val in [0.1, jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4]:
            p = {"theta": u.Angle(theta_val, "rad"), "phi": u.Angle(0, "rad")}
            v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(0, "rad/s")}
            result = cxm.norm(metric, cxc.twosphere, v, at=p)
            assert jnp.allclose(u.ustrip("rad/s", result), 1.0, atol=1e-5)

    # -----------------------------------------------------------------
    # Batched operations

    def test_euclidean_batched(self) -> None:
        """Test that norm works with batched inputs."""
        metric = cxm.metric_of(cxc.cart3d)
        # Batch of 3 vectors
        v = {
            "x": u.Q(jnp.array([3, 0, 1]), "m"),
            "y": u.Q(jnp.array([4, 1, 0]), "m"),
            "z": u.Q(jnp.array([0, 0, 0]), "m"),
        }
        result = cxm.norm(metric, cxc.cart3d, v)
        expected = jnp.array([5.0, 1.0, 1.0])
        assert jnp.allclose(u.ustrip("m", result), expected)


class TestMinkowskiInnerProduct:
    """Tests for Minkowski metric invariance."""

    @pytest.mark.skip(reason="TODO: physical_tangent_transform for SpaceTimeCT")
    def test_invariant_under_coord_transform(self) -> None:
        """Test that Minkowski inner product is invariant under coord transform."""
        chart_from = cxc.SpaceTimeCT(cxc.cart3d)
        chart_to = cxc.SpaceTimeCT(cxc.cyl3d)

        p = {
            "ct": u.Q(1.0, "km"),
            "x": u.Q(2.0, "km"),
            "y": u.Q(3.0, "km"),
            "z": u.Q(4.0, "km"),
        }

        v = {
            "ct": u.Q(0.1, "km/s"),
            "x": u.Q(1.0, "km/s"),
            "y": u.Q(-2.0, "km/s"),
            "z": u.Q(0.5, "km/s"),
        }

        v_to = cxt.physical_tangent_transform(chart_to, chart_from, v, p)

        def pack(chart, vals):
            unit = u.unit_of(vals["ct"])
            return jnp.stack(
                [u.uconvert(unit, vals[k]).value for k in chart.components]
            )

        v_from = pack(chart_from, v)
        B_from = cxt.frame_cart(chart_from, at=p)
        v_cart = cxt.pushforward(B_from, v_from)

        p_to = cxt.point_transform(chart_to, chart_from, p)
        v_to_vals = pack(chart_to, v_to)
        B_to = cxt.frame_cart(chart_to, at=p_to)
        v_cart2 = cxt.pushforward(B_to, v_to_vals)

        metric = cxm.metric_of(chart_from)
        eta = metric.metric_matrix(chart_from, p)
        inner = jnp.einsum("i,ij,j->", v_cart, eta, v_cart)
        inner2 = jnp.einsum("i,ij,j->", v_cart2, eta, v_cart2)

        assert jnp.allclose(inner, inner2, atol=1e-6)


# Keep old test functions for backwards compatibility
def test_metric_of_euclidean_3d() -> None:
    metric = cxm.metric_of(cxc.cart3d)
    assert isinstance(metric, cxm.EuclideanMetric)
    assert metric.signature == (1, 1, 1)


def test_metric_of_cylindrical_3d() -> None:
    metric = cxm.metric_of(cxc.cyl3d)
    assert isinstance(metric, cxm.EuclideanMetric)
    assert metric.signature == (1, 1, 1)


def test_metric_of_spacetime_euclidean() -> None:
    chart = cxc.SpaceTimeEuclidean(cxc.cyl3d)
    metric = cxm.metric_of(chart)
    assert isinstance(metric, cxm.EuclideanMetric)
    assert metric.signature == (1, 1, 1, 1)


def test_metric_of_spacetime_ct() -> None:
    chart = cxc.SpaceTimeCT(cxc.cyl3d)
    metric = cxm.metric_of(chart)
    assert isinstance(metric, cxm.MinkowskiMetric)
    assert metric.signature == (-1, 1, 1, 1)


def test_metric_of_twosphere() -> None:
    metric = cxm.metric_of(cxc.twosphere)
    assert isinstance(metric, cxm.SphereMetric)
    assert metric.signature == (1, 1)


@pytest.mark.skip(reason="physical_tangent_transform not implemented for SpaceTimeCT")
def test_minkowski_inner_product_invariant() -> None:
    chart_from = cxc.SpaceTimeCT(cxc.cart3d)
    chart_to = cxc.SpaceTimeCT(cxc.cyl3d)

    p = {
        "ct": u.Q(1.0, "km"),
        "x": u.Q(2.0, "km"),
        "y": u.Q(3.0, "km"),
        "z": u.Q(4.0, "km"),
    }

    v = {
        "ct": u.Q(0.1, "km/s"),
        "x": u.Q(1.0, "km/s"),
        "y": u.Q(-2.0, "km/s"),
        "z": u.Q(0.5, "km/s"),
    }

    v_to = cxt.physical_tangent_transform(chart_to, chart_from, v, p)

    def pack(chart, vals):
        unit = u.unit_of(vals["ct"])
        return jnp.stack([u.uconvert(unit, vals[k]).value for k in chart.components])

    v_from = pack(chart_from, v)
    B_from = cxt.frame_cart(chart_from, at=p)
    v_cart = cxt.pushforward(B_from, v_from)

    p_to = cxt.point_transform(chart_to, chart_from, p)
    v_to_vals = pack(chart_to, v_to)
    B_to = cxt.frame_cart(chart_to, at=p_to)
    v_cart2 = cxt.pushforward(B_to, v_to_vals)

    metric = cxm.metric_of(chart_from)
    eta = metric.metric_matrix(chart_from, p)
    inner = jnp.einsum("i,ij,j->", v_cart, eta, v_cart)
    inner2 = jnp.einsum("i,ij,j->", v_cart2, eta, v_cart2)

    assert jnp.allclose(inner, inner2, atol=1e-6)
