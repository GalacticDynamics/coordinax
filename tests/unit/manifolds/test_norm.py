"""Tests for the norm() function and AbstractVector.norm() dispatch.

Coverage includes Euclidean and spherical metric behavior, plus JIT/vmap usage.
"""

import jax
import jax.numpy as jnp
import pytest

import quaxed.numpy as qnp
import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm

# =============================================================================
# cxm.norm() standalone function
# =============================================================================


class TestNormEuclidean:
    """Tests for cxm.norm() on Euclidean manifolds with Cartesian coordinates."""

    @pytest.mark.parametrize(
        ("v", "chart", "exp"),
        [
            (
                {"x": u.Q(0, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")},
                cxc.cart3d,
                u.Q(0, "m"),
            ),
            (
                {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")},
                cxc.cart3d,
                u.Q(5, "m"),
            ),
            (
                {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")},
                cxc.cart3d,
                u.Q(1, "m"),
            ),
            (
                {"x": u.Q(1, "km"), "y": u.Q(1, "km"), "z": u.Q(1, "km")},
                cxc.cart3d,
                u.Q(jnp.sqrt(3), "km"),
            ),
            ({"x": u.Q(3, "m"), "y": u.Q(4, "m")}, cxc.cart2d, u.Q(5, "m")),
            ({"x": u.Q(-5, "m")}, cxc.cart1d, u.Q(5, "m")),
        ],
    )
    def test_known_cases(self, v, chart, exp):
        """Known cases: Pythagorean triples, unit vectors, etc."""
        got = cxm.norm(v, chart)
        assert qnp.allclose(got, exp, atol=u.Q(1e-10, exp.unit))

    def test_at_parameter_ignored_for_euclidean(self):
        v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
        p = {"x": u.Q(100, "km"), "y": u.Q(200, "km"), "z": u.Q(300, "km")}
        result = cxm.norm(v, cxc.cart3d, at=p)
        assert qnp.allclose(result, u.Q(5, "m"), atol=u.Q(1e-5, "m"))

    def test_velocity_units(self):
        v = {"x": u.Q(3, "m/s"), "y": u.Q(4, "m/s"), "z": u.Q(0, "m/s")}
        result = cxm.norm(v, cxc.cart3d)
        assert qnp.allclose(result, u.Q(5, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_jit(self):
        @jax.jit
        def compute(v):
            return cxm.norm(v, cxc.cart3d)

        v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
        result = compute(v)
        assert qnp.allclose(result, u.Q(5, "m"), atol=u.Q(1e-5, "m"))

    def test_batched(self):
        """cxm.norm follows scalar-first design; use vmap for batched inputs."""
        v_batch = {
            "x": u.Q(jnp.array([3, 0, 1]), "m"),
            "y": u.Q(jnp.array([4, 1, 0]), "m"),
            "z": u.Q(jnp.array([0, 0, 0]), "m"),
        }
        result = jax.vmap(lambda v: cxm.norm(v, cxc.cart3d))(v_batch)
        expected = jnp.array([5, 1, 1])
        assert qnp.allclose(result, u.Q(expected, "m"), atol=u.Q(1e-5, "m"))


class TestNormSphere:
    """Tests for cxm.norm() on the 2-sphere with spherical coordinates."""

    def test_equator_equal_components(self):
        """At equator sin(theta)=1, norm = sqrt(1 + 1) = sqrt(2)."""
        metric = cxm.HyperSphericalMetric(ndim=2)
        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")}
        v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(1, "rad/s")}
        result = cxm.norm(v, metric, cxc.sph2, at=p)
        assert qnp.allclose(result, u.Q(jnp.sqrt(2), "rad/s"), atol=u.Q(1e-5, "rad/s"))

    def test_near_pole_phi_vanishes(self):
        """Near north pole, phi contribution -> 0."""
        metric = cxm.HyperSphericalMetric(ndim=2)
        p = {"theta": u.Angle(0.001, "rad"), "phi": u.Angle(0, "rad")}
        v = {"theta": u.Q(0, "rad/s"), "phi": u.Q(1, "rad/s")}
        result = cxm.norm(v, metric, cxc.sph2, at=p)
        assert u.ustrip("rad/s", result) < 0.01

    def test_theta_only_norm_is_one(self):
        """Pure theta motion has norm 1."""
        metric = cxm.HyperSphericalMetric(ndim=2)
        for theta_val in [0.1, jnp.pi / 4, jnp.pi / 2]:
            p = {"theta": u.Angle(theta_val, "rad"), "phi": u.Angle(0, "rad")}
            v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(0, "rad/s")}
            result = cxm.norm(v, metric, cxc.sph2, at=p)
            assert qnp.allclose(result, u.Q(1, "rad/s"), atol=u.Q(1e-5, "rad/s"))


# =============================================================================
# cxm.norm() with a packed jnp.Array
# =============================================================================


class TestNormBareArray:
    """Tests for cxm.norm() with a packed jnp.Array input."""

    @pytest.mark.parametrize(
        ("v", "chart", "exp"),
        [
            (jnp.array([3, 4, 0]), cxc.cart3d, 5),
            (jnp.array([1, 0, 0]), cxc.cart3d, 1),
            (jnp.array([3, 4]), cxc.cart2d, 5),
            (jnp.array([5]), cxc.cart1d, 5),
        ],
    )
    def test_euclidean_fast_path(self, v, chart, exp):
        """Euclidean Cartesian fast-path accepts a bare Array without at/usys."""
        result = cxm.norm(v, chart)
        assert jnp.allclose(result, exp)

    def test_euclidean_fast_path_returns_bare_array(self):
        """Fast-path returns a bare jax.Array, not a Quantity."""
        result = cxm.norm(jnp.array([1, 0, 0]), cxc.cart3d)
        assert not isinstance(result, u.AbstractQuantity)

    def test_spherical_chart_requires_usys(self):
        """Non-Cartesian chart with bare Array raises TypeError when usys missing."""
        at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0)}
        with pytest.raises(TypeError, match="usys"):
            cxm.norm(jnp.array([1, 0]), cxc.sph2, at=at)

    def test_sph2_equator_theta_unit_norm(self):
        """On S², v=(1,0) at the equator has norm 1 (metric is identity there)."""
        at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0)}
        result = cxm.norm(jnp.array([1, 0]), cxc.sph2, at=at, usys=u.unitsystems.si)
        assert jnp.allclose(result, 1, atol=1e-6)

    def test_sph2_near_pole_phi_component_suppressed(self):
        """On S², pure phi velocity near the north pole has a tiny norm."""
        at = {"theta": jnp.array(0.001), "phi": jnp.array(0)}
        result = cxm.norm(jnp.array([0, 1]), cxc.sph2, at=at, usys=u.unitsystems.si)
        assert float(result) < 0.01

    def test_euclidean_fast_path_jit(self):
        @jax.jit
        def compute(v):
            return cxm.norm(v, cxc.cart3d)

        result = compute(jnp.array([3, 4, 0]))
        assert jnp.allclose(result, 5)

    def test_euclidean_fast_path_vmap(self):
        """Vectorised norm over a batch of Euclidean vectors."""
        v_batch = jnp.array([[3, 4, 0], [1, 0, 0], [0, 5, 0]])
        result = jax.vmap(lambda v: cxm.norm(v, cxc.cart3d))(v_batch)
        assert jnp.allclose(result, jnp.array([5, 1, 5]))


# =============================================================================
# cxm.norm() with CDict of bare jnp.Arrays
# =============================================================================


class TestNormBareCDict:
    """Tests for cxm.norm() with a CDict of bare jnp.Arrays."""

    def test_euclidean_cart3d_fast_path_no_usys(self):
        """Euclidean Cartesian fast-path: bare CDict accepted without usys."""
        v = {"x": jnp.array(3), "y": jnp.array(4), "z": jnp.array(0)}
        result = cxm.norm(v, cxc.cart3d)
        assert jnp.allclose(result, 5)

    def test_euclidean_cart3d_returns_bare_array(self):
        v = {"x": jnp.array(1), "y": jnp.array(0), "z": jnp.array(0)}
        result = cxm.norm(v, cxc.cart3d)
        assert not isinstance(result, u.AbstractQuantity)

    def test_general_chart_requires_usys(self):
        """CDict of raw arrays on a non-Cartesian chart raises TypeError w/out usys."""
        v = {"theta": jnp.array(1), "phi": jnp.array(0)}
        at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0)}
        with pytest.raises(TypeError, match="usys"):
            cxm.norm(v, cxc.sph2, at=at)

    def test_euclidean_with_explicit_usys(self):
        """Usys is accepted and result still bare-array for Euclidean."""
        v = {"x": jnp.array(3), "y": jnp.array(4), "z": jnp.array(0)}
        at = {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}
        result = cxm.norm(v, cxc.cart3d, at=at, usys=u.unitsystems.si)
        assert jnp.allclose(result, 5)
        assert not isinstance(result, u.AbstractQuantity)


# =============================================================================
# cxm.norm() with a single AbstractQuantity (1-D overload)
# =============================================================================


class TestNormSingleQuantity:
    """Tests for cxm.norm() with a single AbstractQuantity (1-D chart overload)."""

    def test_positive_value(self):
        at = {"x": jnp.array(0)}
        result = cxm.norm(u.Q(5, "m/s"), cxc.cart1d, at=at)
        assert qnp.allclose(result, u.Q(5, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_negative_value_gives_positive_norm(self):
        at = {"x": jnp.array(0)}
        result = cxm.norm(u.Q(-3, "m"), cxc.cart1d, at=at)
        assert qnp.allclose(result, u.Q(3, "m"), atol=u.Q(1e-5, "m"))

    def test_returns_quantity(self):
        at = {"x": jnp.array(0)}
        result = cxm.norm(u.Q(5, "m"), cxc.cart1d, at=at)
        assert isinstance(result, u.AbstractQuantity)

    def test_jit(self):
        at = {"x": jnp.array(0)}

        @jax.jit
        def compute(v):
            return cxm.norm(v, cxc.cart1d, at=at)

        result = compute(u.Q(5, "m"))
        assert qnp.allclose(result, u.Q(5, "m"), atol=u.Q(1e-5, "m"))


# =============================================================================
# AbstractMetric.norm() convenience wrapper
# =============================================================================


class TestNormMetricWrapper:
    """Tests for metric.norm() convenience method."""

    def test_euclidean_cdict_quantities(self):
        metric = cxm.EuclideanMetric(3)
        at = {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}
        v = {"x": u.Q(3, "m/s"), "y": u.Q(4, "m/s"), "z": u.Q(0, "m/s")}
        result = metric.norm(v, cxc.cart3d, at=at)
        assert qnp.allclose(result, u.Q(5, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_euclidean_bare_array_with_usys(self):
        metric = cxm.EuclideanMetric(3)
        at = {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}
        result = metric.norm(
            jnp.array([3, 4, 0]), cxc.cart3d, at=at, usys=u.unitsystems.si
        )
        assert jnp.allclose(result, 5)

    def test_euclidean_bare_cdict_with_usys(self):
        metric = cxm.EuclideanMetric(3)
        at = {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}
        v = {"x": jnp.array(3), "y": jnp.array(4), "z": jnp.array(0)}
        result = metric.norm(v, cxc.cart3d, at=at, usys=u.unitsystems.si)
        assert jnp.allclose(result, 5)

    def test_agrees_with_functional_api(self):
        """metric.norm(v, chart, at=at) == cxm.norm(v, metric, chart, at=at)."""
        metric = cxm.EuclideanMetric(3)
        at = {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}
        v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
        result_wrapper = metric.norm(v, cxc.cart3d, at=at)
        result_functional = cxm.norm(v, metric, cxc.cart3d, at=at)
        assert qnp.allclose(result_wrapper, result_functional, atol=u.Q(1e-5, "m"))

    def test_spherical_metric_wrapper_matches_direct(self):
        """HyperSphericalMetric.norm matches cxm.norm at S² equator."""
        metric = cxm.HyperSphericalMetric(ndim=2)
        at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")}
        v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(0, "rad/s")}
        result_wrapper = metric.norm(v, cxc.sph2, at=at)
        result_direct = cxm.norm(v, metric, cxc.sph2, at=at)
        assert qnp.allclose(result_wrapper, result_direct, atol=u.Q(1e-5, "rad/s"))


# =============================================================================
# AbstractManifold.norm() convenience wrapper
# =============================================================================


class TestNormManifoldWrapper:
    """Tests for M.norm() convenience method."""

    def test_euclidean_manifold_cdict_quantities(self):
        M = cxm.EuclideanManifold(3)
        chart = cxc.Cart3D(M=M)
        at = {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}
        v = {"x": u.Q(3, "m/s"), "y": u.Q(4, "m/s"), "z": u.Q(0, "m/s")}
        result = M.norm(v, chart, at=at)
        assert qnp.allclose(result, u.Q(5, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_euclidean_manifold_bare_array_with_usys(self):
        M = cxm.EuclideanManifold(3)
        chart = cxc.Cart3D(M=M)
        at = {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}
        result = M.norm(jnp.array([3, 4, 0]), chart, at=at, usys=u.unitsystems.si)
        assert jnp.allclose(result, 5)

    def test_agrees_with_functional_api(self):
        """M.norm(v, chart, at=at) == cxm.norm(v, M.metric, chart, at=at)."""
        M = cxm.EuclideanManifold(3)
        chart = cxc.Cart3D(M=M)
        at = {"x": jnp.array(0), "y": jnp.array(0), "z": jnp.array(0)}
        v = {"x": u.Q(1, "m"), "y": u.Q(1, "m"), "z": u.Q(1, "m")}
        result_wrapper = M.norm(v, chart, at=at)
        result_functional = cxm.norm(v, M.metric, chart, at=at)
        assert qnp.allclose(result_wrapper, result_functional, atol=u.Q(1e-5, "m"))

    def test_hyperspherical_manifold_predefined_chart(self):
        """cxc.sph2.M.norm uses the manifold attached to the predefined chart."""
        M = cxc.sph2.M  # HyperSphericalManifold(2)
        at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")}
        v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(0, "rad/s")}
        result = M.norm(v, cxc.sph2, at=at)
        assert qnp.allclose(result, u.Q(1, "rad/s"), atol=u.Q(1e-5, "rad/s"))


# =============================================================================
# cxm.norm() on the full spherical 3D chart (mixed units)
# =============================================================================


class TestNormSph3D:
    """Tests for cxm.norm() on sph3d (r, theta, phi) with mixed-unit vectors."""

    def setup_method(self):
        self.at = {
            "r": u.Q(5, "m"),
            "theta": u.Q(jnp.pi / 2, "rad"),
            "phi": u.Q(0, "rad"),
        }

    def test_purely_radial_norm_equals_vr(self):
        """Pure radial velocity: norm = |vr|, independent of position."""
        v = {"r": u.Q(1, "m/s"), "theta": u.Q(0, "rad/s"), "phi": u.Q(0, "rad/s")}
        result = cxm.norm(v, cxc.sph3d, at=self.at)
        assert qnp.allclose(result, u.Q(1, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_purely_theta_at_equator(self):
        """Pure theta velocity at equator: norm = r * |vθ| = 5 m * 1 rad/s = 5 m/s."""
        v = {"r": u.Q(0, "m/s"), "theta": u.Q(1, "rad/s"), "phi": u.Q(0, "rad/s")}
        result = cxm.norm(v, cxc.sph3d, at=self.at)
        # g_θθ = r² = 25 m² → norm = sqrt(25 m² * 1 rad²/s²) = 5 m/s
        assert qnp.allclose(result, u.Q(5, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_purely_phi_at_equator(self):
        """Pure phi velocity at equator (sin θ = 1): norm = r * |vφ| = 5 m/s."""
        v = {"r": u.Q(0, "m/s"), "theta": u.Q(0, "rad/s"), "phi": u.Q(1, "rad/s")}
        result = cxm.norm(v, cxc.sph3d, at=self.at)
        # g_φφ = r²sin²θ = 25 m² → norm = 5 m/s
        assert qnp.allclose(result, u.Q(5, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_zero_vector(self):
        v = {"r": u.Q(0, "m/s"), "theta": u.Q(0, "rad/s"), "phi": u.Q(0, "rad/s")}
        result = cxm.norm(v, cxc.sph3d, at=self.at)
        assert qnp.allclose(result, u.Q(0, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_returns_quantity(self):
        v = {"r": u.Q(1, "m/s"), "theta": u.Q(0, "rad/s"), "phi": u.Q(0, "rad/s")}
        result = cxm.norm(v, cxc.sph3d, at=self.at)
        assert isinstance(result, u.AbstractQuantity)

    def test_jit(self):
        @jax.jit
        def compute(v, at):
            return cxm.norm(v, cxc.sph3d, at=at)

        v = {"r": u.Q(1, "m/s"), "theta": u.Q(0, "rad/s"), "phi": u.Q(0, "rad/s")}
        result = compute(v, self.at)
        assert qnp.allclose(result, u.Q(1, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_vmap_over_base_points(self):
        """Vmap over r: purely radial norm equals |vr| regardless of r."""
        at_batch = {
            "r": u.Q(jnp.array([1, 5, 10]), "m"),
            "theta": u.Q(jnp.full(3, jnp.pi / 2), "rad"),
            "phi": u.Q(jnp.zeros(3), "rad"),
        }
        v = {
            "r": u.Q(jnp.ones(3), "m/s"),
            "theta": u.Q(jnp.zeros(3), "rad/s"),
            "phi": u.Q(jnp.zeros(3), "rad/s"),
        }
        result = jax.vmap(lambda v, at: cxm.norm(v, cxc.sph3d, at=at))(v, at_batch)
        assert qnp.allclose(result, u.Q(jnp.ones(3), "m/s"), atol=u.Q(1e-5, "m/s"))


# =============================================================================
# Low-level (metric_matrix, v) dispatch
# =============================================================================


class TestNormLowLevel:
    """Tests for cxm.norm() with a precomputed metric matrix."""

    def test_identity_matrix_bare_array(self):
        G = jnp.eye(3)
        v = jnp.array([3, 4, 0])
        result = cxm.norm(G, v)
        assert jnp.allclose(result, 5)

    def test_identity_matrix_quantity(self):
        G = jnp.eye(3)
        v = u.Q(jnp.array([3, 4, 0]), "m/s")
        result = cxm.norm(G, v)
        assert qnp.allclose(result, u.Q(5, "m/s"), atol=u.Q(1e-5, "m/s"))

    def test_identity_matrix_cdict_quantities(self):
        G = jnp.eye(3)
        v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
        result = cxm.norm(G, v)
        assert qnp.allclose(result, u.Q(5, "m"), atol=u.Q(1e-5, "m"))

    def test_identity_matrix_cdict_bare(self):
        G = jnp.eye(3)
        v = {"x": jnp.array(3), "y": jnp.array(4), "z": jnp.array(0)}
        result = cxm.norm(G, v)
        assert jnp.allclose(result, 5)

    def test_scaled_metric(self):
        """G = 4·I → norm = 2 · euclidean_norm."""
        G = 4 * jnp.eye(3)
        v = jnp.array([1, 0, 0])
        result = cxm.norm(G, v)
        assert jnp.allclose(result, 2)

    def test_returns_bare_array_for_bare_v(self):
        G = jnp.eye(2)
        v = jnp.array([1, 0])
        result = cxm.norm(G, v)
        assert not isinstance(result, u.AbstractQuantity)

    def test_returns_quantity_for_quantity_v(self):
        G = jnp.eye(2)
        v = u.Q(jnp.array([1, 0]), "m")
        result = cxm.norm(G, v)
        assert isinstance(result, u.AbstractQuantity)


# =============================================================================
# Error conditions
# =============================================================================


class TestNormErrors:
    """Tests that norm() raises on invalid inputs."""

    def test_bare_array_on_curved_chart_no_usys(self):
        """Bare Array on a non-Cartesian chart requires usys."""
        at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0)}
        with pytest.raises(TypeError, match="usys"):
            cxm.norm(jnp.array([1, 0]), cxc.sph2, at=at)

    def test_bare_cdict_on_curved_chart_no_usys(self):
        """CDict of bare arrays on a non-Cartesian chart requires usys."""
        v = {"theta": jnp.array(1), "phi": jnp.array(0)}
        at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0)}
        with pytest.raises(TypeError, match="usys"):
            cxm.norm(v, cxc.sph2, at=at)

    def test_metric_mismatch_raises_value_error(self):
        """Passing a metric that doesn't belong to the chart raises ValueError."""
        wrong_metric = cxm.EuclideanMetric(3)  # sph2 uses HyperSphericalMetric(ndim=2)
        at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")}
        v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(0, "rad/s")}
        with pytest.raises(ValueError, match=r"[Mm]etric"):
            cxm.norm(v, wrong_metric, cxc.sph2, at=at)


# =============================================================================
# JAX transformations on curved (spherical) manifolds
# =============================================================================


class TestNormJAXCompatSphere:
    """JIT and vmap for norm() on the HyperSphericalMetric."""

    def test_jit_sph2_cdict_quantities(self):
        metric = cxm.HyperSphericalMetric(ndim=2)
        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")}

        @jax.jit
        def compute(v, at):
            return cxm.norm(v, metric, cxc.sph2, at=at)

        v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(0, "rad/s")}
        result = compute(v, p)
        assert qnp.allclose(result, u.Q(1, "rad/s"), atol=u.Q(1e-5, "rad/s"))

    def test_vmap_over_base_point_phi_norm_equals_sin_theta(self):
        """Norm of unit phi-velocity = sin(theta) on S²."""
        metric = cxm.HyperSphericalMetric(ndim=2)
        v = {"theta": u.Q(0, "rad/s"), "phi": u.Q(1, "rad/s")}
        thetas = jnp.array([jnp.pi / 6, jnp.pi / 4, jnp.pi / 2])
        expected = jnp.sin(thetas)  # g_φφ = sin²θ → norm = |sinθ|

        def at_theta(theta):
            return {"theta": u.Angle(theta, "rad"), "phi": u.Angle(0, "rad")}

        results = jax.vmap(lambda t: cxm.norm(v, metric, cxc.sph2, at=at_theta(t)))(
            thetas
        )
        assert qnp.allclose(results, u.Q(expected, "rad/s"), atol=u.Q(1e-5, "rad/s"))

    def test_vmap_over_vectors_at_fixed_point(self):
        """Norm scales linearly with vector magnitude."""
        metric = cxm.HyperSphericalMetric(ndim=2)
        at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")}
        magnitudes = jnp.array([1, 2, 3])
        v_batch = {
            "theta": u.Q(magnitudes, "rad/s"),
            "phi": u.Q(jnp.zeros(3), "rad/s"),
        }
        results = jax.vmap(lambda v: cxm.norm(v, metric, cxc.sph2, at=at))(v_batch)
        assert qnp.allclose(results, u.Q(magnitudes, "rad/s"), atol=u.Q(1e-5, "rad/s"))

    def test_position_dependence_for_curved_metric(self):
        """Norm of phi-velocity differs at different latitudes on S²."""
        metric = cxm.HyperSphericalMetric(ndim=2)
        v = {"theta": u.Q(0, "rad/s"), "phi": u.Q(1, "rad/s")}
        at_equator = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")}
        at_30deg = {"theta": u.Angle(jnp.pi / 6, "rad"), "phi": u.Angle(0, "rad")}

        norm_eq = cxm.norm(v, metric, cxc.sph2, at=at_equator)
        norm_30 = cxm.norm(v, metric, cxc.sph2, at=at_30deg)

        # sin(π/2) = 1 > sin(π/6) = 0.5
        assert u.ustrip("rad/s", norm_eq) > u.ustrip("rad/s", norm_30)
