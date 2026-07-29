"""Pullback metric consistency: RoundMetric vs Jacobian pullback on S².

For the unit two-sphere S², the round metric in (θ, φ) coordinates gives
g = diag(1, sin²θ).  The same result must follow from the Jacobian pullback
of the flat metric on R³ via the standard Cartesian embedding:

    (θ, φ) → (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))

These tests assert that both paths agree numerically at sample points and
across a range of angles verified with Hypothesis.
"""

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinaxs.api.manifolds as cxmapi
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unit_sphere_embedded():
    """EmbeddedManifold for the unit two-sphere (dimensionless, radius=1)."""
    return cxm.EmbeddedManifold(
        intrinsic=cxm.S2, ambient=cxm.R3, embed_map=cxm.TwoSphereIn3D(radius=1)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_dense_matrix(theta, phi):
    """Expected round-metric as a dense 2×2 array at (theta, phi)."""
    return jnp.array([[1, 0], [0, jnp.sin(theta) ** 2]])


# ---------------------------------------------------------------------------
# Type contract tests
# ---------------------------------------------------------------------------


class TestPullbackConsistencyTypes:
    """Verify the metric_matrix return types for both paths."""

    def test_round_metric_returns_diagonal(self):
        pt = {"theta": jnp.array(jnp.pi / 3), "phi": jnp.array(jnp.pi / 4)}
        g = cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2)
        assert isinstance(g, DiagonalMetric)

    def test_embedded_metric_returns_dense(self, unit_sphere_embedded):
        pt = {"theta": jnp.array(jnp.pi / 3), "phi": jnp.array(jnp.pi / 4)}
        g = cxmapi.metric_matrix(unit_sphere_embedded, pt, cxc.sph2)
        assert isinstance(g, DenseMetric)


# ---------------------------------------------------------------------------
# Numerical consistency tests
# ---------------------------------------------------------------------------


class TestPullbackConsistencyNumerical:
    """Both paths give the same metric matrix at sample points."""

    @pytest.mark.parametrize(
        ("theta", "phi"),
        [
            (jnp.pi / 2, 0),  # equator, phi=0
            (jnp.pi / 3, jnp.pi / 4),  # off-equator
            (jnp.pi / 6, jnp.pi),  # high latitude, phi=π
            (0.1, 2.5),  # near pole, arbitrary phi
        ],
        ids=["equator-0", "off-equator", "high-lat-pi", "near-pole"],
    )
    def test_sample_point(self, unit_sphere_embedded, theta, phi):
        pt = {"theta": jnp.array(theta), "phi": jnp.array(phi)}

        g_round = cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2)
        g_pullback = cxmapi.metric_matrix(unit_sphere_embedded, pt, cxc.sph2)

        # RoundMetric (diagonal) and Jacobian pullback (dense) must agree.
        expected = g_round.to_dense().matrix  # plain array, shape (2, 2)
        actual = g_pullback.matrix.value  # QuantityMatrix.value, shape (2, 2)

        assert jnp.allclose(actual, expected, atol=1e-6), (
            f"Mismatch at theta={theta}, phi={phi}:\n"
            f"  expected={expected}\n  actual={actual}"
        )

    @given(
        theta=st.floats(
            min_value=0.05, max_value=3.09, allow_nan=False, allow_infinity=False
        ),
        phi=st.floats(
            min_value=0, max_value=6.28, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_hypothesis_s2(self, unit_sphere_embedded, theta, phi):
        pt = {"theta": jnp.array(theta), "phi": jnp.array(phi)}

        g_round = cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2)
        g_pullback = cxmapi.metric_matrix(unit_sphere_embedded, pt, cxc.sph2)

        expected = g_round.to_dense().matrix
        actual = g_pullback.matrix.value

        assert jnp.allclose(actual, expected, atol=1e-5), (
            f"Mismatch at theta={theta:.4f}, phi={phi:.4f}:\n"
            f"  expected={expected}\n  actual={actual}"
        )


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestPullbackConsistencyJIT:
    """Both metric paths are JIT-compatible."""

    def test_round_metric_jit(self):
        @jax.jit
        def compute(theta, phi):
            pt = {"theta": theta, "phi": phi}
            return cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2).diagonal

        result = compute(jnp.array(jnp.pi / 3), jnp.array(0))
        assert result.shape == (2,)

    def test_pullback_metric_jit(self):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2, ambient=cxm.R3, embed_map=cxm.TwoSphereIn3D(radius=1)
        )

        @jax.jit
        def compute(theta, phi):
            pt = {"theta": theta, "phi": phi}
            return cxmapi.metric_matrix(M, pt, cxc.sph2).matrix.value

        result = compute(jnp.array(jnp.pi / 3), jnp.array(0))
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# Unit preservation for non-trivial radius
# ---------------------------------------------------------------------------


class TestPullbackMetricUnits:
    """For a sphere with physical radius, the metric carries correct units."""

    def test_radius_1km_at_equator(self):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(1, "km")),
        )
        at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0, "rad")}
        g = cxmapi.metric_matrix(M, at, cxc.sph2)
        assert isinstance(g, DenseMetric)
        # At the equator sin(π/2)=1, so metric should be identity × km²/rad²
        assert jnp.allclose(g.matrix.value, jnp.eye(2), atol=1e-6)
        assert str(g.matrix.unit[0, 0]) == "km2 / rad2"

    def test_radius_2m_metric_scaled(self):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(2, "m")),
        )
        at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0, "rad")}
        g = cxmapi.metric_matrix(M, at, cxc.sph2)
        # Metric = R² × I at equator → values should be [[4, 0], [0, 4]]
        assert jnp.allclose(g.matrix.value, 4 * jnp.eye(2), atol=1e-6)
        assert str(g.matrix.unit[0, 0]) == "m2 / rad2"


# ---------------------------------------------------------------------------
# Non-canonical two-sphere charts: metric must match the R³ embedding J^T J
# ---------------------------------------------------------------------------


def _embedding_metric(chart, coords_rad):
    """Induced metric of ``chart`` from the unit-sphere R³ embedding, via J^T J."""
    keys = chart.components

    def embed(x):
        p = {k: u.Q(x[i], "rad") for i, k in enumerate(keys)}
        s = cxc.pt_map(p, chart, cxc.sph2)
        th, ph = u.ustrip("rad", s["theta"]), u.ustrip("rad", s["phi"])
        return jnp.array(
            [jnp.sin(th) * jnp.cos(ph), jnp.sin(th) * jnp.sin(ph), jnp.cos(th)]
        )

    x0 = jnp.array([coords_rad[k] for k in keys])
    jmat = jax.jacfwd(embed)(x0)  # (3, n)
    return jmat.T @ jmat


class TestNonCanonicalTwoSphereMetric:
    """metric_matrix for LonLat/Math/LonCosLat charts must be the true metric."""

    @pytest.mark.parametrize(
        ("chart", "coords"),
        [
            (cxc.math_sph2, {"theta": 0.6, "phi": 0.9}),
            (cxc.lonlat_sph2, {"lon": 0.6, "lat": 0.7}),
            (cxc.loncoslat_sph2, {"lon_coslat": 0.4, "lat": 0.7}),
        ],
    )
    def test_metric_matches_embedding(self, chart, coords):
        pt = {k: u.Q(v, "rad") for k, v in coords.items()}
        g = cxmapi.metric_matrix(cxm.S2, pt, chart)
        assert isinstance(g, DenseMetric)
        got = jnp.asarray(g.matrix.value)
        ref = _embedding_metric(chart, coords)
        assert jnp.allclose(got, ref, atol=1e-9), f"{got}\n!=\n{ref}"

    def test_integer_angles_do_not_break_jacfwd(self):
        """Integer-valued angles are promoted to float so the pullback jacfwd works."""
        pt = {"lon": u.Q(0, "rad"), "lat": u.Q(0, "rad")}  # integer magnitudes
        g = cxmapi.metric_matrix(cxm.S2, pt, cxc.lonlat_sph2)
        assert isinstance(g, DenseMetric)
        assert bool(jnp.all(jnp.isfinite(g.matrix.value)))

    @pytest.mark.parametrize(
        ("chart", "at", "expected"),
        [
            # LonLat (lon, lat) -> diag(cos^2 lat, 1)
            (cxc.lonlat_sph2, {"lon": 0.6, "lat": 0.7}, (math.cos(0.7) ** 2, 1.0)),
            # Math (theta=azimuth, phi=polar) -> diag(sin^2 phi, 1)
            (cxc.math_sph2, {"theta": 0.6, "phi": 0.9}, (math.sin(0.9) ** 2, 1.0)),
        ],
    )
    def test_scale_factors_defined_for_orthogonal_charts(self, chart, at, expected):
        """LonLat and Math are orthogonal, so scale factors exist."""
        h = cxm.scale_factors(chart, at={k: u.Q(v, "rad") for k, v in at.items()})
        assert jnp.allclose(jnp.asarray(h.value), jnp.asarray(expected), atol=1e-9)

    @pytest.mark.parametrize(
        ("chart", "keys"),
        [
            (cxc.lonlat_sph2, ("lon", "lat")),
            (cxc.math_sph2, ("theta", "phi")),
        ],
    )
    def test_scale_factors_match_metric_diagonal(self, chart, keys):
        """They must agree with the diagonal of the full metric."""
        at = {k: u.Q(v, "rad") for k, v in zip(keys, (0.4, 0.8), strict=True)}
        h = jnp.asarray(cxm.scale_factors(chart, at=at).value)
        g = jnp.asarray(cxmapi.metric_matrix(cxm.S2, at, chart).matrix.value)
        assert jnp.allclose(h, jnp.diagonal(g), atol=1e-12)

    def test_scale_factors_refused_for_non_orthogonal_chart(self):
        """LonCosLat is genuinely non-orthogonal: no per-axis factors exist."""
        at = {"lon_coslat": u.Q(0.4, "rad"), "lat": u.Q(0.7, "rad")}
        with pytest.raises(NotImplementedError, match="non-orthogonal"):
            cxm.scale_factors(cxc.loncoslat_sph2, at=at)


# ---------------------------------------------------------------------------
# Non-Riemannian ambient
# ---------------------------------------------------------------------------


def _worldline(ct_of, x_of):
    """1-D submanifold of Minkowski space: lambda -> (ct, x, 0, 0)."""
    embed_map = cxm.CustomEmbeddingMap(
        intrinsic=cxc.cart1d,
        ambient=cxc.minkowskict,
        embed_fn=lambda p, *, usys=None: {
            "ct": ct_of(p["x"]),
            "x": x_of(p["x"]),
            "y": p["x"] * 0,
            "z": p["x"] * 0,
        },
        project_fn=lambda p, *, usys=None: {"x": p["ct"]},
    )
    return cxm.EmbeddedManifold(
        intrinsic=cxm.R1, ambient=cxm.minkowski4d, embed_map=embed_map
    )


class TestLorentzianAmbient:
    """The pullback is J^T G J, not J^T J: G carries the ambient signature."""

    @pytest.mark.parametrize(
        ("name", "ct_of", "x_of", "expected"),
        [
            ("timelike", lambda s: s, lambda s: s * 0, -1.0),
            ("spacelike", lambda s: s * 0, lambda s: s, 1.0),
            ("null", lambda s: s, lambda s: s, 0.0),
            # A boost cannot change the character of a worldline.
            (
                "boosted-timelike",
                lambda s: s * jnp.cosh(0.7),
                lambda s: s * jnp.sinh(0.7),
                -1.0,
            ),
        ],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_induced_metric_carries_ambient_signature(
        self, name, ct_of, x_of, expected
    ):
        del name
        M = _worldline(ct_of, x_of)
        g = cxmapi.metric_matrix(M, {"x": u.Q(1.0, "m")}, cxc.cart1d)
        assert jnp.allclose(g.matrix.value, jnp.array([[expected]]), atol=1e-12)

    def test_signature_of_indefinite_pullback_is_refused(self):
        """It depends on the embedding, so there is no dimension-only answer."""
        M = _worldline(lambda s: s, lambda s: s * 0)
        assert M.metric.ndim == 1  # still answerable
        with pytest.raises(NotImplementedError, match="depends on the embedding"):
            _ = M.metric.signature

    def test_riemannian_ambient_signature_unaffected(self, unit_sphere_embedded):
        assert unit_sphere_embedded.metric.signature == (1, 1)
