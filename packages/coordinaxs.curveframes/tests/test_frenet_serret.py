"""Closed-form Frenet-Serret values on the unit circle.

Structural guarantees shared with Bishop -- orthonormality, right-handedness,
inverse semantics, `act`, `frame_transition`, jit/vmap -- are asserted once in
``test_parallel_transport_contract.py``. What is left here is what is specific to
Frenet-Serret: the actual (T, N, B) values, which Bishop does not share because
its normals are parallel-transported rather than curvature-derived.

For a unit circle at tau=0:
    gamma = (1, 0, 0), T = (0, 1, 0), N = (-1, 0, 0), B = (0, 0, 1)
    R = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    The inverse map is p' -> R^T p' + gamma; its rotation rows are the
    *columns* of the forward R:
    inv_T = col 0 of R = (0, -1, 0)
    inv_N = col 1 of R = (1, 0, 0)
    inv_B = col 2 of R = (0, 0, 1)
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.transforms as cxfm
import unxt as u

import coordinaxs.curveframes as cxfc
from .conftest import circle, circle_yr, helix, inverse_rotation, straight_line

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def circle_fs() -> cxfc.FrenetSerretBuilder:
    return cxfc.FrenetSerretBuilder(circle, "s")


@pytest.fixture
def circle_yr_fs() -> cxfc.FrenetSerretBuilder:
    return cxfc.FrenetSerretBuilder(circle_yr, "yr")


# ── Triad values ──────────────────────────────────────────────────────


class TestFrenetSerretTriadValues:
    """T, N, B take their curvature-derived values on the unit circle."""

    def test_tangent_is_dimensionless(self, circle_fs: cxfc.FrenetSerretBuilder):
        """The raw derivative carries km/s; after normalisation it is unitless."""
        assert circle_fs.tangent(u.Q(0, "s")).unit == u.unit("")

    @pytest.mark.parametrize(
        ("field", "tau_val", "expected"),
        [
            ("normal", 0, [-1, 0, 0]),  # points inward
            ("normal", jnp.pi / 2, [0, -1, 0]),
            ("binormal", 0, [0, 0, 1]),  # circle lies in the xy-plane
            ("binormal", 2, [0, 0, 1]),  # ... at every tau
        ],
    )
    def test_field_value(
        self,
        circle_fs: cxfc.FrenetSerretBuilder,
        field: str,
        tau_val: float,
        expected: list[float],
    ):
        got = getattr(circle_fs, field)(u.Q(tau_val, "s"))
        np.testing.assert_allclose(got.value, expected, atol=1e-5)


class TestFrenetSerretInverseValues:
    """The inverse frame fields are the columns of R (see module docstring)."""

    @pytest.mark.parametrize(
        ("row", "expected"), [(0, [0, -1, 0]), (1, [1, 0, 0]), (2, [0, 0, 1])]
    )
    def test_inverse_field_at_zero(
        self, circle_fs: cxfc.FrenetSerretBuilder, row: int, expected: list[float]
    ):
        Rinv = inverse_rotation(circle_fs, u.Q(0.0, "s"))
        np.testing.assert_allclose(Rinv[row], expected, atol=1e-5)

    @pytest.mark.parametrize("tau_val", [0.0, 1.0, jnp.pi])
    def test_inverse_is_RT_p_plus_gamma(
        self, circle_fs: cxfc.FrenetSerretBuilder, tau_val: float
    ):
        """The inverse acts as ``p' -> R^T p' + gamma``."""
        tau = u.Q(tau_val, "s")
        p = u.Q(jnp.array([2.0, 3.0, 4.0]), "km")

        R = circle_fs.rotation_matrix(tau)
        gamma = circle_fs.location(tau)
        expected = u.Q(R.T @ p.ustrip("km"), "km") + gamma

        got = cxfm.act(cxfm.TimeDep(circle_fs).inverse, tau, p)
        np.testing.assert_allclose(got.ustrip("km"), expected.ustrip("km"), atol=1e-5)


class TestFrenetSerretOpaqueUnits:
    """Frenet-Serret values on a curve whose internal unit (yr) is opaque.

    The unit-handling itself is contract-level (`TestOpaqueUnits`); what is
    left here is the curvature-derived N and B, which Bishop does not share.
    """

    @pytest.mark.parametrize(
        ("field", "expected"), [("normal", [-1, 0, 0]), ("binormal", [0, 0, 1])]
    )
    def test_field_at_zero(
        self, circle_yr_fs: cxfc.FrenetSerretBuilder, field: str, expected: list[float]
    ):
        got = getattr(circle_yr_fs, field)(u.Q(0, "yr"))
        np.testing.assert_allclose(got.value, expected, atol=1e-5)


class TestFrenetSerretAct:
    """`act` values that depend on the Frenet-Serret triad specifically."""

    def test_act_forward_off_curve(self, circle_fs: cxfc.FrenetSerretBuilder):
        """At tau=0, p=(2,0,0) km => delta=(1,0,0) => R @ delta = (0,-1,0) km."""
        op = cxfm.TimeDep(circle_fs)
        result = cxfm.act(op, u.Q(0, "s"), u.Q(jnp.array([2.0, 0.0, 0.0]), "km"))
        np.testing.assert_allclose(result.ustrip("km"), [0, -1, 0], atol=1e-6)


class TestTangentFastPathMatchesBase:
    """`FrenetSerretBuilder.tangent` is a fast path over the base accessor.

    The override skips the full rotation matrix (and so the second derivative
    that only N and B need) and normalises gamma' directly. Its docstring
    claims "the value is identical"; nothing else in the suite checks that, so
    a divergence -- a dropped normalisation, a `_param` mismatch -- would be
    silent. The base accessor (row 0 of R) is the oracle.
    """

    @pytest.mark.parametrize("tau", [0.0, 0.7, 2.3])
    def test_override_equals_row0_of_R(self, tau: float):
        builder = cxfc.FrenetSerretBuilder(helix, "s")
        t = u.Q(tau, "s")
        base = cxfc.AbstractCurveFrameBuilder.tangent(builder, t)
        np.testing.assert_allclose(base.value, builder.tangent(t).value, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(base.value), 1.0, atol=1e-6)


def cubic(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """``(t, t^3, 0)`` km: an inflection at ``t = 0``."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, t**3, jnp.zeros_like(t)]), "km")


def circle_r2(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Circle of radius 2 km: curvature 0.5 per km."""
    t = tau.ustrip("s")
    return u.Q(2.0 * jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


class TestCurvature:
    """Curvature is defined where the *frame* is not: it reads zero.

    These cover the straight line and the inflection, which `rotation_matrix`
    refuses -- that contrast is the point of the accessor.
    """

    def test_unit_circle(self) -> None:
        k = cxfc.FrenetSerretBuilder(circle, "s").curvature(u.Q(0.4, "s"))
        np.testing.assert_allclose(k.ustrip("1/km"), 1.0, atol=1e-8)

    def test_radius_scales_inversely(self) -> None:
        k = cxfc.FrenetSerretBuilder(circle_r2, "s").curvature(u.Q(0.4, "s"))
        np.testing.assert_allclose(k.ustrip("1/km"), 0.5, atol=1e-8)

    def test_helix(self) -> None:
        """For ``(a cos t, a sin t, b t)``: kappa = a / (a^2 + b^2)."""
        k = cxfc.FrenetSerretBuilder(helix, "s").curvature(u.Q(0.7, "s"))
        np.testing.assert_allclose(k.ustrip("1/km"), 1.0 / 1.09, atol=1e-8)

    def test_dimension_is_inverse_length(self) -> None:
        k = cxfc.FrenetSerretBuilder(circle, "s").curvature(u.Q(0.0, "s"))
        assert u.dimension_of(k) == u.dimension("1/length")

    def test_defined_on_a_straight_line(self) -> None:
        """Zero, and *not* refused -- the frame is, the curvature is not."""
        b = cxfc.FrenetSerretBuilder(straight_line, "s")
        np.testing.assert_allclose(
            b.curvature(u.Q(3.0, "s")).ustrip("1/km"), 0.0, atol=1e-12
        )
        with pytest.raises(Exception, match="curvature"):
            b.rotation_matrix(u.Q(3.0, "s"))

    def test_defined_at_an_inflection(self) -> None:
        """Same contrast on the cubic, where only one parameter degenerates."""
        b = cxfc.FrenetSerretBuilder(cubic, "s")
        np.testing.assert_allclose(
            b.curvature(u.Q(0.0, "s")).ustrip("1/km"), 0.0, atol=1e-12
        )
        with pytest.raises(Exception, match="curvature"):
            b.rotation_matrix(u.Q(0.0, "s"))

    @pytest.mark.parametrize("t", [-1.0, -0.3, 0.3, 1.0])
    def test_matches_signed_planar_magnitude(self, t: float) -> None:
        """`|kappa_s| == kappa` pins the two builders against each other."""
        tau = u.Q(t, "s")
        k = cxfc.FrenetSerretBuilder(cubic, "s").curvature(tau).ustrip("1/km")
        ks = cxfc.SignedPlanarBuilder(cubic, "s").signed_curvature(tau).ustrip("1/km")
        np.testing.assert_allclose(k, abs(ks), atol=1e-10)


class TestTorsion:
    """Torsion, unlike curvature, *is* undefined where the curvature vanishes.

    Its denominator is ``|gamma' x gamma''|^2``, which is
    ``(kappa |gamma'|^3)^2``.
    """

    def test_helix(self) -> None:
        """For ``(a cos t, a sin t, b t)``: torsion = b / (a^2 + b^2)."""
        tor = cxfc.FrenetSerretBuilder(helix, "s").torsion(u.Q(0.7, "s"))
        np.testing.assert_allclose(tor.ustrip("1/km"), 0.3 / 1.09, atol=1e-8)

    def test_zero_on_a_planar_curve(self) -> None:
        """A plane curve has no torsion, wherever it is defined."""
        tor = cxfc.FrenetSerretBuilder(circle, "s").torsion(u.Q(0.4, "s"))
        np.testing.assert_allclose(tor.ustrip("1/km"), 0.0, atol=1e-8)

    def test_dimension_is_inverse_length(self) -> None:
        tor = cxfc.FrenetSerretBuilder(helix, "s").torsion(u.Q(0.0, "s"))
        assert u.dimension_of(tor) == u.dimension("1/length")

    def test_refuses_on_a_straight_line(self) -> None:
        with pytest.raises(Exception, match="torsion"):
            cxfc.FrenetSerretBuilder(straight_line, "s").torsion(u.Q(3.0, "s"))

    def test_refuses_at_an_inflection(self) -> None:
        with pytest.raises(Exception, match="torsion"):
            cxfc.FrenetSerretBuilder(cubic, "s").torsion(u.Q(0.0, "s"))

    def test_curvature_is_defined_where_torsion_is_not(self) -> None:
        """The asymmetry, pinned: same parameter, one answers and one refuses."""
        b = cxfc.FrenetSerretBuilder(cubic, "s")
        np.testing.assert_allclose(
            b.curvature(u.Q(0.0, "s")).ustrip("1/km"), 0.0, atol=1e-12
        )
        with pytest.raises(Exception, match="torsion"):
            b.torsion(u.Q(0.0, "s"))
