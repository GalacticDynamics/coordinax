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
from .conftest import circle, circle_yr, helix, inverse_rotation

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
