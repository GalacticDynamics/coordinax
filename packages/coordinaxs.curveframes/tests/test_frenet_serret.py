"""Closed-form Frenet-Serret values on the unit circle.

Structural guarantees shared with Bishop -- orthonormality, right-handedness,
inverse semantics, `act`, `frame_transition`, jit/vmap -- are asserted once in
`test_parallel_transport_contract.py`. What is left here is what is specific to
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
import quaxed.numpy as qnp
import unxt as u

import coordinaxs.curveframes as cxfc

# ── Fixtures ──────────────────────────────────────────────────────────


def _circle_curve(tau: u.Q) -> u.Q:
    """Unit circle in the x-y plane, period = 2*pi seconds."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def _circle_curve_yr(tau: u.Q) -> u.Q:
    """Circle in x-y plane with angular speed omega = 2*pi rad/yr.

    This curve internally converts tau to radians, so the "natural" tau-unit
    is opaque to the caller — a harder test case.
    """
    omega = u.Q(2 * jnp.pi, "rad/yr")
    phase = (omega * tau).uconvert("rad").ustrip("rad")
    x = u.Q(5, "km") * jnp.cos(phase)
    y = u.Q(5, "km") * jnp.sin(phase)
    z = u.Q(0, "km") * jnp.ones_like(phase)
    return qnp.stack([x, y, z], axis=-1)


def _inverse_rotation(builder: cxfc.AbstractCurveFrameBuilder, tau: u.Q):
    """Rotation matrix of the inverse family at ``tau`` (i.e. R^T)."""
    return cxfm.Parametric(builder).inverse.materialize(tau)[0].R


@pytest.fixture
def circle_fs() -> cxfc.FrenetSerretBuilder:
    return cxfc.FrenetSerretBuilder(_circle_curve)


@pytest.fixture
def circle_yr_fs() -> cxfc.FrenetSerretBuilder:
    return cxfc.FrenetSerretBuilder(_circle_curve_yr, "yr")


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
        ("row", "expected"),
        [(0, [0, -1, 0]), (1, [1, 0, 0]), (2, [0, 0, 1])],
    )
    def test_inverse_field_at_zero(
        self, circle_fs: cxfc.FrenetSerretBuilder, row: int, expected: list[float]
    ):
        Rinv = _inverse_rotation(circle_fs, u.Q(0.0, "s"))
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

        got = cxfm.act(cxfm.Parametric(circle_fs).inverse, tau, p)
        np.testing.assert_allclose(got.ustrip("km"), expected.ustrip("km"), atol=1e-5)


class TestFrenetSerretOpaqueUnits:
    """A curve whose internal unit (yr) differs from the caller's."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [("tangent", [0, 1, 0]), ("normal", [-1, 0, 0]), ("binormal", [0, 0, 1])],
    )
    def test_field_at_zero(
        self,
        circle_yr_fs: cxfc.FrenetSerretBuilder,
        field: str,
        expected: list[float],
    ):
        got = getattr(circle_yr_fs, field)(u.Q(0, "yr"))
        np.testing.assert_allclose(got.value, expected, atol=1e-5)

    def test_tau_unit_is_stored(self, circle_yr_fs: cxfc.FrenetSerretBuilder):
        assert circle_yr_fs.tau_unit == u.unit("yr")

    def test_inverse_maps_origin_to_curve(self, circle_yr_fs: cxfc.FrenetSerretBuilder):
        """For the yr-circle at tau=0 the curve is at (5, 0, 0) km."""
        inv = cxfm.Parametric(circle_yr_fs).inverse
        got = cxfm.act(inv, u.Q(0.0, "yr"), u.Q(jnp.array([0.0, 0.0, 0.0]), "km"))
        np.testing.assert_allclose(got.ustrip("km"), [5, 0, 0], atol=1e-3)


class TestFrenetSerretAct:
    """`act` values that depend on the Frenet-Serret triad specifically."""

    def test_act_forward_off_curve(self, circle_fs: cxfc.FrenetSerretBuilder):
        """At tau=0, p=(2,0,0) km => delta=(1,0,0) => R @ delta = (0,-1,0) km."""
        op = cxfm.Parametric(circle_fs)
        result = cxfm.act(op, u.Q(0, "s"), u.Q(jnp.array([2.0, 0.0, 0.0]), "km"))
        np.testing.assert_allclose(result.ustrip("km"), [0, -1, 0], atol=1e-6)
