"""Tests for FrenetSerretTransform."""

import jax
import jax.numpy as jnp
import pytest

import quaxed.numpy as qnp
import unxt as u

import coordinax.curveframes as cxfc

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


@pytest.fixture
def circle_fs() -> cxfc.FrenetSerretTransform:
    return cxfc.FrenetSerretTransform.from_curve(_circle_curve)


@pytest.fixture
def circle_yr_fs() -> cxfc.FrenetSerretTransform:
    return cxfc.FrenetSerretTransform.from_curve(_circle_curve_yr, tau_unit="yr")


# ── Location ──────────────────────────────────────────────────────────


class TestFrenetSerretTransformLocation:
    """The location field should be the curve itself."""

    def test_location_at_zero(self, circle_fs: cxfc.FrenetSerretTransform):
        loc = circle_fs.location(u.Q(0, "s"))
        expected = u.Q(jnp.array([1, 0, 0]), "km")
        assert jnp.allclose(loc.value, expected.value, atol=1e-6)


# ── Tangent ───────────────────────────────────────────────────────────


class TestFrenetSerretTransformTangent:
    """T should be the unit tangent vector."""

    def test_tangent_at_zero(self, circle_fs: cxfc.FrenetSerretTransform):
        """At tau=0 on a unit circle, T = (0, 1, 0)."""
        T = circle_fs.tangent(u.Q(0, "s"))
        expected = jnp.array([0, 1, 0])
        assert jnp.allclose(T.value, expected, atol=1e-5)

    def test_tangent_at_pi_over_2(self, circle_fs: cxfc.FrenetSerretTransform):
        """At tau=pi/2, T = (-1, 0, 0)."""
        T = circle_fs.tangent(u.Q(jnp.pi / 2, "s"))
        expected = jnp.array([-1, 0, 0])
        assert jnp.allclose(T.value, expected, atol=1e-5)

    def test_tangent_is_unit_vector(self, circle_fs: cxfc.FrenetSerretTransform):
        T = circle_fs.tangent(u.Q(1.23, "s"))
        norm = jnp.sqrt(jnp.sum(T.value**2))
        assert jnp.allclose(norm, 1, atol=1e-5)

    def test_tangent_has_speed_units(self, circle_fs: cxfc.FrenetSerretTransform):
        """The raw (un-normalised) derivative should carry km/s units."""
        T = circle_fs.tangent(u.Q(0, "s"))
        # After normalization: dimensionless
        assert T.unit == u.unit("")


# ── Normal ────────────────────────────────────────────────────────────


class TestFrenetSerretTransformNormal:
    """N should be the unit principal-normal vector."""

    def test_normal_at_zero(self, circle_fs: cxfc.FrenetSerretTransform):
        """At tau=0 on a unit circle, N = (-1, 0, 0) (points inward)."""
        N = circle_fs.normal(u.Q(0, "s"))
        expected = jnp.array([-1, 0, 0])
        assert jnp.allclose(N.value, expected, atol=1e-5)

    def test_normal_at_pi_over_2(self, circle_fs: cxfc.FrenetSerretTransform):
        """At tau=pi/2, N = (0, -1, 0)."""
        N = circle_fs.normal(u.Q(jnp.pi / 2, "s"))
        expected = jnp.array([0, -1, 0])
        assert jnp.allclose(N.value, expected, atol=1e-5)

    def test_normal_is_unit_vector(self, circle_fs: cxfc.FrenetSerretTransform):
        N = circle_fs.normal(u.Q(0.7, "s"))
        norm = jnp.sqrt(jnp.sum(N.value**2))
        assert jnp.allclose(norm, 1, atol=1e-5)


# ── Binormal ──────────────────────────────────────────────────────────


class TestFrenetSerretTransformBinormal:
    """B = T x N should be a unit vector perpendicular to both."""

    def test_binormal_at_zero(self, circle_fs: cxfc.FrenetSerretTransform):
        """For a circle in the x-y plane, B = (0, 0, 1) everywhere."""
        B = circle_fs.binormal(u.Q(0, "s"))
        expected = jnp.array([0, 0, 1])
        assert jnp.allclose(B.value, expected, atol=1e-5)

    def test_binormal_is_unit_vector(self, circle_fs: cxfc.FrenetSerretTransform):
        B = circle_fs.binormal(u.Q(2, "s"))
        norm = jnp.sqrt(jnp.sum(B.value**2))
        assert jnp.allclose(norm, 1, atol=1e-5)


# ── Orthonormality ────────────────────────────────────────────────────


class TestFrenetSerretTransformOrthonormality:
    """T, N, B should form an orthonormal right-handed triad."""

    @pytest.mark.parametrize("tau_val", [0, 0.5, 1, 2.5, jnp.pi])
    def test_orthogonality(self, circle_fs: cxfc.FrenetSerretTransform, tau_val: float):
        tau = u.Q(tau_val, "s")
        T = circle_fs.tangent(tau).value
        N = circle_fs.normal(tau).value
        B = circle_fs.binormal(tau).value
        assert jnp.allclose(jnp.dot(T, N), 0, atol=1e-5)
        assert jnp.allclose(jnp.dot(T, B), 0, atol=1e-5)
        assert jnp.allclose(jnp.dot(N, B), 0, atol=1e-5)

    @pytest.mark.parametrize("tau_val", [0, 1, jnp.pi])
    def test_right_handed(self, circle_fs: cxfc.FrenetSerretTransform, tau_val: float):
        """B should equal T x N (right-handed frame)."""
        tau = u.Q(tau_val, "s")
        T = circle_fs.tangent(tau).value
        N = circle_fs.normal(tau).value
        B = circle_fs.binormal(tau).value
        assert jnp.allclose(jnp.cross(T, N), B, atol=1e-5)


# ── Opaque-unit curve ────────────────────────────────────────────────


class TestFrenetSerretTransformOpaqueUnits:
    """Test with a curve whose internal unit (yr) differs from caller's."""

    def test_tangent_at_zero(self, circle_yr_fs: cxfc.FrenetSerretTransform):
        """Tangent at tau=0 should still be (0, 1, 0) direction."""
        T = circle_yr_fs.tangent(u.Q(0, "yr"))
        expected = jnp.array([0, 1, 0])
        assert jnp.allclose(T.value, expected, atol=1e-5)

    def test_normal_at_zero(self, circle_yr_fs: cxfc.FrenetSerretTransform):
        N = circle_yr_fs.normal(u.Q(0, "yr"))
        expected = jnp.array([-1, 0, 0])
        assert jnp.allclose(N.value, expected, atol=1e-5)

    def test_binormal_at_zero(self, circle_yr_fs: cxfc.FrenetSerretTransform):
        B = circle_yr_fs.binormal(u.Q(0, "yr"))
        expected = jnp.array([0, 0, 1])
        assert jnp.allclose(B.value, expected, atol=1e-5)


# ── JAX transformations ──────────────────────────────────────────────


class TestFrenetSerretTransformJAX:
    """Verify compatibility with jit and vmap."""

    def test_jit(self, circle_fs: cxfc.FrenetSerretTransform):
        T = jax.jit(circle_fs.tangent)(u.Q(0, "s"))
        expected = jnp.array([0, 1, 0])
        assert jnp.allclose(T.value, expected, atol=1e-5)

    def test_vmap(self, circle_fs: cxfc.FrenetSerretTransform):
        taus = u.Q(jnp.linspace(0, 2 * jnp.pi, 8), "s")
        Ts = jax.vmap(circle_fs.tangent)(taus)
        # All tangent vectors should be unit length
        norms = jnp.sqrt(jnp.sum(Ts.value**2, axis=-1))
        assert jnp.allclose(norms, 1, atol=1e-5)


# ── from_ constructor ────────────────────────────────────────────────


class TestFrenetSerretTransformFrom:
    """Test the ``from_()`` constructor and dispatch mechanism."""

    def test_from_dispatches_to_from_curve(self):
        fs = cxfc.FrenetSerretTransform.from_(_circle_curve)
        loc = fs.location(u.Q(0, "s"))
        expected = u.Q(jnp.array([1, 0, 0]), "km")
        assert jnp.allclose(loc.value, expected.value, atol=1e-6)


# ── Inverse ───────────────────────────────────────────────────────────

# For a unit circle at tau=0:
#   gamma = (1, 0, 0), T = (0, 1, 0), N = (-1, 0, 0), B = (0, 0, 1)
# R = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
# inv_location = -R @ gamma = -[T·g, N·g, B·g] = -[0, -1, 0] = (0, 1, 0)
# inv_T = col 0 of R = [T[0], N[0], B[0]] = [0, -1, 0]
# inv_N = col 1 of R = [T[1], N[1], B[1]] = [1, 0, 0]
# inv_B = col 2 of R = [T[2], N[2], B[2]] = [0, 0, 1]


class TestFrenetSerretTransformInverse:
    """The inverse frame fields should satisfy R^T semantics."""

    def test_inverse_location_at_zero(self, circle_fs: cxfc.FrenetSerretTransform):
        inv = circle_fs.inverse
        loc = inv.location(u.Q(0, "s"))
        expected = jnp.array([0, 1, 0])
        assert jnp.allclose(loc.value, expected, atol=1e-5)

    def test_inverse_tangent_at_zero(self, circle_fs: cxfc.FrenetSerretTransform):
        inv = circle_fs.inverse
        T = inv.tangent(u.Q(0, "s"))
        expected = jnp.array([0, -1, 0])
        assert jnp.allclose(T.value, expected, atol=1e-5)

    def test_inverse_normal_at_zero(self, circle_fs: cxfc.FrenetSerretTransform):
        inv = circle_fs.inverse
        N = inv.normal(u.Q(0, "s"))
        expected = jnp.array([1, 0, 0])
        assert jnp.allclose(N.value, expected, atol=1e-5)

    def test_inverse_binormal_at_zero(self, circle_fs: cxfc.FrenetSerretTransform):
        inv = circle_fs.inverse
        B = inv.binormal(u.Q(0, "s"))
        expected = jnp.array([0, 0, 1])
        assert jnp.allclose(B.value, expected, atol=1e-5)

    @pytest.mark.parametrize("tau_val", [0, 0.5, 1, 2.5, jnp.pi])
    def test_inverse_orthonormality(
        self, circle_fs: cxfc.FrenetSerretTransform, tau_val: float
    ):
        inv = circle_fs.inverse
        tau = u.Q(tau_val, "s")
        T = inv.tangent(tau).value
        N = inv.normal(tau).value
        B = inv.binormal(tau).value
        assert jnp.allclose(jnp.dot(T, N), 0, atol=1e-5)
        assert jnp.allclose(jnp.dot(T, B), 0, atol=1e-5)
        assert jnp.allclose(jnp.dot(N, B), 0, atol=1e-5)
        assert jnp.allclose(jnp.linalg.norm(T), 1, atol=1e-5)
        assert jnp.allclose(jnp.linalg.norm(N), 1, atol=1e-5)
        assert jnp.allclose(jnp.linalg.norm(B), 1, atol=1e-5)

    @pytest.mark.parametrize("tau_val", [0, 1, jnp.pi])
    def test_roundtrip_forward_inverse(
        self, circle_fs: cxfc.FrenetSerretTransform, tau_val: float
    ):
        """Apply forward then inverse: R_inv @ (R @ (p - g) - g_inv) == p."""
        tau = u.Q(tau_val, "s")
        p = u.Q(jnp.array([2, 3, 4]), "km")

        # Forward: p' = R @ (p - gamma)
        g = circle_fs.location(tau)
        T = circle_fs.tangent(tau)
        N = circle_fs.normal(tau)
        B = circle_fs.binormal(tau)
        diff = p - g
        p_fwd = qnp.stack([qnp.sum(T * diff), qnp.sum(N * diff), qnp.sum(B * diff)])

        # Inverse: p_recovered = R_inv @ (p' - gamma_inv)
        inv = circle_fs.inverse
        g_inv = inv.location(tau)
        Ti = inv.tangent(tau)
        Ni = inv.normal(tau)
        Bi = inv.binormal(tau)
        diff_inv = p_fwd - g_inv
        p_rec = qnp.stack(
            [qnp.sum(Ti * diff_inv), qnp.sum(Ni * diff_inv), qnp.sum(Bi * diff_inv)]
        )

        assert jnp.allclose(p_rec.value, p.value, atol=1e-4)

    @pytest.mark.parametrize("tau_val", [0, 1, jnp.pi])
    def test_double_inverse(
        self, circle_fs: cxfc.FrenetSerretTransform, tau_val: float
    ):
        """inverse.inverse should recover the original frame fields."""
        tau = u.Q(tau_val, "s")
        loc = circle_fs.location(tau)
        loc2 = circle_fs.inverse.inverse.location(tau)
        assert jnp.allclose(loc.value, loc2.value, atol=1e-4)

        T = circle_fs.tangent(tau)
        T2 = circle_fs.inverse.inverse.tangent(tau)
        assert jnp.allclose(T.value, T2.value, atol=1e-4)

    def test_inverse_jit(self, circle_fs: cxfc.FrenetSerretTransform):
        inv = circle_fs.inverse
        loc = jax.jit(inv.location)(u.Q(0, "s"))
        exp = jnp.array([0, 1, 0])
        assert jnp.allclose(loc.value, exp, atol=1e-5)

    def test_inverse_opaque_units(self, circle_yr_fs: cxfc.FrenetSerretTransform):
        """Inverse should work with opaque-unit curves too."""
        inv = circle_yr_fs.inverse
        tau = u.Q(0, "yr")
        # For the yr-circle at tau=0: gamma=(5,0,0)km, T=(0,1,0), N=(-1,0,0)
        # inv_location = -[T·g, N·g, B·g] = -[0, -5, 0] = (0, 5, 0)
        loc = inv.location(tau)
        expected = jnp.array([0, 5, 0])
        assert jnp.allclose(loc.value, expected, atol=1e-3)
