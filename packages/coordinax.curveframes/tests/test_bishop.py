"""Tests for BishopTransform."""

import jax
import jax.numpy as jnp
import pytest

import quaxed.numpy as qnp
import unxt as u

import coordinax.curveframes as cxfc

# ── Fixtures ──────────────────────────────────────────────────────────


def _circle_curve(tau: u.Quantity) -> u.Quantity:
    """Unit circle in the x-y plane, period = 2*pi seconds."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def _straight_line(tau: u.Quantity) -> u.Quantity:
    """Straight line along x-axis (kappa=0 everywhere).

    Frenet-Serret frame is singular on this curve, but Bishop is not.
    """
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")


def _helix_curve(tau: u.Quantity) -> u.Quantity:
    """Helix with pitch along z-axis."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


def _circle_curve_yr(tau: u.Quantity) -> u.Quantity:
    """Circle in x-y plane with tau in years."""
    omega = u.Q(2 * jnp.pi, "rad/yr")
    phase = (omega * tau).uconvert("rad").ustrip("rad")
    x = u.Q(5.0, "km") * jnp.cos(phase)
    y = u.Q(5.0, "km") * jnp.sin(phase)
    z = u.Q(0.0, "km") * jnp.ones_like(phase)
    return qnp.stack([x, y, z], axis=-1)


@pytest.fixture
def circle_bishop() -> cxfc.BishopTransform:
    return cxfc.BishopTransform.from_curve(_circle_curve)


@pytest.fixture
def line_bishop() -> cxfc.BishopTransform:
    return cxfc.BishopTransform.from_curve(_straight_line)


@pytest.fixture
def helix_bishop() -> cxfc.BishopTransform:
    return cxfc.BishopTransform.from_curve(_helix_curve)


@pytest.fixture
def circle_yr_bishop() -> cxfc.BishopTransform:
    return cxfc.BishopTransform.from_curve(_circle_curve_yr, tau_unit="yr")


# ── Location ──────────────────────────────────────────────────────────


class TestBishopTransformLocation:
    """The location field should be the curve itself."""

    def test_location_at_zero(self, circle_bishop: cxfc.BishopTransform):
        loc = circle_bishop.location(u.Q(0.0, "s"))
        expected = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
        assert jnp.allclose(loc.value, expected.value, atol=1e-6)

    def test_location_is_curve(self, circle_bishop: cxfc.BishopTransform):
        assert circle_bishop.location is circle_bishop.curve


# ── Tangent ───────────────────────────────────────────────────────────


class TestBishopTransformTangent:
    """T should be the unit tangent vector (same as Frenet-Serret)."""

    def test_tangent_at_zero(self, circle_bishop: cxfc.BishopTransform):
        """At tau=0 on a unit circle, T = (0, 1, 0)."""
        T = circle_bishop.tangent(u.Q(0.0, "s"))
        expected = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(T.value, expected, atol=1e-5)

    def test_tangent_at_pi_over_2(self, circle_bishop: cxfc.BishopTransform):
        """At tau=pi/2, T = (-1, 0, 0)."""
        T = circle_bishop.tangent(u.Q(jnp.pi / 2, "s"))
        expected = jnp.array([-1.0, 0.0, 0.0])
        assert jnp.allclose(T.value, expected, atol=1e-5)

    def test_tangent_is_unit_vector(self, circle_bishop: cxfc.BishopTransform):
        T = circle_bishop.tangent(u.Q(1.23, "s"))
        norm = jnp.sqrt(jnp.sum(T.value**2))
        assert jnp.allclose(norm, 1.0, atol=1e-5)

    def test_tangent_straight_line(self, line_bishop: cxfc.BishopTransform):
        """Tangent of a straight line along x is always (1,0,0)."""
        T = line_bishop.tangent(u.Q(5.0, "s"))
        expected = jnp.array([1.0, 0.0, 0.0])
        assert jnp.allclose(T.value, expected, atol=1e-5)


# ── Normal1 / Normal2 ────────────────────────────────────────────────


class TestBishopTransformNormals:
    """U1 and U2 should be parallel-transported normal vectors."""

    def test_normal1_is_unit_vector(self, circle_bishop: cxfc.BishopTransform):
        U1 = circle_bishop.normal1(u.Q(0.7, "s"))
        norm = jnp.sqrt(jnp.sum(U1.value**2))
        assert jnp.allclose(norm, 1.0, atol=1e-5)

    def test_normal2_is_unit_vector(self, circle_bishop: cxfc.BishopTransform):
        U2 = circle_bishop.normal2(u.Q(0.7, "s"))
        norm = jnp.sqrt(jnp.sum(U2.value**2))
        assert jnp.allclose(norm, 1.0, atol=1e-5)

    def test_normals_perpendicular_to_tangent(
        self, circle_bishop: cxfc.BishopTransform
    ):
        tau = u.Q(1.0, "s")
        T = circle_bishop.tangent(tau).value
        U1 = circle_bishop.normal1(tau).value
        U2 = circle_bishop.normal2(tau).value
        assert jnp.allclose(jnp.dot(T, U1), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.dot(T, U2), 0.0, atol=1e-4)

    def test_normals_perpendicular_to_each_other(
        self, circle_bishop: cxfc.BishopTransform
    ):
        tau = u.Q(1.0, "s")
        U1 = circle_bishop.normal1(tau).value
        U2 = circle_bishop.normal2(tau).value
        assert jnp.allclose(jnp.dot(U1, U2), 0.0, atol=1e-4)

    def test_straight_line_normals_exist(self, line_bishop: cxfc.BishopTransform):
        """Bishop normals are defined even on a straight line (kappa=0)."""
        U1 = line_bishop.normal1(u.Q(1.0, "s"))
        U2 = line_bishop.normal2(u.Q(1.0, "s"))
        norm1 = jnp.sqrt(jnp.sum(U1.value**2))
        norm2 = jnp.sqrt(jnp.sum(U2.value**2))
        assert jnp.allclose(norm1, 1.0, atol=1e-4)
        assert jnp.allclose(norm2, 1.0, atol=1e-4)

    def test_straight_line_normals_constant(self, line_bishop: cxfc.BishopTransform):
        """On a straight line, parallel transport keeps U1, U2 constant."""
        U1_0 = line_bishop.normal1(u.Q(0.0, "s")).value
        U1_5 = line_bishop.normal1(u.Q(5.0, "s")).value
        assert jnp.allclose(U1_0, U1_5, atol=1e-4)


# ── Orthonormality ────────────────────────────────────────────────────


class TestBishopTransformOrthonormality:
    """T, U1, U2 should form an orthonormal right-handed triad."""

    @pytest.mark.parametrize("tau_val", [0.0, 0.5, 1.0, 2.5, jnp.pi])
    def test_orthogonality(self, circle_bishop: cxfc.BishopTransform, tau_val: float):
        tau = u.Q(tau_val, "s")
        T = circle_bishop.tangent(tau).value
        U1 = circle_bishop.normal1(tau).value
        U2 = circle_bishop.normal2(tau).value
        assert jnp.allclose(jnp.dot(T, U1), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.dot(T, U2), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.dot(U1, U2), 0.0, atol=1e-4)

    @pytest.mark.parametrize("tau_val", [0.0, 1.0, jnp.pi])
    def test_right_handed(self, circle_bishop: cxfc.BishopTransform, tau_val: float):
        """U2 should equal T x U1 (right-handed frame)."""
        tau = u.Q(tau_val, "s")
        T = circle_bishop.tangent(tau).value
        U1 = circle_bishop.normal1(tau).value
        U2 = circle_bishop.normal2(tau).value
        assert jnp.allclose(jnp.cross(T, U1), U2, atol=1e-4)


# ── Opaque-unit curve ────────────────────────────────────────────────


class TestBishopTransformOpaqueUnits:
    """Test with a curve whose internal unit (yr) differs from caller's."""

    def test_tangent_at_zero(self, circle_yr_bishop: cxfc.BishopTransform):
        T = circle_yr_bishop.tangent(u.Q(0.0, "yr"))
        expected = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(T.value, expected, atol=1e-5)

    def test_normals_orthogonal_at_zero(self, circle_yr_bishop: cxfc.BishopTransform):
        tau = u.Q(0.0, "yr")
        T = circle_yr_bishop.tangent(tau).value
        U1 = circle_yr_bishop.normal1(tau).value
        U2 = circle_yr_bishop.normal2(tau).value
        assert jnp.allclose(jnp.dot(T, U1), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.dot(T, U2), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.dot(U1, U2), 0.0, atol=1e-4)


# ── tau_0 parameter ──────────────────────────────────────────────────


class TestBishopTransformTau0:
    """The tau_0 field sets the reference parameter."""

    def test_default_tau_0(self, circle_bishop: cxfc.BishopTransform):
        """Default tau_0 is Q(0.0, tau_unit)."""
        assert jnp.allclose(circle_bishop.tau_0.value, 0.0)

    def test_custom_tau_0(self):
        """Custom tau_0 shifts the origin of parallel transport."""
        bt = cxfc.BishopTransform.from_curve(_circle_curve, tau_0=u.Q(1.0, "s"))
        # The tangent is still computed correctly
        T = bt.tangent(u.Q(1.0, "s"))
        norm = jnp.sqrt(jnp.sum(T.value**2))
        assert jnp.allclose(norm, 1.0, atol=1e-5)

    def test_initial_normal_field(self):
        """initial_normal is stored for reconstruction."""
        bt = cxfc.BishopTransform.from_curve(_circle_curve)
        # initial_normal is None (auto) or a 3-vector
        if bt.initial_normal is not None:
            assert bt.initial_normal.shape == (3,)


# ── JAX transformations ──────────────────────────────────────────────


class TestBishopTransformJAX:
    """Verify compatibility with jit and vmap."""

    def test_jit_tangent(self, circle_bishop: cxfc.BishopTransform):
        T = jax.jit(circle_bishop.tangent)(u.Q(0.0, "s"))
        expected = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(T.value, expected, atol=1e-5)

    def test_jit_normal1(self, circle_bishop: cxfc.BishopTransform):
        """normal1 (ODE-based) works under jit."""
        U1 = jax.jit(circle_bishop.normal1)(u.Q(0.5, "s"))
        norm = jnp.sqrt(jnp.sum(U1.value**2))
        assert jnp.allclose(norm, 1.0, atol=1e-4)

    def test_vmap(self, circle_bishop: cxfc.BishopTransform):
        taus = u.Q(jnp.linspace(0, 2 * jnp.pi, 8), "s")
        Ts = jax.vmap(circle_bishop.tangent)(taus)
        norms = jnp.sqrt(jnp.sum(Ts.value**2, axis=-1))
        assert jnp.allclose(norms, 1.0, atol=1e-5)


# ── from_ constructor ────────────────────────────────────────────────


class TestBishopTransformFrom:
    """Test the ``from_()`` constructor and dispatch mechanism."""

    def test_from_dispatches_to_from_curve(self):
        bt = cxfc.BishopTransform.from_(_circle_curve)
        loc = bt.location(u.Q(0.0, "s"))
        expected = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
        assert jnp.allclose(loc.value, expected.value, atol=1e-6)


# ── Inverse ───────────────────────────────────────────────────────────


class TestBishopTransformInverse:
    """The inverse frame fields should satisfy R^T semantics."""

    def test_inverse_is_bishop_transform(self, circle_bishop: cxfc.BishopTransform):
        inv = circle_bishop.inverse
        assert isinstance(inv, cxfc.BishopTransform)

    @pytest.mark.parametrize("tau_val", [0.0, 0.5, 1.0, 2.5, jnp.pi])
    def test_inverse_orthonormality(
        self, circle_bishop: cxfc.BishopTransform, tau_val: float
    ):
        inv = circle_bishop.inverse
        tau = u.Q(tau_val, "s")
        T = inv.tangent(tau).value
        U1 = inv.normal1(tau).value
        U2 = inv.normal2(tau).value
        assert jnp.allclose(jnp.dot(T, U1), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.dot(T, U2), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.dot(U1, U2), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.linalg.norm(T), 1.0, atol=1e-4)
        assert jnp.allclose(jnp.linalg.norm(U1), 1.0, atol=1e-4)
        assert jnp.allclose(jnp.linalg.norm(U2), 1.0, atol=1e-4)

    @pytest.mark.parametrize("tau_val", [0.0, 1.0, jnp.pi])
    def test_roundtrip_forward_inverse(
        self, circle_bishop: cxfc.BishopTransform, tau_val: float
    ):
        """Apply forward then inverse: recovers original point."""
        tau = u.Q(tau_val, "s")
        p = u.Q(jnp.array([2.0, 3.0, 4.0]), "km")

        # Forward: p' = R @ (p - gamma)
        g = circle_bishop.location(tau)
        T = circle_bishop.tangent(tau)
        U1 = circle_bishop.normal1(tau)
        U2 = circle_bishop.normal2(tau)
        diff = p - g
        p_fwd = qnp.stack([qnp.sum(T * diff), qnp.sum(U1 * diff), qnp.sum(U2 * diff)])

        # Inverse: p_recovered = R_inv @ (p' - gamma_inv)
        inv = circle_bishop.inverse
        g_inv = inv.location(tau)
        Ti = inv.tangent(tau)
        U1i = inv.normal1(tau)
        U2i = inv.normal2(tau)
        diff_inv = p_fwd - g_inv
        p_rec = qnp.stack(
            [
                qnp.sum(Ti * diff_inv),
                qnp.sum(U1i * diff_inv),
                qnp.sum(U2i * diff_inv),
            ]
        )

        assert jnp.allclose(p_rec.value, p.value, atol=1e-3)

    @pytest.mark.parametrize("tau_val", [0.0, 1.0, jnp.pi])
    def test_double_inverse(self, circle_bishop: cxfc.BishopTransform, tau_val: float):
        """inverse.inverse should recover the original frame fields."""
        tau = u.Q(tau_val, "s")
        loc = circle_bishop.location(tau)
        loc2 = circle_bishop.inverse.inverse.location(tau)
        assert jnp.allclose(loc.value, loc2.value, atol=1e-4)

        T = circle_bishop.tangent(tau)
        T2 = circle_bishop.inverse.inverse.tangent(tau)
        assert jnp.allclose(T.value, T2.value, atol=1e-4)

    def test_inverse_jit(self, circle_bishop: cxfc.BishopTransform):
        inv = circle_bishop.inverse
        loc = jax.jit(inv.location)(u.Q(0.0, "s"))
        # Location is defined, check it runs
        assert loc.shape == (3,)

    def test_inverse_opaque_units(self, circle_yr_bishop: cxfc.BishopTransform):
        """Inverse should work with opaque-unit curves too."""
        inv = circle_yr_bishop.inverse
        tau = u.Q(0.0, "yr")
        loc = inv.location(tau)
        assert loc.shape == (3,)


# ── Helix (3D curve) ────────────────────────────────────────────────


class TestBishopTransformHelix:
    """Bishop frame on a helix — tests 3D behaviour."""

    def test_tangent_is_unit(self, helix_bishop: cxfc.BishopTransform):
        T = helix_bishop.tangent(u.Q(1.0, "s"))
        norm = jnp.sqrt(jnp.sum(T.value**2))
        assert jnp.allclose(norm, 1.0, atol=1e-5)

    def test_orthonormality(self, helix_bishop: cxfc.BishopTransform):
        tau = u.Q(1.0, "s")
        T = helix_bishop.tangent(tau).value
        U1 = helix_bishop.normal1(tau).value
        U2 = helix_bishop.normal2(tau).value
        assert jnp.allclose(jnp.dot(T, U1), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.dot(T, U2), 0.0, atol=1e-4)
        assert jnp.allclose(jnp.dot(U1, U2), 0.0, atol=1e-4)

    def test_roundtrip(self, helix_bishop: cxfc.BishopTransform):
        """Forward then inverse roundtrip on helix."""
        tau = u.Q(1.0, "s")
        p = u.Q(jnp.array([2.0, -1.0, 3.0]), "km")

        g = helix_bishop.location(tau)
        T = helix_bishop.tangent(tau)
        U1 = helix_bishop.normal1(tau)
        U2 = helix_bishop.normal2(tau)
        diff = p - g
        p_fwd = qnp.stack([qnp.sum(T * diff), qnp.sum(U1 * diff), qnp.sum(U2 * diff)])

        inv = helix_bishop.inverse
        g_inv = inv.location(tau)
        Ti = inv.tangent(tau)
        U1i = inv.normal1(tau)
        U2i = inv.normal2(tau)
        diff_inv = p_fwd - g_inv
        p_rec = qnp.stack(
            [
                qnp.sum(Ti * diff_inv),
                qnp.sum(U1i * diff_inv),
                qnp.sum(U2i * diff_inv),
            ]
        )

        assert jnp.allclose(p_rec.value, p.value, atol=1e-3)
