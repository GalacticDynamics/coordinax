"""The signed planar frame: defined where Frenet--Serret is not.

Closed-form values and the degeneracies that motivate the type. The
structural guarantees shared with the other builders -- orthonormality,
right-handedness, `frame_transition` integration, JAX compatibility -- are
asserted once in `test_parallel_transport_contract.py`, parametrized.
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc
from .conftest import circle, helix, straight_line

Z = jnp.array([0.0, 0.0, 1.0])


def cubic(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """``(t, t^3, 0)`` km: an inflection at ``t = 0``, where Frenet dies."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, t**3, jnp.zeros_like(t)]), "km")


def wobble(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """A circle with a 1e-9 out-of-plane drift: planar to within tolerance."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 1e-9 * t]), "km")


class TestDefinedWhereFrenetIsNot:
    """The capability the type exists for."""

    @pytest.mark.parametrize("t", [-1e-3, -1e-9, 0.0, 1e-9, 1e-3])
    def test_normal_is_finite_through_the_inflection(self, t: float) -> None:
        """`N` is finite and continuous across the cubic's inflection."""
        N = cxfc.SignedPlanarBuilder(cubic, "s").normal(u.Q(t, "s"))
        assert jnp.all(jnp.isfinite(N.value))
        # Near the inflection the tangent is along +x, so N is along +y.
        np.testing.assert_allclose(N.value, [0.0, 1.0, 0.0], atol=1e-5)

    def test_frenet_refuses_at_the_same_point(self) -> None:
        """The contrast that motivates the type (#856 made this raise)."""
        with pytest.raises(Exception, match="curvature"):
            cxfc.FrenetSerretBuilder(cubic, "s").normal(u.Q(0.0, "s"))

    def test_defined_on_a_straight_line(self) -> None:
        """Frenet is undefined *everywhere* on a line; this is not."""
        b = cxfc.SignedPlanarBuilder(straight_line, "s")
        N = b.normal(u.Q(5.0, "s"))
        np.testing.assert_allclose(N.value, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(jnp.linalg.norm(N.value), 1.0, atol=1e-12)


class TestClosedForm:
    """Values on the unit circle, where this frame coincides with Frenet."""

    def test_triad_at_tau_zero(self) -> None:
        R = cxfc.SignedPlanarBuilder(circle, "s").rotation_matrix(u.Q(0.0, "s"))
        np.testing.assert_allclose(R[0], [0.0, 1.0, 0.0], atol=1e-10)  # T
        np.testing.assert_allclose(R[1], [-1.0, 0.0, 0.0], atol=1e-10)  # N
        np.testing.assert_allclose(R[2], [0.0, 0.0, 1.0], atol=1e-10)  # B

    def test_binormal_is_the_plane_normal(self) -> None:
        """On a planar curve row 2 *is* n-hat, for every parameter."""
        b = cxfc.SignedPlanarBuilder(circle, "s")
        for t in [0.0, 0.7, 2.5]:
            np.testing.assert_allclose(b.binormal(u.Q(t, "s")).value, Z, atol=1e-10)

    def test_agrees_with_frenet_on_the_circle(self) -> None:
        """CCW travel puts 'left' at the centre of curvature: same normal."""
        sp = cxfc.SignedPlanarBuilder(circle, "s")
        fs = cxfc.FrenetSerretBuilder(circle, "s")
        for t in [0.0, 0.7, 2.5]:
            tau = u.Q(t, "s")
            np.testing.assert_allclose(
                sp.normal(tau).value, fs.normal(tau).value, atol=1e-8
            )

    def test_opposes_frenet_where_the_curve_turns_right(self) -> None:
        """On the cubic's t<0 branch the two normals are antiparallel."""
        tau = u.Q(-1.0, "s")
        sp = cxfc.SignedPlanarBuilder(cubic, "s").normal(tau).value
        fs = cxfc.FrenetSerretBuilder(cubic, "s").normal(tau).value
        np.testing.assert_allclose(jnp.dot(sp, fs), -1.0, atol=1e-8)


class TestGauge:
    """`plane_normal` is the gauge, and it is an input."""

    def test_flipping_the_plane_normal_flips_the_normal(self) -> None:
        up = cxfc.SignedPlanarBuilder(circle, "s").normal(u.Q(0.0, "s"))
        down = cxfc.SignedPlanarBuilder(circle, "s", plane_normal=-Z).normal(
            u.Q(0.0, "s")
        )
        np.testing.assert_allclose(down.value, -up.value, atol=1e-10)

    def test_plane_normal_need_not_be_normalised(self) -> None:
        a = cxfc.SignedPlanarBuilder(circle, "s").normal(u.Q(0.3, "s"))
        b = cxfc.SignedPlanarBuilder(
            circle, "s", plane_normal=jnp.array([0.0, 0.0, 7.5])
        ).normal(u.Q(0.3, "s"))
        np.testing.assert_allclose(a.value, b.value, atol=1e-10)

    def test_zero_plane_normal_raises(self) -> None:
        """A zero vector names no plane; it would normalise to NaN."""
        b = cxfc.SignedPlanarBuilder(circle, "s", plane_normal=jnp.zeros(3))
        with pytest.raises(Exception, match="zero length"):
            b.normal(u.Q(0.0, "s"))


class TestPlanarityGuard:
    """Out-of-plane curves are refused, not silently projected."""

    def test_helix_raises(self) -> None:
        with pytest.raises(Exception, match="plane"):
            cxfc.SignedPlanarBuilder(helix, "s").normal(u.Q(0.0, "s"))

    def test_message_names_bishop(self) -> None:
        """Advice in an error message rots as quietly as a comment."""
        with pytest.raises(Exception, match="BishopBuilder"):
            cxfc.SignedPlanarBuilder(helix, "s").normal(u.Q(0.0, "s"))

    def test_wrong_plane_raises_on_a_planar_curve(self) -> None:
        """The xy-circle is planar, but not in the plane normal to x-hat."""
        b = cxfc.SignedPlanarBuilder(
            circle, "s", plane_normal=jnp.array([1.0, 0.0, 0.0])
        )
        with pytest.raises(Exception, match="plane"):
            b.normal(u.Q(0.0, "s"))

    def test_near_planar_curve_is_accepted(self) -> None:
        """Tolerance is sqrt(eps) ~ 1.5e-8 in f64, so a 1e-9 drift passes."""
        N = cxfc.SignedPlanarBuilder(wobble, "s").normal(u.Q(0.0, "s"))
        np.testing.assert_allclose(N.value, [-1.0, 0.0, 0.0], atol=1e-6)
