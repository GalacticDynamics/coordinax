"""The signed planar frame: defined where Frenet--Serret is not.

Closed-form values and the degeneracies that motivate the type. The
structural guarantees shared with the other builders -- orthonormality,
right-handedness, `frame_transition` integration, JAX compatibility -- are
asserted once in `test_parallel_transport_contract.py`, parametrized.
"""

__all__: tuple[str, ...] = ()

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm
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

    def test_frenet_refuses_on_the_same_straight_line(self) -> None:
        """The contrast that motivates the type, pinned on the line too."""
        with pytest.raises(Exception, match="curvature"):
            cxfc.FrenetSerretBuilder(straight_line, "s").normal(u.Q(5.0, "s"))


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

    def test_vmap_over_a_quantity_plane_normal(self) -> None:
        """A traced `Quantity` `plane_normal` must not hit `__array__`.

        `jnp.asarray` on a traced `Quantity` raises `TracerArrayConversionError`;
        `_float` (shared with `BishopBuilder`) strips the unit first instead.
        """
        normals = u.Q(jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), "")

        def at_normal(n: u.AbstractQuantity) -> jax.Array:
            return (
                cxfc.SignedPlanarBuilder(circle, "s", plane_normal=n)
                .normal(u.Q(0.0, "s"))
                .value
            )

        got = jax.vmap(at_normal)(normals)
        np.testing.assert_allclose(got[0], -got[1], atol=1e-10)


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
            b.normal(u.Q(0.3, "s"))

    def test_the_guard_is_pointwise(self) -> None:
        """A wrong plane can pass at isolated parameters, by construction.

        At ``tau=0`` the circle's tangent is ``(-0, 1, 0)``, which lies in the
        yz-plane exactly, so a builder given ``plane_normal=x-hat`` is asked
        nothing false at that one point and answers. The guard checks the
        parameter it is evaluated at -- there is no curve-wide check a
        pointwise API could run.
        """
        b = cxfc.SignedPlanarBuilder(
            circle, "s", plane_normal=jnp.array([1.0, 0.0, 0.0])
        )
        N = b.normal(u.Q(0.0, "s"))  # does not raise
        np.testing.assert_allclose(N.value, [0.0, 0.0, 1.0], atol=1e-10)

    def test_near_planar_curve_is_accepted(self) -> None:
        """Tolerance is sqrt(eps) ~ 1.5e-8 in f64, so a 1e-9 drift passes."""
        N = cxfc.SignedPlanarBuilder(wobble, "s").normal(u.Q(0.0, "s"))
        np.testing.assert_allclose(N.value, [-1.0, 0.0, 0.0], atol=1e-6)


def circle_r2(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Circle of radius 2 km: signed curvature 0.5 per km."""
    t = tau.ustrip("s")
    return u.Q(2.0 * jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


class TestSignedCurvature:
    """`kappa_s` is finite at inflections and changes sign through them."""

    def test_unit_circle(self) -> None:
        """Radius 1 km, traversed CCW: kappa_s = +1 per km."""
        k = cxfc.SignedPlanarBuilder(circle, "s").signed_curvature(u.Q(0.4, "s"))
        np.testing.assert_allclose(k.ustrip("1/km"), 1.0, atol=1e-8)

    def test_radius_scales_inversely(self) -> None:
        """Radius 2 km: kappa_s = 0.5 per km."""
        k = cxfc.SignedPlanarBuilder(circle_r2, "s").signed_curvature(u.Q(0.4, "s"))
        np.testing.assert_allclose(k.ustrip("1/km"), 0.5, atol=1e-8)

    def test_dimension_is_inverse_length(self) -> None:
        k = cxfc.SignedPlanarBuilder(circle, "s").signed_curvature(u.Q(0.0, "s"))
        assert u.dimension_of(k) == u.dimension("1/length")

    def test_zero_on_a_straight_line(self) -> None:
        k = cxfc.SignedPlanarBuilder(straight_line, "s").signed_curvature(u.Q(3.0, "s"))
        np.testing.assert_allclose(k.ustrip("1/km"), 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        ("t", "expected"), [(-1e-3, -6e-3), (0.0, 0.0), (1e-3, 6e-3)]
    )
    def test_passes_smoothly_through_an_inflection(
        self, t: float, expected: float
    ) -> None:
        """On ``(t, t^3, 0)``: finite at the inflection, and sign-changing.

        These are the values the issue's own worked table reports, which is
        what pins the sign convention against the (negated) formula its prose
        gives.
        """
        k = cxfc.SignedPlanarBuilder(cubic, "s").signed_curvature(u.Q(t, "s"))
        assert jnp.isfinite(k.ustrip("1/km"))
        np.testing.assert_allclose(k.ustrip("1/km"), expected, atol=1e-9)

    def test_flipping_the_plane_normal_flips_the_sign(self) -> None:
        up = cxfc.SignedPlanarBuilder(circle, "s").signed_curvature(u.Q(0.4, "s"))
        down = cxfc.SignedPlanarBuilder(circle, "s", plane_normal=-Z).signed_curvature(
            u.Q(0.4, "s")
        )
        np.testing.assert_allclose(down.ustrip("1/km"), -up.ustrip("1/km"), atol=1e-10)

    def test_helix_raises(self) -> None:
        """The planarity guard covers this accessor too, not just the triad."""
        with pytest.raises(Exception, match="BishopBuilder"):
            cxfc.SignedPlanarBuilder(helix, "s").signed_curvature(u.Q(0.0, "s"))

    def test_dt_ds_equals_kappa_s_times_normal(self) -> None:
        """`dT/ds = kappa_s N`: the relation that fixes the sign convention.

        If this passes with the issue's stated formula it would fail; it only
        holds for `kappa_s = (gamma' x gamma'') . n_hat / |gamma'|^3`.
        """
        b = cxfc.SignedPlanarBuilder(cubic, "s")
        tau = u.Q(0.8, "s")

        dT_dtau = u.experimental.jacfwd(b.tangent, units=("s",))(tau)
        speed = jnp.linalg.norm(
            u.experimental.jacfwd(b.curve, units=("s",))(tau).ustrip("km/s")
        )
        lhs = dT_dtau.ustrip("1/s") / speed  # dT/ds, per km

        rhs = b.signed_curvature(tau).ustrip("1/km") * b.normal(tau).value
        np.testing.assert_allclose(lhs, rhs, atol=1e-8)


class TestFrame:
    """The frame class, and the dispatches it inherits from the ABC."""

    def test_from_curve_builds_the_right_builder(self) -> None:
        frame = cxfc.SignedPlanarFrame.from_curve(cxf.Alice(), circle, "s")
        assert frame.base_frame == cxf.Alice()
        assert isinstance(frame.xop.builder, cxfc.SignedPlanarBuilder)

    def test_plane_normal_reaches_the_builder(self) -> None:
        frame = cxfc.SignedPlanarFrame.from_curve(
            cxf.Alice(), circle, "s", plane_normal=-Z
        )
        np.testing.assert_allclose(frame.xop.builder.plane_normal, -Z, atol=0)

    def test_station_reaches_the_builder(self) -> None:
        """`station=` is forwarded to the builder, and lands in its own slot.

        `from_curve` constructs the builder positionally, and `station` and
        `plane_normal` are adjacent in that order -- so they are the pair a
        slip would transpose. `test_plane_normal_reaches_the_builder` catches
        that indirectly; this says it directly.
        """
        station = u.Q(0.4, "s")
        frame = cxfc.SignedPlanarFrame.from_curve(
            cxf.Alice(), circle, "s", station=station
        )
        assert frame.xop.builder.station == station
        assert frame.xop.builder.plane_normal is None

    def test_frame_transition_round_trip(self) -> None:
        """`frame_transition` is inherited, not registered. Pin that."""
        frame = cxfc.SignedPlanarFrame.from_curve(cxf.Alice(), circle, "s")
        tau = u.Q(0.0, "s")
        p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")

        to_curve = cxf.frame_transition(cxf.Alice(), frame)
        # gamma(0) = (1, 0, 0) km, so the curve-frame origin sits on p.
        out = cxfm.act(to_curve, tau, p)
        np.testing.assert_allclose(out.ustrip("km"), [0.0, 0.0, 0.0], atol=1e-10)

        from_curve = cxf.frame_transition(frame, cxf.Alice())
        back = cxfm.act(from_curve, tau, out)
        np.testing.assert_allclose(back.ustrip("km"), p.ustrip("km"), atol=1e-10)


class PlanarCircle(eqx.Module):
    """A circle whose ``radius`` (in km) is a differentiable pytree leaf."""

    radius: Any

    def __call__(self, tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        return u.Q(
            self.radius * jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km"
        )


class TestCurvatureVector:
    """`kappa_s * N`: continuous through an inflection, where `N` alone is not.

    The curvature vector is the object that behaves the way a reader expects
    the Frenet normal to behave. It points at the centre of curvature, and it
    passes through zero at an inflection instead of flipping -- which the
    *unit* normal cannot do, since "unit", "points at the centre of
    curvature" and "continuous" are incompatible there.
    """

    def test_equals_signed_curvature_times_normal(self) -> None:
        b = cxfc.SignedPlanarBuilder(cubic, "s")
        tau = u.Q(0.7, "s")
        expect = b.signed_curvature(tau).ustrip("1/km") * b.normal(tau).value
        np.testing.assert_allclose(
            b.curvature_vector(tau).ustrip("1/km"), expect, atol=1e-12
        )

    def test_dimension_is_inverse_length(self) -> None:
        kv = cxfc.SignedPlanarBuilder(circle, "s").curvature_vector(u.Q(0.0, "s"))
        assert u.dimension_of(kv) == u.dimension("1/length")

    @pytest.mark.parametrize(
        ("t", "expected"),
        # kappa_s = 6t / (1 + 9t^4)^{3/2}, so ~6t for small t.
        [(-1e-6, [0.0, -6e-6, 0.0]), (0.0, [0.0, 0.0, 0.0]), (1e-6, [0.0, 6e-6, 0.0])],
    )
    def test_passes_through_zero_at_the_inflection(
        self, t: float, expected: list[float]
    ) -> None:
        """Continuous and vanishing, where the unit normal jumps by 180 deg."""
        kv = cxfc.SignedPlanarBuilder(cubic, "s").curvature_vector(u.Q(t, "s"))
        np.testing.assert_allclose(kv.ustrip("1/km"), expected, atol=1e-12)

    def test_points_at_the_centre_of_curvature(self) -> None:
        """Where curvature is nonzero it is parallel to the Frenet normal.

        Both branches of the cubic, so the branch on which the signed-planar
        *unit* normal opposes Frenet is covered too.
        """
        sp = cxfc.SignedPlanarBuilder(cubic, "s")
        fs = cxfc.FrenetSerretBuilder(cubic, "s")
        for t in (-1.0, 1.0):
            tau = u.Q(t, "s")
            kv = sp.curvature_vector(tau).ustrip("1/km")
            direction = kv / jnp.linalg.norm(kv)
            np.testing.assert_allclose(direction, fs.normal(tau).value, atol=1e-8)

    def test_magnitude_is_abs_signed_curvature(self) -> None:
        b = cxfc.SignedPlanarBuilder(cubic, "s")
        for t in (-1.0, -0.2, 0.0, 0.2, 1.0):
            tau = u.Q(t, "s")
            got = jnp.linalg.norm(b.curvature_vector(tau).ustrip("1/km"))
            want = abs(b.signed_curvature(tau).ustrip("1/km"))
            np.testing.assert_allclose(got, want, atol=1e-10)

    def test_zero_on_a_straight_line(self) -> None:
        kv = cxfc.SignedPlanarBuilder(straight_line, "s").curvature_vector(
            u.Q(3.0, "s")
        )
        np.testing.assert_allclose(kv.ustrip("1/km"), [0.0, 0.0, 0.0], atol=1e-12)


class TestJAX:
    """`Helix` in test_capabilities.py is not planar, so grad coverage lives here."""

    def test_grad_through_a_curve_parameter(self) -> None:
        """`kappa_s = 1/r`, so `d(kappa_s)/dr = -1/r^2` in closed form."""

        def kappa(radius: Any) -> Any:
            builder = cxfc.SignedPlanarBuilder(PlanarCircle(radius), "s")
            return builder.signed_curvature(u.Q(0.3, "s")).ustrip("1/km")

        got = jax.grad(kappa)(1.5)
        np.testing.assert_allclose(got, -1.0 / 1.5**2, rtol=1e-6)

    def test_jit_and_vmap_across_the_inflection(self) -> None:
        """The whole point: a batch spanning `t = 0` stays finite under jit."""
        builder = cxfc.SignedPlanarBuilder(cubic, "s")
        taus = u.Q(jnp.linspace(-1.0, 1.0, 5), "s")

        @eqx.filter_jit
        def normal_at(tau: u.AbstractQuantity) -> jax.Array:
            return builder.normal(tau).value

        got = jax.vmap(normal_at)(taus)
        assert got.shape == (5, 3)
        assert jnp.all(jnp.isfinite(got))

    def test_eager_and_jit_agree(self) -> None:
        builder = cxfc.SignedPlanarBuilder(cubic, "s")
        tau = u.Q(0.0, "s")
        eager = builder.rotation_matrix(tau)
        jitted = eqx.filter_jit(builder.rotation_matrix)(tau)
        np.testing.assert_allclose(jitted, eager, atol=1e-12)

    def test_vmap_over_station(self) -> None:
        """A frame field: vmap the fixed curve parameter, not tau."""
        stations = u.Q(jnp.linspace(0.0, 1.5, 5), "s")

        def at_station(g: u.AbstractQuantity) -> jax.Array:
            b = cxfc.SignedPlanarBuilder(circle, "s", g)
            return b.normal(u.Q(0.0, "s")).value

        batched = jax.vmap(at_station)(stations)
        assert batched.shape == (5, 3)

        for i in range(5):
            expected = at_station(stations[i])
            assert jnp.allclose(batched[i], expected, atol=1e-6)
