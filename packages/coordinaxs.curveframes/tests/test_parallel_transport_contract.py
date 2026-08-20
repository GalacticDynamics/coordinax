"""The contract every `AbstractCurveFrameBuilder` must satisfy.

Frenet-Serret and Bishop differ in *which* orthonormal triad they carry along
the curve, and in the closed-form values of that triad. They do not differ in
any of the structural guarantees below -- orthonormality, right-handedness,
inverse semantics, `frame_transition` integration, JAX compatibility -- so
those are asserted once here, parametrized over both types via the `pt_case`
fixture.

The builders are wrapped by `coordinax.transforms.TimeDep`: the builder
carries the frame *fields* (``location``, ``tangent``, ...) and the
`TimeDep` carries the *transform* algebra (``act``, ``inverse``,
``evaluate_at``). Both halves are exercised here.

Closed-form values live in ``test_frenet_serret.py`` / ``test_bishop.py``.
"""

__all__: tuple[str, ...] = ()

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import quaxed.numpy as qnp
import unxt as u

import coordinaxs.curveframes as cxfc
from .conftest import circle, inverse_rotation

TAUS = [0, 0.5, 1, 2.5, jnp.pi]

#: Base point and velocity for the tangent/jet contract, as component dicts.
AT = {"x": u.Q(2.0, "km"), "y": u.Q(-1.0, "km"), "z": u.Q(3.0, "km")}
VEL = {"x": u.Q(0.1, "km/s"), "y": u.Q(0.2, "km/s"), "z": u.Q(-0.3, "km/s")}


@eqx.filter_jit
def jit_act(op: object, tau: object, x: object) -> object:
    """`eqx.filter_jit` handles builders carrying array leaves.

    Plain `jax.jit` cannot: a `BishopBuilder` holds ``tau_0`` (and possibly
    ``initial_normal``) as real pytree leaves, so the operator argument is not
    a valid static argument.
    """
    return cxfm.act(op, tau, x)


# ===================================================================
# Frame fields


class TestTriad:
    """T and the two normals form an orthonormal right-handed triad."""

    def test_location_is_the_curve(self, pt_case: SimpleNamespace) -> None:
        """The location field is gamma itself: gamma(0) = (1,0,0) km."""
        loc = pt_case.builder.location(u.Q(0, "s"))
        np.testing.assert_allclose(loc.value, [1, 0, 0], atol=pt_case.tol.tight)

    def test_location_matches_curve_off_zero(self, pt_case: SimpleNamespace) -> None:
        """`location` is the curve, evaluated -- not a copy of it."""
        tau = u.Q(1.3, "s")
        np.testing.assert_allclose(
            pt_case.builder.location(tau).value,
            pt_case.builder.curve(tau).value,
            atol=pt_case.tol.tight,
        )

    @pytest.mark.parametrize(
        ("tau_val", "expected"), [(0, [0, 1, 0]), (jnp.pi / 2, [-1, 0, 0])]
    )
    def test_tangent_value(
        self, pt_case: SimpleNamespace, tau_val: float, expected: list[float]
    ) -> None:
        """On the unit circle T = (0,1,0) at tau=0 and (-1,0,0) at tau=pi/2."""
        T = pt_case.builder.tangent(u.Q(tau_val, "s"))
        np.testing.assert_allclose(T.value, expected, atol=pt_case.tol.field)

    @pytest.mark.parametrize("tau_val", TAUS)
    def test_unit_length(self, pt_case: SimpleNamespace, tau_val: float) -> None:
        """Every frame field is a unit vector."""
        triad = pt_case.fields(pt_case.builder, u.Q(tau_val, "s"))
        for name, e in zip(pt_case.triad, triad, strict=True):
            assert jnp.allclose(jnp.linalg.norm(e.value), 1, atol=pt_case.tol.field), (
                name
            )

    @pytest.mark.parametrize("tau_val", TAUS)
    def test_orthogonality(self, pt_case: SimpleNamespace, tau_val: float) -> None:
        """The three frame fields are mutually orthogonal."""
        e0, e1, e2 = (
            e.value for e in pt_case.fields(pt_case.builder, u.Q(tau_val, "s"))
        )
        assert jnp.allclose(jnp.dot(e0, e1), 0, atol=pt_case.tol.field)
        assert jnp.allclose(jnp.dot(e0, e2), 0, atol=pt_case.tol.field)
        assert jnp.allclose(jnp.dot(e1, e2), 0, atol=pt_case.tol.field)

    @pytest.mark.parametrize("tau_val", TAUS)
    def test_right_handed(self, pt_case: SimpleNamespace, tau_val: float) -> None:
        """The triad is right-handed: e0 x e1 == e2."""
        e0, e1, e2 = (
            e.value for e in pt_case.fields(pt_case.builder, u.Q(tau_val, "s"))
        )
        np.testing.assert_allclose(jnp.cross(e0, e1), e2, atol=pt_case.tol.field)

    @pytest.mark.parametrize("tau_val", TAUS)
    def test_rotation_matrix_rows_are_the_triad(
        self, pt_case: SimpleNamespace, tau_val: float
    ) -> None:
        """The builder's R has the frame fields as its rows."""
        tau = u.Q(tau_val, "s")
        R = pt_case.builder.rotation_matrix(tau)
        for i, e in enumerate(pt_case.fields(pt_case.builder, tau)):
            np.testing.assert_allclose(R[i], e.value, atol=pt_case.tol.field)


# ===================================================================
# Inverse


class TestInverse:
    """The inverse frame fields satisfy R^T semantics."""

    def test_inverse_is_timedep(self, pt_case: SimpleNamespace) -> None:
        """`.inverse` is itself a `TimeDep` family."""
        assert isinstance(pt_case.xop.inverse, cxfm.TimeDep)

    @pytest.mark.parametrize("tau_val", TAUS)
    def test_inverse_orthonormality(
        self, pt_case: SimpleNamespace, tau_val: float
    ) -> None:
        """The inverse triad is itself orthonormal.

        Not fully subsumed by `test_inverse_rotation_is_transpose` plus the
        forward triad's orthonormality: chaining those two bounds (each good
        to `tol.field`) only guarantees this one to a small constant multiple
        of `tol.field`, via the triangle inequality on the dot products.
        """
        Rinv = inverse_rotation(pt_case.builder, u.Q(tau_val, "s"))
        e0, e1, e2 = Rinv[0], Rinv[1], Rinv[2]
        atol = 3 * pt_case.tol.field
        assert jnp.allclose(jnp.dot(e0, e1), 0, atol=atol)
        assert jnp.allclose(jnp.dot(e0, e2), 0, atol=atol)
        assert jnp.allclose(jnp.dot(e1, e2), 0, atol=atol)
        for v in (e0, e1, e2):
            assert jnp.allclose(jnp.linalg.norm(v), 1, atol=atol)

    @pytest.mark.parametrize("tau_val", TAUS)
    def test_inverse_rotation_is_transpose(
        self, pt_case: SimpleNamespace, tau_val: float
    ) -> None:
        """The inverse triad is the *columns* of the forward R."""
        tau = u.Q(tau_val, "s")
        R = pt_case.builder.rotation_matrix(tau)
        Rinv = inverse_rotation(pt_case.builder, tau)
        np.testing.assert_allclose(Rinv, R.T, atol=pt_case.tol.field)

    @pytest.mark.parametrize("tau_val", [0, 1, jnp.pi])
    def test_roundtrip_forward_inverse(
        self, pt_case: SimpleNamespace, tau_val: float
    ) -> None:
        """R_inv @ (R @ (p - gamma) - gamma_inv) == p.

        The forward half is built by hand from the frame-field accessors and
        cross-checked against `act`, so this pins both the fields and the
        transform they induce.
        """
        tau = u.Q(tau_val, "s")
        p = u.Q(jnp.array([2.0, 3.0, 4.0]), "km")

        diff = p - pt_case.builder.location(tau)
        p_fwd = qnp.stack(
            [qnp.sum(e * diff) for e in pt_case.fields(pt_case.builder, tau)]
        )

        xop = pt_case.xop
        np.testing.assert_allclose(
            cxfm.act(xop, tau, p).ustrip("km"),
            p_fwd.ustrip("km"),
            atol=pt_case.tol.loose,
        )

        p_rec = cxfm.act(xop.inverse, tau, p_fwd)
        np.testing.assert_allclose(
            p_rec.ustrip("km"), p.ustrip("km"), atol=pt_case.tol.loose
        )

    @pytest.mark.parametrize("tau_val", [0, 1, jnp.pi])
    def test_double_inverse(self, pt_case: SimpleNamespace, tau_val: float) -> None:
        """`inverse.inverse` recovers the original family, builder and all."""
        tau = u.Q(tau_val, "s")
        xop = pt_case.xop
        assert xop.inverse.inverse.builder is xop.builder

        p = u.Q(jnp.array([2.0, 3.0, 4.0]), "km")
        np.testing.assert_allclose(
            cxfm.act(xop.inverse.inverse, tau, p).ustrip("km"),
            cxfm.act(xop, tau, p).ustrip("km"),
            atol=pt_case.tol.field,
        )

    def test_inverse_jit(self, pt_case: SimpleNamespace) -> None:
        """The inverse is JIT-compatible; the frame origin maps to gamma(0)."""
        got = jit_act(
            pt_case.xop.inverse, u.Q(0.0, "s"), u.Q(jnp.array([0.0, 0.0, 0.0]), "km")
        )
        np.testing.assert_allclose(got.ustrip("km"), [1, 0, 0], atol=pt_case.tol.field)


# ===================================================================
# JAX


class TestJAX:
    """The builder and its family compose with jit and vmap."""

    def test_jit_tangent(self, pt_case: SimpleNamespace) -> None:
        T = eqx.filter_jit(pt_case.builder.tangent)(u.Q(0.0, "s"))
        np.testing.assert_allclose(T.value, [0, 1, 0], atol=pt_case.tol.field)

    def test_vmap_tangent(self, pt_case: SimpleNamespace) -> None:
        taus = u.Q(jnp.linspace(0, 2 * jnp.pi, 8), "s")
        Ts = jax.vmap(pt_case.builder.tangent)(taus)
        norms = jnp.sqrt(jnp.sum(Ts.value**2, axis=-1))
        assert jnp.allclose(norms, 1, atol=pt_case.tol.field)


# ===================================================================
# Constructors


class TestConstructors:
    """A builder is built from a bare curve; the frame wraps it in TimeDep."""

    def test_builder_from_bare_curve(self, pt_case: SimpleNamespace) -> None:
        built = pt_case.builder_cls(circle, "s")
        np.testing.assert_allclose(
            built.location(u.Q(0, "s")).value, [1, 0, 0], atol=pt_case.tol.tight
        )

    def test_frame_is_well_typed(self, pt_case: SimpleNamespace) -> None:
        """Both construction routes give a frame wrapping this type's builder.

        `from_curve` (``pt_case.frame``) and the direct
        ``base_frame + xop + xop_inv`` constructor must agree on every
        structural claim.
        """
        xop = pt_case.xop
        direct = pt_case.frame_cls(base_frame=cxf.Alice(), xop=xop, xop_inv=xop.inverse)
        for frame in (pt_case.frame, direct):
            assert isinstance(frame, cxfc.AbstractParallelTransportFrame)
            assert isinstance(frame, cxf.AbstractTransformedReferenceFrame)
            assert isinstance(frame.base_frame, cxf.Alice)
            assert isinstance(frame.xop, cxfm.TimeDep)
            assert isinstance(frame.xop.builder, pt_case.builder_cls)

    def test_frame_from_curve_accepts_tau_unit(self, pt_case: SimpleNamespace) -> None:
        frame = pt_case.frame_cls.from_curve(cxf.Alice(), circle, tau_unit="yr")
        assert frame.xop.builder.tau_unit == u.unit("yr")


# ===================================================================
# Opaque units


class TestOpaqueUnits:
    """A curve whose internal unit (yr) differs from the caller's."""

    def test_tau_unit_is_stored(self, pt_case: SimpleNamespace) -> None:
        assert pt_case.yr_builder.tau_unit == u.unit("yr")

    def test_tangent_at_zero(self, pt_case: SimpleNamespace) -> None:
        T = pt_case.yr_builder.tangent(u.Q(0, "yr"))
        np.testing.assert_allclose(T.value, [0, 1, 0], atol=pt_case.tol.field)

    def test_triad_orthogonal_at_zero(self, pt_case: SimpleNamespace) -> None:
        e0, e1, e2 = (e.value for e in pt_case.fields(pt_case.yr_builder, u.Q(0, "yr")))
        assert jnp.allclose(jnp.dot(e0, e1), 0, atol=pt_case.tol.field)
        assert jnp.allclose(jnp.dot(e0, e2), 0, atol=pt_case.tol.field)
        assert jnp.allclose(jnp.dot(e1, e2), 0, atol=pt_case.tol.field)

    def test_inverse_maps_origin_to_curve(self, pt_case: SimpleNamespace) -> None:
        """For the yr-circle at tau=0 the curve is at (5, 0, 0) km."""
        inv = cxfm.TimeDep(pt_case.yr_builder).inverse
        got = cxfm.act(inv, u.Q(0.0, "yr"), u.Q(jnp.array([0.0, 0.0, 0.0]), "km"))
        np.testing.assert_allclose(got.ustrip("km"), [5, 0, 0], atol=pt_case.tol.loose)


# ===================================================================
# act


class TestAct:
    """Active-transform semantics on a bare Quantity."""

    def test_point_on_curve_maps_to_origin(self, pt_case: SimpleNamespace) -> None:
        """A point at gamma(0) has delta = 0, so it maps to (0,0,0)."""
        p = u.Q(jnp.array([1, 0, 0]), "km")
        result = cxfm.act(pt_case.xop, u.Q(0, "s"), p)
        np.testing.assert_allclose(
            result.ustrip("km"), [0, 0, 0], atol=pt_case.tol.field
        )

    def test_inverse_maps_origin_back_to_curve(self, pt_case: SimpleNamespace) -> None:
        """The curve-frame origin at tau=0 maps back to gamma(0)."""
        result = cxfm.act(
            pt_case.xop.inverse, u.Q(0, "s"), u.Q(jnp.array([0, 0, 0]), "km")
        )
        np.testing.assert_allclose(
            result.ustrip("km"), [1, 0, 0], atol=pt_case.tol.field
        )

    def test_act_inverse_roundtrip(self, pt_case: SimpleNamespace) -> None:
        tau, p = u.Q(0.5, "s"), u.Q(jnp.array([3, -1, 2]), "km")
        fwd = cxfm.act(pt_case.xop, tau, p)
        back = cxfm.act(pt_case.xop.inverse, tau, fwd)
        np.testing.assert_allclose(
            back.ustrip("km"), p.ustrip("km"), atol=pt_case.tol.field
        )

    def test_different_tau_gives_different_result(
        self, pt_case: SimpleNamespace
    ) -> None:
        """The frame rotates along the curve, so tau matters."""
        p = u.Q(jnp.array([2, 0, 0]), "km")
        r1 = cxfm.act(pt_case.xop, u.Q(0, "s"), p)
        r2 = cxfm.act(pt_case.xop, u.Q(1, "s"), p)
        assert not np.allclose(r1.ustrip("km"), r2.ustrip("km"), atol=1e-3)

    def test_act_jit(self, pt_case: SimpleNamespace) -> None:
        tau, p = u.Q(0, "s"), u.Q(jnp.array([2, 0, 0]), "km")
        eager = cxfm.act(pt_case.xop, tau, p)
        jitted = jit_act(pt_case.xop, tau, p)
        np.testing.assert_allclose(
            jitted.ustrip("km"), eager.ustrip("km"), atol=pt_case.tol.plumbing
        )

    def test_act_vmap_over_tau(self, pt_case: SimpleNamespace) -> None:
        taus = u.Q(jnp.linspace(0, 2, 5), "s")
        p = u.Q(jnp.array([2, 0, 0]), "km")
        xop = pt_case.xop
        results = jax.vmap(lambda t: cxfm.act(xop, t, p))(taus)
        eager = [cxfm.act(xop, taus[i], p).ustrip("km") for i in range(len(taus))]
        np.testing.assert_allclose(
            results.ustrip("km"), eager, atol=pt_case.tol.plumbing
        )


# ===================================================================
# Tangent data and jets


def _prolongation_by_finite_differences(
    xop: cxfm.TimeDep, tau: u.AbstractQuantity, h: float = 1e-4
) -> dict:
    r"""The order-1 prolongation of ``xop``, by central differences.

    The first prolongation is the total $\tau$-derivative of the *point*
    action along the straight-line curve through `AT` with velocity `VEL`,
    so it can be recovered from point actions alone -- no autodiff, and
    therefore a genuinely independent oracle for `act`/`act_jet`.
    """

    def y(t: u.AbstractQuantity) -> dict:
        dt = t - tau
        x = {k: AT[k] + VEL[k] * dt for k in AT}
        return cxfm.act(xop, t, x, cxc.cart3d, cxr.point)

    dtau = u.Q(h, "s")
    plus, minus = y(tau + dtau), y(tau - dtau)
    return {k: (plus[k] - minus[k]) / (2 * dtau) for k in AT}


class TestTangentAndJet:
    r"""Tangent data and jets propagate through the $\tau$-dependent family.

    A curve frame is time-dependent, so ``act`` on a velocity is the
    *kinematic prolongation*, not the frozen-$\tau$ pushforward: it must pick
    up the $\dot R$ and $\dot\gamma$ terms. That path funnels through
    ``jax.jvp`` of the point action, which is why it is a distinct capability
    from the point action itself -- and why it must be exercised for *both*
    frame types.
    """

    @pytest.mark.parametrize("tau_val", [0.0, 0.7, 2.0])
    def test_act_on_velocity_matches_finite_differences(
        self, pt_case: SimpleNamespace, tau_val: float
    ) -> None:
        """``act`` on tangent data is the total tau-derivative it claims to be."""
        tau = u.Q(tau_val, "s")
        got = cxfm.act(
            pt_case.xop, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=AT
        )
        want = _prolongation_by_finite_differences(pt_case.xop, tau)
        for k in AT:
            np.testing.assert_allclose(
                got[k].ustrip("km/s"),
                want[k].ustrip("km/s"),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"component {k}",
            )

    @pytest.mark.parametrize("tau_val", [0.0, 0.7, 2.0])
    def test_act_jet_matches_finite_differences(
        self, pt_case: SimpleNamespace, tau_val: float
    ) -> None:
        """``act_jet`` slot 1 agrees with the same oracle; slot 0 is the point."""
        tau = u.Q(tau_val, "s")
        out = cxfm.act_jet(pt_case.xop, tau, {0: AT, 1: VEL}, cxc.cart3d)

        point = cxfm.act(pt_case.xop, tau, AT, cxc.cart3d, cxr.point)
        want = _prolongation_by_finite_differences(pt_case.xop, tau)
        for k in AT:
            np.testing.assert_allclose(
                out[0][k].ustrip("km"),
                point[k].ustrip("km"),
                atol=pt_case.tol.field,
                err_msg=f"slot 0, component {k}",
            )
            np.testing.assert_allclose(
                out[1][k].ustrip("km/s"),
                want[k].ustrip("km/s"),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"slot 1, component {k}",
            )

    def test_act_on_velocity_is_not_the_pushforward(
        self, pt_case: SimpleNamespace
    ) -> None:
        r"""Discriminator: the frozen-tau pushforward gives a *different* answer.

        Without the $\dot R$/$\dot\gamma$ terms the two would coincide, so
        this pins that the prolongation is actually being taken.
        """
        tau = u.Q(0.7, "s")
        prolonged = cxfm.act(
            pt_case.xop, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=AT
        )
        frozen = cxfm.pushforward(
            pt_case.xop, tau, VEL, cxc.cart3d, cxr.coord_vel, at=AT
        )
        assert not np.allclose(
            [prolonged[k].ustrip("km/s") for k in AT],
            [frozen[k].ustrip("km/s") for k in AT],
            atol=1e-3,
        )

    def test_act_on_velocity_jit(self, pt_case: SimpleNamespace) -> None:
        """The tangent path is JIT-compatible."""
        tau = u.Q(0.7, "s")
        kw = {"at": AT}
        eager = cxfm.act(
            pt_case.xop, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, **kw
        )
        jitted = eqx.filter_jit(
            lambda op: cxfm.act(
                op, tau, VEL, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, **kw
            )
        )(pt_case.xop)
        for k in AT:
            np.testing.assert_allclose(
                jitted[k].ustrip("km/s"),
                eager[k].ustrip("km/s"),
                atol=pt_case.tol.plumbing,
            )


# ===================================================================
# frame_transition


class TestFrameTransition:
    """`frame_transition` integrates with the curve frames."""

    @pytest.mark.parametrize("direction", ["to", "from"])
    def test_transition_is_a_transform(
        self, pt_case: SimpleNamespace, direction: str
    ) -> None:
        a, b = cxf.Alice(), pt_case.frame
        op = (
            cxf.frame_transition(a, b)
            if direction == "to"
            else cxf.frame_transition(b, a)
        )
        assert isinstance(op, cxfm.AbstractTransform)

    def test_alice_roundtrip(self, pt_case: SimpleNamespace) -> None:
        """Alice -> curve frame -> Alice is the identity."""
        tau, p = u.Q(0.5, "s"), u.Q(jnp.array([3, -1, 2]), "km")
        fwd = cxf.frame_transition(cxf.Alice(), pt_case.frame)
        bwd = cxf.frame_transition(pt_case.frame, cxf.Alice())
        back = cxfm.act(bwd, tau, cxfm.act(fwd, tau, p))
        np.testing.assert_allclose(
            back.ustrip("km"), p.ustrip("km"), atol=pt_case.tol.field
        )

    def test_alex_chain_roundtrip(self, pt_case: SimpleNamespace) -> None:
        """Curve frame -> Alex -> curve frame is the identity."""
        tau, p = u.Q(0, "s"), u.Q(jnp.array([2, 0, 0]), "km")
        frame = pt_case.frame
        p_frame = cxfm.act(cxf.frame_transition(cxf.Alice(), frame), tau, p)
        p_alex = cxfm.act(cxf.frame_transition(frame, cxf.Alex()), tau, p_frame)
        p_back = cxfm.act(cxf.frame_transition(cxf.Alex(), frame), tau, p_alex)
        np.testing.assert_allclose(
            p_back.ustrip("km"), p_frame.ustrip("km"), atol=pt_case.tol.field
        )

    def test_full_chain_roundtrip(self, pt_case: SimpleNamespace) -> None:
        """Alice -> frame -> Alex -> frame -> Alice recovers the original."""
        tau, p = u.Q(0.3, "s"), u.Q(jnp.array([5, -2, 1]), "km")
        frame = pt_case.frame
        op1 = cxf.frame_transition(cxf.Alice(), frame)
        op2 = cxf.frame_transition(frame, cxf.Alex())
        p_alex = cxfm.act(op2, tau, cxfm.act(op1, tau, p))
        op3 = cxf.frame_transition(cxf.Alex(), frame)
        op4 = cxf.frame_transition(frame, cxf.Alice())
        back = cxfm.act(op4, tau, cxfm.act(op3, tau, p_alex))
        np.testing.assert_allclose(
            back.ustrip("km"), p.ustrip("km"), atol=pt_case.tol.loose
        )

    def test_transition_matches_direct_transform(
        self, pt_case: SimpleNamespace
    ) -> None:
        """`frame_transition(Alice, frame)` applies the same map as the xop."""
        tau, p = u.Q(0.5, "s"), u.Q(jnp.array([2, 1, 0]), "km")
        op = cxf.frame_transition(cxf.Alice(), pt_case.frame)
        np.testing.assert_allclose(
            cxfm.act(op, tau, p).ustrip("km"),
            cxfm.act(pt_case.xop, tau, p).ustrip("km"),
            atol=pt_case.tol.plumbing,
        )
