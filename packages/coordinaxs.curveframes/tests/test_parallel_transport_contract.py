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
``materialize``). Both halves are exercised here.

Closed-form values live in `test_frenet_serret.py` / `test_bishop.py`.
"""

__all__: tuple[str, ...] = ()

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm
import quaxed.numpy as qnp
import unxt as u

import coordinaxs.curveframes as cxfc

TAUS = [0, 0.5, 1, 2.5, jnp.pi]


@eqx.filter_jit
def jit_act(op: object, tau: object, x: object) -> object:
    """`eqx.filter_jit` handles builders carrying array leaves.

    Plain `jax.jit` cannot: a `BishopBuilder` holds ``tau_0`` (and possibly
    ``initial_normal``) as real pytree leaves, so the operator argument is not
    a valid static argument.
    """
    return cxfm.act(op, tau, x)


def inverse_rotation(builder: object, tau: u.AbstractQuantity) -> object:
    """Rotation matrix of the inverse family at ``tau`` (i.e. R^T).

    The rows of this matrix are the inverse frame's triad, i.e. the *columns*
    of the forward R.
    """
    return cxfm.TimeDep(builder).inverse.materialize(tau)[0].R


# ===================================================================
# Frame fields


class TestTriad:
    """T and the two normals form an orthonormal right-handed triad."""

    def test_location_is_the_curve(self, pt_case: SimpleNamespace) -> None:
        """The location field is gamma itself: gamma(0) = (1,0,0) km."""
        loc = pt_case.builder.location(u.Q(0, "s"))
        np.testing.assert_allclose(loc.value, [1, 0, 0], atol=pt_case.tol.location)

    def test_location_matches_curve_off_zero(self, pt_case: SimpleNamespace) -> None:
        """`location` is the curve, evaluated -- not a copy of it."""
        tau = u.Q(1.3, "s")
        np.testing.assert_allclose(
            pt_case.builder.location(tau).value,
            pt_case.builder.curve(tau).value,
            atol=pt_case.tol.location,
        )

    def test_tangent_at_zero(self, pt_case: SimpleNamespace) -> None:
        """At tau=0 on the unit circle, T = (0, 1, 0)."""
        T = pt_case.builder.tangent(u.Q(0, "s"))
        np.testing.assert_allclose(T.value, [0, 1, 0], atol=pt_case.tol.field)

    def test_tangent_at_pi_over_2(self, pt_case: SimpleNamespace) -> None:
        """At tau=pi/2, T = (-1, 0, 0)."""
        T = pt_case.builder.tangent(u.Q(jnp.pi / 2, "s"))
        np.testing.assert_allclose(T.value, [-1, 0, 0], atol=pt_case.tol.field)

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
        assert jnp.allclose(jnp.dot(e0, e1), 0, atol=pt_case.tol.orthogonality)
        assert jnp.allclose(jnp.dot(e0, e2), 0, atol=pt_case.tol.orthogonality)
        assert jnp.allclose(jnp.dot(e1, e2), 0, atol=pt_case.tol.orthogonality)

    @pytest.mark.parametrize("tau_val", TAUS)
    def test_right_handed(self, pt_case: SimpleNamespace, tau_val: float) -> None:
        """The triad is right-handed: e0 x e1 == e2."""
        e0, e1, e2 = (
            e.value for e in pt_case.fields(pt_case.builder, u.Q(tau_val, "s"))
        )
        np.testing.assert_allclose(
            jnp.cross(e0, e1), e2, atol=pt_case.tol.orthogonality
        )

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
        """The inverse triad is itself orthonormal."""
        Rinv = inverse_rotation(pt_case.builder, u.Q(tau_val, "s"))
        e0, e1, e2 = Rinv[0], Rinv[1], Rinv[2]
        assert jnp.allclose(jnp.dot(e0, e1), 0, atol=pt_case.tol.orthogonality)
        assert jnp.allclose(jnp.dot(e0, e2), 0, atol=pt_case.tol.orthogonality)
        assert jnp.allclose(jnp.dot(e1, e2), 0, atol=pt_case.tol.orthogonality)
        for v in (e0, e1, e2):
            assert jnp.allclose(jnp.linalg.norm(v), 1, atol=pt_case.tol.orthogonality)

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
            atol=pt_case.tol.transform_roundtrip,
        )

        p_rec = cxfm.act(xop.inverse, tau, p_fwd)
        np.testing.assert_allclose(
            p_rec.ustrip("km"), p.ustrip("km"), atol=pt_case.tol.transform_roundtrip
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
            atol=pt_case.tol.double_inverse,
        )

    def test_inverse_jit(self, pt_case: SimpleNamespace) -> None:
        """The inverse is JIT-compatible; the frame origin maps to gamma(0)."""
        got = jit_act(
            pt_case.xop.inverse, u.Q(0.0, "s"), u.Q(jnp.array([0.0, 0.0, 0.0]), "km")
        )
        np.testing.assert_allclose(
            got.ustrip("km"), [1, 0, 0], atol=pt_case.tol.act_roundtrip
        )


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

    def test_builder_from_bare_curve(self, pt_case: SimpleNamespace, curve) -> None:
        built = pt_case.builder_cls(curve)
        np.testing.assert_allclose(
            built.location(u.Q(0, "s")).value, [1, 0, 0], atol=pt_case.tol.location
        )

    def test_frame_is_parallel_transport_frame(self, pt_case: SimpleNamespace) -> None:
        assert isinstance(pt_case.frame, cxfc.AbstractParallelTransportFrame)

    def test_frame_is_transformed_reference_frame(
        self, pt_case: SimpleNamespace
    ) -> None:
        assert isinstance(pt_case.frame, cxf.AbstractTransformedReferenceFrame)

    def test_frame_direct_construction(self, pt_case: SimpleNamespace) -> None:
        """A frame can be built from base_frame + xop + xop_inv."""
        xop = pt_case.xop
        frame = pt_case.frame_cls(base_frame=cxf.Alice(), xop=xop, xop_inv=xop.inverse)
        assert isinstance(frame.base_frame, cxf.Alice)
        assert isinstance(frame.xop, cxfm.TimeDep)
        assert isinstance(frame.xop.builder, pt_case.builder_cls)

    def test_frame_from_curve_accepts_tau_unit(
        self, pt_case: SimpleNamespace, curve
    ) -> None:
        frame = pt_case.frame_cls.from_curve(cxf.Alice(), curve, tau_unit="yr")
        assert frame.xop.builder.tau_unit == u.unit("yr")

    def test_frame_xop_wraps_a_matching_builder(self, pt_case: SimpleNamespace) -> None:
        """`xop` is a `TimeDep` wrapping this type's builder."""
        assert isinstance(pt_case.frame.xop, cxfm.TimeDep)
        assert isinstance(pt_case.frame.xop.builder, pt_case.builder_cls)


# ===================================================================
# act


class TestAct:
    """Active-transform semantics on a bare Quantity."""

    def test_point_on_curve_maps_to_origin(self, pt_case: SimpleNamespace, arr) -> None:
        """A point at gamma(0) has delta = 0, so it maps to (0,0,0)."""
        p = u.Q(jnp.array([1, 0, 0]), "km")
        result = cxfm.act(pt_case.xop, u.Q(0, "s"), p)
        np.testing.assert_allclose(arr(result, "km"), [0, 0, 0], atol=pt_case.tol.act)

    def test_inverse_maps_origin_back_to_curve(
        self, pt_case: SimpleNamespace, arr
    ) -> None:
        """The curve-frame origin at tau=0 maps back to gamma(0)."""
        result = cxfm.act(
            pt_case.xop.inverse, u.Q(0, "s"), u.Q(jnp.array([0, 0, 0]), "km")
        )
        np.testing.assert_allclose(
            arr(result, "km"), [1, 0, 0], atol=pt_case.tol.act_roundtrip
        )

    def test_act_inverse_roundtrip(self, pt_case: SimpleNamespace, arr) -> None:
        tau, p = u.Q(0.5, "s"), u.Q(jnp.array([3, -1, 2]), "km")
        fwd = cxfm.act(pt_case.xop, tau, p)
        back = cxfm.act(pt_case.xop.inverse, tau, fwd)
        np.testing.assert_allclose(
            arr(back, "km"), arr(p, "km"), atol=pt_case.tol.act_roundtrip
        )

    def test_different_tau_gives_different_result(
        self, pt_case: SimpleNamespace, arr
    ) -> None:
        """The frame rotates along the curve, so tau matters."""
        p = u.Q(jnp.array([2, 0, 0]), "km")
        r1 = cxfm.act(pt_case.xop, u.Q(0, "s"), p)
        r2 = cxfm.act(pt_case.xop, u.Q(1, "s"), p)
        assert not np.allclose(arr(r1, "km"), arr(r2, "km"), atol=1e-3)

    def test_act_jit(self, pt_case: SimpleNamespace, arr) -> None:
        tau, p = u.Q(0, "s"), u.Q(jnp.array([2, 0, 0]), "km")
        eager = cxfm.act(pt_case.xop, tau, p)
        jitted = jit_act(pt_case.xop, tau, p)
        np.testing.assert_allclose(
            arr(jitted, "km"), arr(eager, "km"), atol=pt_case.tol.plumbing
        )

    def test_act_vmap_over_tau(self, pt_case: SimpleNamespace) -> None:
        taus = u.Q(jnp.linspace(0, 2, 5), "s")
        p = u.Q(jnp.array([2, 0, 0]), "km")
        xop = pt_case.xop
        results = jax.vmap(lambda t: cxfm.act(xop, t, p))(taus)
        assert results.shape == (5, 3)


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

    def test_alice_roundtrip(self, pt_case: SimpleNamespace, arr) -> None:
        """Alice -> curve frame -> Alice is the identity."""
        tau, p = u.Q(0.5, "s"), u.Q(jnp.array([3, -1, 2]), "km")
        fwd = cxf.frame_transition(cxf.Alice(), pt_case.frame)
        bwd = cxf.frame_transition(pt_case.frame, cxf.Alice())
        back = cxfm.act(bwd, tau, cxfm.act(fwd, tau, p))
        np.testing.assert_allclose(
            arr(back, "km"), arr(p, "km"), atol=pt_case.tol.chain
        )

    def test_alex_chain_roundtrip(self, pt_case: SimpleNamespace, arr) -> None:
        """Curve frame -> Alex -> curve frame is the identity."""
        tau, p = u.Q(0, "s"), u.Q(jnp.array([2, 0, 0]), "km")
        frame = pt_case.frame
        p_frame = cxfm.act(cxf.frame_transition(cxf.Alice(), frame), tau, p)
        p_alex = cxfm.act(cxf.frame_transition(frame, cxf.Alex()), tau, p_frame)
        p_back = cxfm.act(cxf.frame_transition(cxf.Alex(), frame), tau, p_alex)
        np.testing.assert_allclose(
            arr(p_back, "km"), arr(p_frame, "km"), atol=pt_case.tol.chain
        )

    def test_full_chain_roundtrip(self, pt_case: SimpleNamespace, arr) -> None:
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
            arr(back, "km"), arr(p, "km"), atol=pt_case.tol.full_chain
        )

    def test_transition_matches_direct_transform(
        self, pt_case: SimpleNamespace, arr
    ) -> None:
        """`frame_transition(Alice, frame)` applies the same map as the xop."""
        tau, p = u.Q(0.5, "s"), u.Q(jnp.array([2, 1, 0]), "km")
        op = cxf.frame_transition(cxf.Alice(), pt_case.frame)
        np.testing.assert_allclose(
            arr(cxfm.act(op, tau, p), "km"),
            arr(cxfm.act(pt_case.xop, tau, p), "km"),
            atol=pt_case.tol.plumbing,
        )
