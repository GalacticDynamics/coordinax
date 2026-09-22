"""R is a rotation to machine precision, not to solver tolerance.

The transport solve returns a U1 satisfying neither ``|U1| = 1`` nor
``U1 . T = 0``; orthonormalising against the tangent restores both, so ``R^T``
is an exact inverse rather than an approximate one.

Two conventions the thresholds depend on. ``1e-14`` is far below the
``1e-6``/``1e-5`` tiers the rest of the Bishop suite uses, which are too loose
to see this. And every `assert_allclose` passes ``rtol=0``, because the default
``1e-7`` is relative to the desired value and would set a ~1e-7 tolerance on
the non-zero targets here (``det R = 1``, a round trip 3 km out).
"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm
import unxt as u

import coordinaxs.curveframes as cxfc


def helix(tau):
    """Helix with pitch along the z-axis; the conftest curve, kept local."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


#: Machine-precision bar. See the module docstring on why not `tol.field`.
ATOL = 1e-14

#: Including two far from tau_0: the orthonormality residual must not grow with
#: the length of the transport, however far the solve marches.
TAUS = [1.0, 5.0, 20.0, 100.0, 500.0]


@pytest.fixture
def helix_bishop() -> cxfc.BishopBuilder:
    return cxfc.BishopBuilder(helix, "s", normal_0="auto")


class TestRIsARotation:
    """``R R^T = I`` and ``det R = 1``, at every tau, to machine precision."""

    @pytest.mark.parametrize("tau", TAUS)
    def test_rows_are_orthonormal(self, helix_bishop: cxfc.BishopBuilder, tau: float):
        R = helix_bishop.rotation_matrix(u.Q(tau, "s"))
        np.testing.assert_allclose(R @ R.T, jnp.eye(3), rtol=0, atol=ATOL)

    @pytest.mark.parametrize("tau", TAUS)
    def test_it_is_a_rotation_not_a_reflection(
        self, helix_bishop: cxfc.BishopBuilder, tau: float
    ):
        R = helix_bishop.rotation_matrix(u.Q(tau, "s"))
        np.testing.assert_allclose(jnp.linalg.det(R), 1.0, rtol=0, atol=ATOL)

    def test_the_batched_solve_is_orthonormal_too(
        self, helix_bishop: cxfc.BishopBuilder
    ):
        """`rotation_matrices` orthonormalises at its own, separate site.

        It saves interior points of one sweep rather than solving per tau.
        """
        Rs = helix_bishop.rotation_matrices(u.Q(jnp.asarray(TAUS), "s"))
        err = jnp.abs(Rs @ jnp.swapaxes(Rs, -1, -2) - jnp.eye(3))
        assert float(jnp.max(err)) < ATOL
        np.testing.assert_allclose(jnp.linalg.det(Rs), 1.0, rtol=0, atol=ATOL)


class TestTheInverseIsExact:
    """``R^T`` inverts ``R``, so a round trip through the frame is the identity.

    This is what orthonormality costs a caller: the accuracy of
    Alice -> curve frame -> Alice. Frenet--Serret, which has no solve,
    round-trips at ~2e-16; Bishop does too.
    """

    @pytest.mark.parametrize("tau", [1.0, 20.0, 100.0])
    def test_position_round_trip(self, tau: float):
        frame = cxfc.BishopFrame.from_curve(cxf.Alice(), helix, "s", normal_0="auto")
        t, p = u.Q(tau, "s"), u.Q(jnp.array([2.0, -1.0, 3.0]), "km")
        fwd = cxf.frame_transition(cxf.Alice(), frame)
        bwd = cxf.frame_transition(frame, cxf.Alice())
        back = cxfm.act(bwd, t, cxfm.act(fwd, t, p))
        # `1e-13`, not ATOL: the round trip also carries gamma's own
        # cancellation, which is ~1e-14 at tau=100 for Frenet-Serret as well.
        np.testing.assert_allclose(
            back.ustrip("km"), p.ustrip("km"), rtol=0, atol=1e-13
        )


class TestTheTransportIsStillRotationMinimising:
    """Orthogonalising U1 must not add a twist about the tangent.

    Projecting out the ``U1 . T`` residual moves U1 along T, out of the normal
    plane it spans with U2, so the gauge within that plane is untouched.
    Orthogonalising the *pair* instead would rotate the gauge and still pass
    every assertion above, which is why this is pinned.
    """

    @pytest.mark.parametrize("tau", [1.0, 5.0, 20.0])
    def test_the_frame_does_not_spin_about_the_tangent(self, tau: float):
        """``U1' . U2`` and ``U2' . U1`` are equal and opposite, and both ~0."""
        b = cxfc.BishopBuilder(helix, "s", normal_0="auto")

        def row(i: int):
            return lambda t: b.rotation_matrix(u.Q(t, "s"))[i]

        U1, U2 = row(1), row(2)
        a = jnp.dot(jax.jacfwd(U1)(tau), U2(tau))
        c = jnp.dot(jax.jacfwd(U2)(tau), U1(tau))
        # Antisymmetry is structural and holds to machine precision; the
        # common magnitude is the solve's own transport error, which does grow
        # with |tau - tau_0| and is bounded here rather than pinned.
        np.testing.assert_allclose(a + c, 0.0, rtol=0, atol=ATOL)
        assert float(jnp.abs(a)) < 1e-8
