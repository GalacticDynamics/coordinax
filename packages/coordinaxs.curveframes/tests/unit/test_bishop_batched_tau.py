"""`rotation_matrices` evaluates many tau from one ODE solve (#650)."""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def helix(tau):
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


def _builder():
    return cxfc.BishopBuilder(helix, "s")


def _per_tau(b, taus):
    return jnp.stack([b.rotation_matrix(u.Q(float(t), "s")) for t in taus])


class TestAgreesWithThePerTauAccessor:
    """One solve must give what N solves give -- otherwise it is just faster."""

    @pytest.mark.parametrize(
        "taus",
        [
            jnp.linspace(0.1, 2.0, 8),
            jnp.asarray([-1.5, -0.5]),
            jnp.asarray([1.7, 0.3, 2.0, 0.9]),
            jnp.asarray([0.0, 0.5, 1.0]),
            jnp.asarray([0.7]),
        ],
        ids=["ascending", "negative-side", "unsorted", "includes-tau_0", "single"],
    )
    def test_matches(self, taus):
        b = _builder()
        got = b.rotation_matrices(u.Q(taus, "s"))
        assert got.shape == (len(taus), 3, 3)
        assert jnp.allclose(got, _per_tau(b, taus), atol=1e-7)

    def test_unsorted_input_keeps_the_callers_order(self):
        """`SaveAt` needs ascending ts; the permutation must be undone.

        Without inverting it the rows come back sorted, which is wrong for
        every caller and silently so -- the shape is unchanged.
        """
        b = _builder()
        taus = jnp.asarray([2.0, 0.3, 1.1])
        got = b.rotation_matrices(u.Q(taus, "s"))
        for i, t in enumerate(taus):
            one = b.rotation_matrix(u.Q(float(t), "s"))
            assert jnp.allclose(got[i], one, atol=1e-7)


class TestStraddlingIsRefused:
    """The sweep is monotonic outward from tau_0, so a mixed sign needs two."""

    def test_straddling_tau_0_raises(self):
        b = _builder()
        with pytest.raises(Exception, match="one side of tau_0"):
            b.rotation_matrices(u.Q(jnp.asarray([-1.0, 1.0]), "s"))

    def test_each_side_alone_is_fine(self):
        """The refusal is about mixing, not about sign."""
        b = _builder()
        assert b.rotation_matrices(u.Q(jnp.asarray([-1.0, -0.2]), "s")).shape == (
            2,
            3,
            3,
        )
        assert b.rotation_matrices(u.Q(jnp.asarray([1.0, 0.2]), "s")).shape == (2, 3, 3)


class TestFrameProperties:
    """The batched result is a frame, not merely close to one."""

    def test_rows_are_orthonormal(self):
        b = _builder()
        Rs = b.rotation_matrices(u.Q(jnp.linspace(0.2, 2.0, 6), "s"))
        eye = jnp.eye(3)
        for R in Rs:
            assert jnp.allclose(R @ R.T, eye, atol=1e-6)

    def test_it_is_a_rotation_not_a_reflection(self):
        b = _builder()
        Rs = b.rotation_matrices(u.Q(jnp.linspace(0.2, 2.0, 6), "s"))
        assert jnp.allclose(jnp.linalg.det(Rs), 1.0, atol=1e-6)


def test_it_survives_jit():
    """The guard uses `error_if` so it traces; the solve must too."""
    b = _builder()
    taus = jnp.linspace(0.1, 1.5, 5)
    f = jax.jit(lambda ts: b.rotation_matrices(u.Q(ts, "s")))
    assert jnp.allclose(f(taus), b.rotation_matrices(u.Q(taus, "s")), atol=1e-7)
