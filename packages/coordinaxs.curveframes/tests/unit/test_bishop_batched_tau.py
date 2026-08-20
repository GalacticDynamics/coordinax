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


class TestRoutingIsNotBypassed:
    """`rotation_matrix` routes through `_resolve`; batching must not skip it.

    Regression: it did. With a pinned ``station`` the per-tau accessor returns
    the *same* frame for every tau, while the batched one varied with tau --
    silently, and by 0.2 in the helix case below.
    """

    def test_a_pinned_station_gives_one_frame_for_every_tau(self):
        b = cxfc.BishopBuilder(helix, "s", station=u.Q(0.7, "s"))
        taus = jnp.asarray([0.5, 1.0, 1.9])
        got = b.rotation_matrices(u.Q(taus, "s"))
        assert jnp.allclose(got, _per_tau(b, taus), atol=1e-7)
        # ...and they are all the same frame, which is the point of a station.
        assert jnp.allclose(got[0], got[-1], atol=1e-12)

    def test_a_two_argument_curve_is_refused(self):
        """Each tau selects a different slice, so there is no shared solve."""

        def moving(tau, t):
            x, s = tau.ustrip("s"), t.ustrip("s")
            return u.Q(jnp.stack([jnp.cos(x + s), jnp.sin(x + s), 0.3 * x]), "km")

        b = cxfc.BishopBuilder(moving, "s", station=u.Q(0.4, "s"))
        with pytest.raises(ValueError, match="one-argument curve"):
            b.rotation_matrices(u.Q(jnp.asarray([0.5, 1.0]), "s"))

    def test_an_empty_batch_is_refused(self):
        with pytest.raises(ValueError, match="at least one tau"):
            _builder().rotation_matrices(u.Q(jnp.asarray([]), "s"))


class TestTheSolveStaysAtTau0WhenItShould:
    """All-at-tau_0 must not march the curve away to answer about tau_0."""

    def test_a_locally_defined_curve_is_not_evaluated_outside(self):
        """The span is zero here, and substituting a nonzero one would show.

        The curve refuses to be evaluated more than a whisker from tau_0, so a
        solve that swept s in [0, 1] over a fabricated unit span would raise.
        """
        import equinox as eqx

        def local_only(tau):
            t = tau.ustrip("s")
            t = eqx.error_if(t, jnp.abs(t) > 1e-3, "curve evaluated away from tau_0")
            return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")

        b = cxfc.BishopBuilder(local_only, "s")
        got = b.rotation_matrices(u.Q(jnp.asarray([0.0, 0.0]), "s"))
        assert got.shape == (2, 3, 3)
        assert jnp.allclose(got[0], b.rotation_matrix(u.Q(0.0, "s")), atol=1e-9)
