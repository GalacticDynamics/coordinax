"""Tests for the LorentzBoost operator.

The defining property of a Lorentz transformation is that it preserves the
Minkowski form: ``Λᵀ η Λ = η``. Most tests here check that invariance (or a
consequence of it) rather than the matrix entries, so they stay meaningful if
the matrix is ever rewritten.
"""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from coordinax.transforms._src import groups

ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

#: A spread of boosts: axis-aligned, oblique, zero, and ultrarelativistic.
BETAS = [
    [0.0, 0.0, 0.0],
    [0.6, 0.0, 0.0],
    [0.0, 0.8, 0.0],
    [0.0, 0.0, -0.5],
    [0.3, -0.4, 0.5],
    [0.99, 0.0, 0.0],
]

ATOL = 1e-5


def _interval(four_vec):
    """Minkowski interval ``-ct² + x² + y² + z²`` of a packed 4-vector."""
    return jnp.einsum("i,ij,j->", four_vec, ETA, four_vec)


class TestDefiningInvariance:
    """`Λᵀ η Λ = η` — everything else follows from this."""

    @pytest.mark.parametrize("beta", BETAS)
    def test_preserves_the_minkowski_form(self, beta):
        lam = cxfm.LorentzBoost(beta)._raw_matrix
        assert jnp.allclose(lam.T @ ETA @ lam, ETA, atol=ATOL)

    @pytest.mark.parametrize("beta", BETAS)
    def test_is_proper_and_orthochronous(self, beta):
        """``det Λ = +1`` and ``Λ⁰⁰ ≥ 1``: no parity flip, no time reversal."""
        lam = cxfm.LorentzBoost(beta)._raw_matrix
        assert jnp.allclose(jnp.linalg.det(lam), 1.0, atol=ATOL)
        assert float(lam[0, 0]) >= 1.0 - ATOL

    @pytest.mark.parametrize("beta", BETAS)
    def test_interval_of_an_event_is_invariant(self, beta):
        """A boost preserves the interval of every event, of any causal type."""
        lam = cxfm.LorentzBoost(beta)._raw_matrix
        for ev in ([2.0, 1.0, 0.0, 0.0], [1.0, 3.0, -1.0, 2.0], [1.0, 1.0, 0.0, 0.0]):
            x = jnp.asarray(ev)
            assert jnp.allclose(_interval(lam @ x), _interval(x), atol=ATOL)

    @pytest.mark.parametrize("beta", BETAS)
    def test_light_cone_is_preserved(self, beta):
        """A null vector stays null — the invariance of the speed of light."""
        lam = cxfm.LorentzBoost(beta)._raw_matrix
        null = jnp.asarray([1.0, 1.0, 0.0, 0.0])
        assert jnp.allclose(_interval(lam @ null), 0.0, atol=ATOL)


class TestGroupStructure:
    """Boosts compose, invert, and sit in the right place in the taxonomy."""

    def test_zero_boost_is_the_identity(self):
        """``beta = 0`` gives ``I``, not ``nan`` from the ``0/0`` in the matrix."""
        lam = cxfm.LorentzBoost([0.0, 0.0, 0.0])._raw_matrix
        assert not bool(jnp.any(jnp.isnan(lam)))
        assert jnp.allclose(lam, jnp.eye(4), atol=ATOL)

    @pytest.mark.parametrize("beta", BETAS)
    def test_inverse_is_the_opposite_boost(self, beta):
        op = cxfm.LorentzBoost(beta)
        assert jnp.allclose(op.inverse.beta, -op.beta, atol=ATOL)
        round_trip = op.inverse._raw_matrix @ op._raw_matrix
        assert jnp.allclose(round_trip, jnp.eye(4), atol=ATOL)

    def test_neg_matches_inverse(self):
        op = cxfm.LorentzBoost([0.3, -0.4, 0.5])
        assert jnp.allclose((-op).beta, op.inverse.beta, atol=ATOL)

    @pytest.mark.parametrize(("phi1", "phi2"), [(0.3, 0.5), (-0.2, 0.7), (1.0, 1.5)])
    def test_collinear_boosts_add_rapidities(self, phi1, phi2):
        """The reason rapidity exists: collinear composition is addition.

        Velocities do *not* add (that is the relativistic velocity-addition
        formula); rapidities do, so this is a sharp check on the matrix.
        """
        m1 = cxfm.LorentzBoost.from_rapidity(phi1)._raw_matrix
        m2 = cxfm.LorentzBoost.from_rapidity(phi2)._raw_matrix
        combined = cxfm.LorentzBoost.from_rapidity(phi1 + phi2)._raw_matrix
        assert jnp.allclose(m2 @ m1, combined, atol=ATOL)

    def test_velocity_addition_is_not_naive(self):
        """Guard the physics: 0.6c then 0.6c is 0.882c, never 1.2c."""
        m = cxfm.LorentzBoost([0.6, 0.0, 0.0])._raw_matrix
        combined = m @ m
        # Recover beta from the composed matrix: beta = Λ⁰ⁱ / Λ⁰⁰.
        beta_combined = float(combined[0, 1] / combined[0, 0])
        assert beta_combined == pytest.approx(2 * 0.6 / (1 + 0.36), abs=ATOL)
        assert beta_combined < 1.0

    def test_group_membership(self):
        gs = cxfm.LorentzBoost([0.5, 0.0, 0.0]).groups()
        assert groups.ProperOrthochronousLorentzGroup in gs
        assert groups.DiffeomorphismGroup in gs


class TestDerivedQuantities:
    """`gamma`, `speed`, and `rapidity` against textbook values."""

    @pytest.mark.parametrize(
        ("beta", "gamma"), [(0.0, 1.0), (0.6, 1.25), (0.8, 5.0 / 3.0)]
    )
    def test_gamma_known_values(self, beta, gamma):
        op = cxfm.LorentzBoost([beta, 0.0, 0.0])
        assert float(op.gamma) == pytest.approx(gamma, abs=ATOL)

    def test_speed_is_the_magnitude_of_beta(self):
        op = cxfm.LorentzBoost([0.3, 0.4, 0.0])
        assert float(op.speed) == pytest.approx(0.5, abs=ATOL)

    def test_rapidity_round_trips_through_beta(self):
        op = cxfm.LorentzBoost.from_rapidity(0.75)
        assert float(op.rapidity) == pytest.approx(0.75, abs=ATOL)

    def test_from_rapidity_normalises_the_direction(self):
        """An unnormalised direction still gives ``|beta| = tanh(phi)``."""
        op = cxfm.LorentzBoost.from_rapidity(0.5, (3.0, 4.0, 0.0))
        assert float(op.speed) == pytest.approx(float(jnp.tanh(0.5)), abs=ATOL)

    def test_from_velocity_divides_by_c(self):
        op = cxfm.LorentzBoost.from_velocity(u.Q([149896229.0, 0.0, 0.0], "m/s"))
        assert float(op.speed) == pytest.approx(0.5, abs=1e-4)

    def test_from_velocity_accepts_other_speed_units(self):
        """Unit handling is delegated to unxt, so km/s must agree with m/s."""
        in_kms = cxfm.LorentzBoost.from_velocity(u.Q([149896.229, 0.0, 0.0], "km/s"))
        assert float(in_kms.speed) == pytest.approx(0.5, abs=1e-4)

    def test_superluminal_boost_is_rejected(self):
        with pytest.raises(Exception, match="subluminal"):
            _ = cxfm.LorentzBoost([1.5, 0.0, 0.0]).gamma


class TestPhysicalPredictions:
    """Time dilation and length contraction, as coordinate statements."""

    def test_time_dilation(self):
        """A clock at rest at the origin ticking ``ct=1`` lands at ``ct=gamma``."""
        op = cxfm.LorentzBoost([0.6, 0.0, 0.0])
        tick = jnp.asarray([1.0, 0.0, 0.0, 0.0])
        out = op._raw_matrix @ tick
        assert float(out[0]) == pytest.approx(1.25, abs=ATOL)

    def test_simultaneity_is_relative(self):
        """Two simultaneous, separated events stop being simultaneous."""
        op = cxfm.LorentzBoost([0.6, 0.0, 0.0])
        here = op._raw_matrix @ jnp.asarray([0.0, 0.0, 0.0, 0.0])
        there = op._raw_matrix @ jnp.asarray([0.0, 1.0, 0.0, 0.0])
        assert float(there[0] - here[0]) != pytest.approx(0.0, abs=1e-3)

    def test_reduces_to_the_galilean_boost_at_low_speed(self):
        """At ``beta << 1`` the point action is ``x -> x + v t``."""
        beta = 1e-6
        op = cxfm.LorentzBoost([beta, 0.0, 0.0])
        ct, x = 5.0, 2.0
        out = op._raw_matrix @ jnp.asarray([ct, x, 0.0, 0.0])
        assert float(out[1]) == pytest.approx(x + beta * ct, abs=1e-9)
        assert float(out[0]) == pytest.approx(ct + beta * x, abs=1e-9)


class TestAct:
    """`act` on the chart-native data forms."""

    OP = cxfm.LorentzBoost([0.6, 0.0, 0.0])

    def test_act_on_cdict(self):
        ev = {
            "ct": u.Q(1.0, "m"),
            "x": u.Q(1.0, "m"),
            "y": u.Q(0.0, "m"),
            "z": u.Q(0.0, "m"),
        }
        out = cxfm.act(self.OP, None, ev, cxc.minkowskict, cxr.point)
        # A null event stays null: ct and x scale together.
        assert float(out["ct"].ustrip("m")) == pytest.approx(2.0, abs=ATOL)
        assert float(out["x"].ustrip("m")) == pytest.approx(2.0, abs=ATOL)

    def test_act_preserves_the_interval_of_a_cdict_event(self):
        ev = {
            "ct": u.Q(3.0, "m"),
            "x": u.Q(1.0, "m"),
            "y": u.Q(-2.0, "m"),
            "z": u.Q(0.5, "m"),
        }
        out = cxfm.act(self.OP, None, ev, cxc.minkowskict, cxr.point)
        before = _interval(jnp.asarray([float(ev[k].ustrip("m")) for k in ev]))
        after = _interval(jnp.asarray([float(out[k].ustrip("m")) for k in ev]))
        assert jnp.allclose(after, before, atol=1e-4)

    def test_act_on_packed_quantity(self):
        q = u.Q([1.0, 1.0, 0.0, 0.0], "m")
        out = cxfm.act(self.OP, None, q, cxc.minkowskict, cxr.point)
        want = jnp.asarray([2.0, 2.0, 0.0, 0.0])
        assert jnp.allclose(out.ustrip("m"), want, atol=ATOL)

    def test_act_round_trips_through_the_inverse(self):
        ev = {
            "ct": u.Q(3.0, "m"),
            "x": u.Q(1.0, "m"),
            "y": u.Q(-2.0, "m"),
            "z": u.Q(0.5, "m"),
        }
        fwd = cxfm.act(self.OP, None, ev, cxc.minkowskict, cxr.point)
        back = cxfm.act(self.OP.inverse, None, fwd, cxc.minkowskict, cxr.point)
        for k, want in ev.items():
            assert float(back[k].ustrip("m")) == pytest.approx(
                float(want.ustrip("m")), abs=1e-4
            )


class TestJAX:
    """The operator is a pytree and survives jit/vmap."""

    def test_jit(self):
        @jax.jit
        def apply(beta, x):
            return cxfm.LorentzBoost(beta)._raw_matrix @ x

        out = apply(jnp.asarray([0.6, 0.0, 0.0]), jnp.asarray([1.0, 1.0, 0.0, 0.0]))
        assert jnp.allclose(out[:2], jnp.asarray([2.0, 2.0]), atol=ATOL)

    def test_vmap_over_betas(self):
        def gamma_of(b):
            return cxfm.LorentzBoost(b).gamma

        betas = jnp.asarray([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.8, 0.0, 0.0]])
        got = jax.vmap(gamma_of)(betas)
        assert jnp.allclose(got, jnp.asarray([1.0, 1.25, 5.0 / 3.0]), atol=ATOL)
