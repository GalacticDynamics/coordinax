"""`Affine` collapses an interleaved affine chain into one kernel (#546)."""

__all__: tuple[str, ...] = ()

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u
from dataclassish import replace

import coordinax as cx
import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm

_ROT = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
_SHIFT = cxfm.Translate.from_([1.0, 0.0, 0.0], "m")
_SCALE = cxfm.Scale.from_factors(jnp.asarray([2.0, 2.0, 2.0]))
_FLIP = cxfm.Reflect.from_normal([1.0, 0.0, 0.0])


def _point(xyz=(1.0, 2.0, 3.0)):
    return cx.Point.from_(u.Q(jnp.asarray(xyz), "m"), cxc.cart3d)


def _xyz(p):
    return np.asarray([float(u.ustrip("m", p[k])) for k in ("x", "y", "z")])


CHAINS = {
    "linear-then-shift": _ROT | _SHIFT,
    "shift-then-linear": _SHIFT | _ROT,
    "interleaved": _ROT | _SHIFT | _ROT | _SHIFT,
    "mixed-types": _SHIFT | _ROT | _FLIP | _SHIFT | _SCALE,
}


class TestAffineCollapse:
    """A maximal run of static affine operators fuses into one `Affine`."""

    @pytest.mark.parametrize("chain", CHAINS.values(), ids=CHAINS.keys())
    def test_it_collapses_to_a_single_affine(self, chain):
        """Including chains no adjacent-pair rule can reach.

        In `Rotate | Translate | Rotate | Translate` no two `Rotate`s are
        neighbours, which is exactly why the Level 1 peephole leaves it alone.
        """
        assert isinstance(cxfm.simplify(chain), cxfm.Affine)

    @pytest.mark.parametrize("chain", CHAINS.values(), ids=CHAINS.keys())
    def test_the_fused_operator_agrees_with_the_chain(self, chain):
        """Behaviour-preserving: the whole point of a *simplification*."""
        p = _point()
        np.testing.assert_allclose(
            _xyz(cxfm.simplify(chain)(p)), _xyz(chain(p)), atol=1e-5
        )

    def test_the_offset_rides_through_the_matrix(self):
        """`T | R` is not `R` with the same offset -- order matters.

        Composing (I, b) then (A, 0) gives `A x + A b`, not `A x + b`. A fused
        operator that merely carried `b` across would agree with the chain only
        when `A` is the identity.
        """
        fused = cxfm.simplify(_SHIFT | _ROT)
        naive = cxfm.Affine(_ROT.matrix, _SHIFT.delta, cxc.cart3d)
        p = _point()
        np.testing.assert_allclose(_xyz(fused(p)), _xyz((_SHIFT | _ROT)(p)), atol=1e-5)
        assert not np.allclose(_xyz(fused(p)), _xyz(naive(p)), atol=1e-5)

    def test_inverse_round_trips(self):
        p = _point()
        fused = cxfm.simplify(CHAINS["interleaved"])
        np.testing.assert_allclose(_xyz(fused.inverse(fused(p))), _xyz(p), atol=1e-5)

    def test_it_reports_the_affine_group(self):
        fused = cxfm.simplify(CHAINS["interleaved"])
        assert cxfm.groups.AffineGroup in fused.groups()


class TestAffineIsTraceSafe:
    """`simplify` keeps the contract #539 established."""

    def test_it_still_fuses_without_approx(self):
        """The fusion is structural, so `approx=False` does not disable it."""
        assert isinstance(
            cxfm.simplify(CHAINS["interleaved"], approx=False), cxfm.Affine
        )

    def test_it_traces(self):
        fused = cxfm.simplify(CHAINS["interleaved"], approx=False)
        got = jax.jit(lambda q: fused(q))(_point())
        np.testing.assert_allclose(_xyz(got), _xyz(fused(_point())), atol=1e-5)


class TestAffineRefusesWhatIsNotStaticAffine:
    """Each of these would be a *wrong* fusion, so the pair must not merge."""

    def test_a_fibre_offset_is_refused(self):
        """`semantic_kind=vel` shifts the tangent, not the point."""
        kick = replace(_SHIFT, semantic_kind=cxr.vel)
        assert not isinstance(cxfm.simplify(_ROT | kick), cxfm.Affine)

    def test_a_left_adding_offset_is_refused(self):
        assert not isinstance(
            cxfm.simplify(_ROT | replace(_SHIFT, right_add=False)), cxfm.Affine
        )

    def test_a_non_cartesian_offset_is_refused(self):
        """`x + b` is componentwise only in Cartesian components."""
        sph = cxfm.Translate(
            {"r": u.Q(1.0, "m"), "theta": u.Q(0.0, "rad"), "phi": u.Q(0.0, "rad")},
            cxc.sph3d,
        )
        assert not isinstance(cxfm.simplify(_ROT | sph), cxfm.Affine)

    def test_a_time_dependent_boost_is_refused(self):
        """A `Boost`'s offset grows with tau, so no *static* pair holds it."""
        boost = cxfm.Boost(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
            cxc.cart3d,
        )
        assert not isinstance(cxfm.simplify(_ROT | boost), cxfm.Affine)


class TestLevelOneStillWins:
    """Same-type pairs keep their better answers; `Affine` is the fallback."""

    @pytest.mark.parametrize(
        ("chain", "want"),
        [
            (_ROT | _ROT, cxfm.Rotate),
            (_SCALE | _SCALE, cxfm.Scale),
            (_SHIFT | _SHIFT, cxfm.Translate),
        ],
        ids=["rotate", "scale", "translate"],
    )
    def test_same_type_pairs_do_not_become_affine(self, chain, want):
        assert isinstance(cxfm.simplify(chain), want)
