"""`Affine` fuses a run of affine operators into one kernel (#546)."""

__all__: tuple[str, ...] = ()

from typing import ClassVar

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx
import coordinax.transforms as cxfm
from coordinax.transforms import groups

_P = cx.Point.from_([1.0, 0.5, -2.0], "km")


def _rot_z(deg):
    return cxfm.Rotate.from_euler("z", u.Q(deg, "deg"))


def _agree(chain, fused, atol=1e-6):
    a, b = chain(_P), fused(_P)
    return (
        max(abs(float(a[k].ustrip("km")) - float(b[k].ustrip("km"))) for k in "xyz")
        <= atol
    )


class TestAffineCollapse:
    """A maximal run of affine operators becomes a single `Affine`."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda: _rot_z(30) | cxfm.Translate.from_([1.0, 2.0, 3.0], "km"),
            lambda: cxfm.Translate.from_([1.0, 2.0, 3.0], "km") | _rot_z(30),
            lambda: (
                _rot_z(30)
                | cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
                | cxfm.Rotate.from_euler("x", u.Q(45, "deg"))
                | cxfm.Translate.from_([0.0, -1.0, 2.0], "km")
            ),
            lambda: (
                _rot_z(30)
                | cxfm.Scale.from_factors(jnp.asarray([2.0, 1.0, 0.5]))
                | cxfm.Reflect.from_normal([0.0, 0.0, 1.0])
                | cxfm.Translate.from_([1.0, 0.0, 0.0], "km")
            ),
        ],
        ids=["R|T", "T|R", "R|T|R|T", "R|S|F|T"],
    )
    def test_a_run_collapses_and_agrees(self, build):
        """Interleaved rotations defeat pairwise merging; this reaches them."""
        chain = build()
        fused = cxfm.simplify(chain)
        assert isinstance(fused, cxfm.Affine)
        assert _agree(chain, fused)

    def test_the_collapse_is_one_operator(self):
        chain = (
            _rot_z(30)
            | cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
            | cxfm.Rotate.from_euler("x", u.Q(45, "deg"))
            | cxfm.Translate.from_([0.0, -1.0, 2.0], "km")
        )
        assert len(chain.transforms) == 4
        assert not hasattr(cxfm.simplify(chain), "transforms")


class TestAffineGroupTracking:
    """The fused operator keeps the tightest group it can justify."""

    def test_rotation_with_translation_is_still_an_isometry(self):
        """Reporting merely `AffineGroup` would discard that it preserves distance."""
        fused = cxfm.simplify(_rot_z(30) | cxfm.Translate.from_([1.0, 2.0, 3.0], "km"))
        assert groups.EuclideanGroup in fused.groups()

    def test_a_scaling_widens_the_group(self):
        fused = cxfm.simplify(
            _rot_z(30) | cxfm.Scale.from_factors(jnp.asarray([2.0, 1.0, 0.5]))
        )
        assert groups.AffineGroup in fused.groups()
        assert groups.EuclideanGroup not in fused.groups()


class TestAffineRefusesNonAffine:
    """Membership is decided on the lattice, not on Python inheritance."""

    def test_a_lorentz_boost_does_not_fuse(self):
        """`issubclass` says a boost is affine; the lattice says otherwise.

        `LorentzGroup` subclasses `OrthogonalGroup` for code reuse but declares
        `PoincareGroup` as its supergroup. Gating on `issubclass` would fold a
        4x4 spacetime map into a chain of 3x3 spatial ones -- a shape error
        wearing an optimisation's clothes.
        """
        chain = (
            _rot_z(30)
            | cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
            | cxfm.LorentzBoost([0.6, 0.0, 0.0])
        )
        fused = cxfm.simplify(chain)
        assert isinstance(fused, cxfm.Composed)
        assert any(isinstance(o, cxfm.LorentzBoost) for o in fused.transforms)

    def test_the_predicate_disagrees_with_issubclass(self):
        """Pinned directly, since this is the whole basis of the gate."""
        assert not groups.is_subgroup(
            groups.ProperOrthochronousLorentzGroup, groups.AffineGroup
        )
        assert issubclass(groups.ProperOrthochronousLorentzGroup, groups.AffineGroup)
        assert groups.is_subgroup(groups.IdentityGroup, groups.AffineGroup)
        assert not issubclass(groups.IdentityGroup, groups.AffineGroup)


class TestAffineContracts:
    """Inverse, identity collapse, and the trace-safety contract."""

    def _fused(self):
        return cxfm.simplify(
            _rot_z(30)
            | cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
            | cxfm.Rotate.from_euler("x", u.Q(45, "deg"))
        )

    def test_inverse_round_trips(self):
        op = self._fused()
        back = op.inverse(op(_P))
        for k in "xyz":
            assert float(back[k].ustrip("km")) == pytest.approx(
                float(_P[k].ustrip("km")), abs=1e-5
            )

    def test_inverse_keeps_the_group(self):
        """A group is closed under inversion."""
        op = self._fused()
        assert op.inverse.groups() == op.groups()

    def test_fusion_is_value_free_so_it_survives_approx_false(self):
        """Composing `(A, b)` inspects no values, so it is jit-safe."""
        chain = _rot_z(30) | cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
        assert isinstance(cxfm.simplify(chain, approx=False), cxfm.Affine)

    def test_an_inverse_pair_still_cancels(self):
        R = _rot_z(30)
        assert isinstance(cxfm.simplify(R | R.inverse), cxfm.Identity)

    def test_a_fused_identity_collapses_under_approx(self):
        T = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
        assert isinstance(cxfm.simplify(T | T.inverse), cxfm.Identity)

    def test_a_tangent_ignores_the_offset(self):
        """`d(Ax + b) = A dx` -- the translation differentiates away."""
        op = self._fused()
        lin = cxfm.simplify(_rot_z(30) | cxfm.Rotate.from_euler("x", u.Q(45, "deg")))
        v = {k: u.Q(val, "km/s") for k, val in (("x", 1.0), ("y", 2.0), ("z", 0.5))}
        a = cxfm.pushforward(op, None, v, cx.cart3d, cx.representations.coord_vel)
        b = cxfm.pushforward(lin, None, v, cx.cart3d, cx.representations.coord_vel)
        for k in "xyz":
            assert float(a[k].ustrip("km/s")) == pytest.approx(
                float(b[k].ustrip("km/s")), abs=1e-5
            )


class TestAffinePushforwardOnANonFlatChart:
    """A tangent needs its base point once the chart Jacobian is not the identity.

    The offset differentiates away, which is true and was the whole reasoning
    for dropping `at`. It does not follow that the base point is unnecessary:
    on a non-flat chart the tangent goes through the chart Jacobian at `at`,
    and the inverse Jacobian on the way back is anchored at the *image* of the
    base point -- `A at + b`, not `A at`.
    """

    AT: ClassVar = {
        "r": u.Q(2.0, "m"),
        "theta": u.Q(0.9, "rad"),
        "phi": u.Q(0.4, "rad"),
    }
    V: ClassVar = {
        "r": u.Q(1.0, "m/s"),
        "theta": u.Q(0.1, "rad/s"),
        "phi": u.Q(0.2, "rad/s"),
    }
    UNITS: ClassVar = {"r": "m/s", "theta": "rad/s", "phi": "rad/s"}

    def _chain(self):
        return _rot_z(90) | cxfm.Translate.from_([1.0, 0.0, 0.0], "m")

    def test_it_agrees_with_the_chain_it_replaced(self):
        """The check that catches an anchor at `A at` instead of `A at + b`."""
        chain = self._chain()
        fused = cxfm.simplify(chain)
        assert isinstance(fused, cxfm.Affine)

        got = cxfm.pushforward(
            fused,
            None,
            self.V,
            cx.charts.sph3d,
            cx.representations.coord_vel,
            at=self.AT,
        )
        want = cxfm.pushforward(
            chain,
            None,
            self.V,
            cx.charts.sph3d,
            cx.representations.coord_vel,
            at=self.AT,
        )
        for k, unit in self.UNITS.items():
            assert jnp.allclose(
                u.ustrip(unit, got[k]), u.ustrip(unit, want[k]), atol=1e-6
            )

    def test_a_missing_base_point_is_refused(self):
        """Better than mapping the tangent as a point and failing on `rad/s`."""
        fused = cxfm.simplify(self._chain())
        with pytest.raises(TypeError, match="requires 'at'"):
            cxfm.pushforward(
                fused, None, self.V, cx.charts.sph3d, cx.representations.coord_vel
            )


class TestAffineDeclinesRatherThanCrashes:
    """A pair that cannot fold is `_merge`'s ordinary answer, not an error."""

    SHIFT: ClassVar = cxfm.Translate.from_([1.0, 0.0, 0.0], "m")

    @pytest.mark.parametrize("n", [4, 5])
    def test_a_linear_map_with_no_cartesian_chart_declines(self, n):
        """There is no n-D Cartesian chart to express the offset in.

        This used to raise `KeyError: 4` out of `simplify` while looking up the
        component names -- a crash where a refusal was meant.
        """
        big = cxfm.Linear(jnp.diag(jnp.arange(1.0, n + 1.0)))
        for chain in (big | self.SHIFT, self.SHIFT | big):
            assert isinstance(cxfm.simplify(chain), cxfm.Composed)

    def test_mismatched_dimensions_decline(self):
        """A 2-D map and a 3-D offset have no common `A x + b`."""
        flat = cxfm.Linear(jnp.diag(jnp.asarray([2.0, 3.0])))
        assert isinstance(cxfm.simplify(flat | self.SHIFT), cxfm.Composed)


class TestAffineOffsetsInDifferentUnits:
    """Two offsets need not share a unit; the second converts into the first."""

    def test_km_and_m_offsets_fold_correctly(self):
        """A real rotation between them, so the pair actually reaches `Affine`.

        With an identity rotation the chain collapses to two adjacent
        `Translate`s, which merge as a `Translate` and never exercise the
        unit-reconciling branch.
        """
        in_km = cxfm.Translate.from_([1.0, 0.0, 0.0], "km")
        in_m = cxfm.Translate.from_([500.0, 0.0, 0.0], "m")
        chain = in_km | _rot_z(90) | in_m
        fused = cxfm.simplify(chain)
        assert isinstance(fused, cxfm.Affine)
        assert _agree(chain, fused)


class TestWhatDoesNotFold:
    """Each refusal in `_affine_parts`, exercised directly.

    They are reached through `simplify` only in combination, so the astro chain
    covers them jointly. Pinning them one at a time says *which* guard rejects
    what -- a single joint test passes just as happily if two of the three
    guards vanish.
    """

    def test_a_boost_is_refused(self):
        """`Boost` acts as ``x + dv*tau``, so no constant `b` can hold it.

        It declares no ``semantic_kind``, which is what the guard reads. Its
        ``delta`` *is* a component dict like any `AbstractAdd`, so the shape of
        the offset does not distinguish it -- a detail worth pinning, since an
        earlier comment here claimed the opposite.
        """
        from coordinax.transforms._src.actions.affine import _affine_parts

        boost = cxfm.Boost(
            {k: u.Q(v, "km/s") for k, v in (("x", 1.0), ("y", 0.0), ("z", 0.0))},
            cx.cart3d,
        )
        assert isinstance(boost.delta, dict)
        assert getattr(boost, "semantic_kind", None) is None
        assert _affine_parts(boost) is None

    def test_a_fibre_only_translate_is_refused(self):
        """Ladder order >= 1 moves velocities, not points."""
        from coordinax.transforms._src.actions.affine import _affine_parts

        vel_shift = cxfm.Translate(
            {k: u.Q(v, "km/s") for k, v in (("x", 1.0), ("y", 0.0), ("z", 0.0))},
            chart=cx.cart3d,
            semantic_kind=cx.representations.vel,
        )
        assert _affine_parts(vel_shift) is None

    def test_an_operator_of_no_known_shape_is_refused(self):
        """The fallback: affine by group, but neither linear nor additive."""
        from coordinax.transforms._src.actions.affine import _affine_parts

        assert _affine_parts(cxfm.Identity()) is None


def test_an_identity_affine_collapses():
    """`A = I`, `b = 0` is the identity, and `simplify` should say so."""
    R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    T = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
    fused = cxfm.simplify(R | T)
    round_trip = cxfm.simplify(cxfm.Composed((fused, fused.inverse)))
    assert isinstance(round_trip, cxfm.Identity)
