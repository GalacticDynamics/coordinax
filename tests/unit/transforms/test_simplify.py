"""Tests for ``simplify`` pairwise collapse and trace-safety (#539)."""

__all__: tuple[str, ...] = ()


import inspect

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
from .conftest import ZHAT


def _xyz(d):
    return np.array([float(d[k].ustrip("m")) for k in "xyz"])


def _point(x=1.0, y=2.0, z=3.0):
    return {"x": u.Q(x, "m"), "y": u.Q(y, "m"), "z": u.Q(z, "m")}


def _acts_equal(a, b):
    """Two operators act identically on a representative point."""
    p = _point()
    ra = cxfm.act(a, None, p, cxc.cart3d, cxr.point)
    rb = cxfm.act(b, None, p, cxc.cart3d, cxr.point)
    np.testing.assert_allclose(_xyz(ra), _xyz(rb), atol=1e-6)


# ===================================================================
# Pairwise merges


def test_adjacent_rotations_merge_to_one() -> None:
    R1 = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
    R2 = cxfm.Rotate.from_euler("z", u.Q(60, "deg"))
    pipe = cxfm.Composed((R1, R2))
    out = cxfm.simplify(pipe)
    assert isinstance(out, cxfm.Rotate)
    # The single rotation acts exactly like the two-step pipe.
    _acts_equal(out, pipe)
    np.testing.assert_allclose(np.asarray(out.R), np.asarray((R1 @ R2).R))


def test_adjacent_translations_merge_to_one() -> None:
    T1 = cxfm.Translate.from_([1, 2, 3], "km")
    T2 = cxfm.Translate.from_([4, 5, 6], "km")
    pipe = cxfm.Composed((T1, T2))
    out = cxfm.simplify(pipe)
    assert isinstance(out, cxfm.Translate)
    _acts_equal(out, pipe)


def test_inverse_pair_cancels_to_identity() -> None:
    R = cxfm.Rotate.from_euler("z", u.Q(45, "deg"))
    assert cxfm.simplify(cxfm.Composed((R, R.inverse))) is cxfm.identity


def test_identity_strip_re_exposes_adjacency() -> None:
    R = cxfm.Rotate.from_euler("z", u.Q(45, "deg"))
    pipe = cxfm.Composed((R, cxfm.Identity(), R.inverse))
    assert cxfm.simplify(pipe) is cxfm.identity


def test_non_mergeable_pair_is_preserved() -> None:
    """A pair with no combining rule survives `simplify` untouched.

    `Rotate | Translate` used to be the example here, but that pair is static
    affine and now fuses into an `Affine` (#546). A fibre kick does not: it
    shifts the tangent, not the point, so there is no `A x + b` to fuse it into.
    """
    R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    kick = replace(cxfm.Translate.from_([1, 0, 0], "km"), semantic_kind=cxr.vel)
    out = cxfm.simplify(cxfm.Composed((R, kick)))
    assert isinstance(out, cxfm.Composed)
    assert len(out.transforms) == 2
    _acts_equal(out, cxfm.Composed((R, kick)))


def test_different_semantic_kind_translates_do_not_merge() -> None:
    disp = cxfm.Translate.from_([1, 0, 0], "km")
    vel = replace(disp, semantic_kind=cxr.vel)
    out = cxfm.simplify(cxfm.Composed((disp, vel)))
    # A displacement and a velocity-kick are different actions: not merged.
    assert isinstance(out, cxfm.Composed)
    assert len(out.transforms) == 2


def test_time_dependent_rotations_do_not_merge() -> None:
    """`simplify` leaves adjacent `TimeDep` transforms alone, but preserves the act.

    Merging them needs the pointwise ``|`` fallback, which is unsound for a
    fibre offset (see `test_simplify_preserves_time_dependent_fibre_offset`).
    """
    # Distinct rates, so a merge that dropped one operand would be caught.
    a = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(0.3, "rad/s"), axis=ZHAT))
    b = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(0.5, "rad/s"), axis=ZHAT))
    pipe = cxfm.Composed((a, b))

    out = cxfm.simplify(pipe)
    assert isinstance(out, cxfm.Composed)
    assert len(out.transforms) == 2

    tau = u.Q(0.7, "s")
    p = _point()
    np.testing.assert_allclose(_xyz(out(tau, p)), _xyz(b(tau, a(tau, p))), atol=1e-12)


def test_time_dependent_rotations_merge_pointwise_under_matmul() -> None:
    """An EXPLICIT ``a @ b`` still merges two `TimeDep` families, pointwise in tau."""
    a = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(0.3, "rad/s"), axis=ZHAT))
    b = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(0.5, "rad/s"), axis=ZHAT))

    out = a @ b
    assert isinstance(out, cxfm.TimeDep)

    # ...and the merged family acts as the sequential application at a sample tau.
    tau = u.Q(0.7, "s")
    p = _point()
    np.testing.assert_allclose(_xyz(out(tau, p)), _xyz(b(tau, a(tau, p))), atol=1e-12)


def test_simplify_preserves_time_dependent_fibre_offset() -> None:
    """`simplify` must not fold a time-dependent fibre offset into a `TimeDep`.

    Folding it would materialize a `Composed` holding an order-1 offset, which
    `add.py` rejects: a working pipeline would start raising.
    """
    rate = {k: u.Q(v, "km/s") for k, v in (("x", 0.3), ("y", 0.0), ("z", 0.0))}
    kick = cxfm.TimeDep.from_(
        lambda t: cxfm.Translate(
            {k: v * (t / u.Q(1.0, "s")) for k, v in rate.items()},
            chart=cxc.cart3d,
            semantic_kind=cxr.coord_vel.semantic_kind,
        )
    )
    rot = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(0.5, "rad/s"), axis=ZHAT))
    pipe = cxfm.Composed((kick, rot))

    tau = u.Q(2.0, "s")
    v = {"x": u.Q(1.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
    at = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(0.0, "km")}
    kw = {"at": at}

    before = cxfm.act(pipe, tau, v, cxc.cart3d, cxr.coord_vel, **kw)
    after = cxfm.act(cxfm.simplify(pipe), tau, v, cxc.cart3d, cxr.coord_vel, **kw)
    np.testing.assert_allclose(
        [float(after[k].ustrip("km/s")) for k in "xyz"],
        [float(before[k].ustrip("km/s")) for k in "xyz"],
        atol=1e-12,
    )


# ===================================================================
# Trace-safety: the approx flag


def test_approx_false_merges_but_skips_identity_collapse() -> None:
    R = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
    pipe = cxfm.Composed((R, R.inverse))
    # Structural merge happens (Rotate @ Rotate) but the value-inspecting
    # collapse to Identity is skipped.
    out = cxfm.simplify(pipe, approx=False)
    assert isinstance(out, cxfm.Rotate)
    # It is still numerically the identity rotation.
    np.testing.assert_allclose(np.asarray(out.R), np.eye(3), atol=1e-6)


def test_approx_false_works_under_jit() -> None:
    R = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
    pipe = cxfm.Composed((R, R.inverse))
    out = jax.jit(lambda op: cxfm.simplify(op, approx=False))(pipe)
    assert isinstance(out, cxfm.Rotate)


def test_default_simplify_is_not_jit_safe() -> None:
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(lambda op: cxfm.simplify(op))(cxfm.Rotate(jnp.eye(3)))


class TestLorentzBoostSimplify:
    """`simplify` dispatches per operator and has no generic fallback.

    So a missing rule is not a missed optimisation -- it is a crash. Every
    other transform had one; `LorentzBoost` did not, which took out any
    `Composed` containing a boost as well.
    """

    def test_boost_with_velocity_is_returned_unchanged(self):
        op = cxfm.LorentzBoost([0.6, 0.0, 0.0])
        assert isinstance(cxfm.simplify(op), cxfm.LorentzBoost)

    def test_zero_boost_collapses_to_identity(self):
        assert cxfm.simplify(cxfm.LorentzBoost([0.0, 0.0, 0.0])) is cxfm.identity

    def test_zero_boost_is_kept_when_not_approx(self):
        """The zero check inspects values, so `approx=False` must skip it."""
        op = cxfm.LorentzBoost([0.0, 0.0, 0.0])
        assert isinstance(cxfm.simplify(op, approx=False), cxfm.LorentzBoost)

    def test_composed_containing_a_boost_simplifies(self):
        """The regression: this raised `NotFoundLookupError`."""
        rot = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
        got = cxfm.simplify(rot | cxfm.LorentzBoost([0.6, 0.0, 0.0]))
        assert isinstance(got, cxfm.Composed)
        assert len(got.transforms) == 2

    def test_boost_is_preserved_through_simplification(self):
        """Simplifying must not quietly change what the boost does."""
        op = cxfm.LorentzBoost([0.6, 0.0, 0.0])
        assert jnp.allclose(cxfm.simplify(op).matrix, op.matrix)


class TestScaleMerge:
    """Adjacent `Scale`s fuse, as adjacent `Rotate`s already did."""

    def test_two_scalings_merge_into_one(self):
        s1 = cxfm.Scale.from_factors(jnp.asarray([2.0, 3.0, 4.0]))
        s2 = cxfm.Scale.from_factors(jnp.asarray([5.0, 7.0, 11.0]))
        got = cxfm.simplify(s1 | s2)
        assert isinstance(got, cxfm.Scale)
        assert jnp.allclose(jnp.diag(got.matrix), jnp.asarray([10.0, 21.0, 44.0]))

    def test_merge_matches_sequential_application(self):
        s1 = cxfm.Scale.from_factors(jnp.asarray([2.0, 3.0, 4.0]))
        s2 = cxfm.Scale.from_factors(jnp.asarray([5.0, 7.0, 11.0]))
        p = cx.Point.from_([1.0, 2.0, 3.0], "m")
        merged = cxfm.simplify(s1 | s2)(p)
        sequential = s2(s1(p))
        for k in "xyz":
            assert jnp.allclose(merged[k].value, sequential[k].value)

    def test_chain_collapses_to_a_single_op(self):
        s = [cxfm.Scale.from_factors(jnp.asarray([f, f, f])) for f in (2.0, 3.0, 5.0)]
        got = cxfm.simplify(s[0] | s[1] | s[2])
        assert isinstance(got, cxfm.Scale)
        assert jnp.allclose(jnp.diag(got.matrix), jnp.asarray([30.0, 30.0, 30.0]))


class TestLinearFusion:
    """Mixed linear operators fuse into `Linear`, carrying their group along.

    Same-type rules are more specific and still win, so `Rotate | Rotate`
    keeps returning a `Rotate` rather than widening to `Linear`.
    """

    R = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
    S = cxfm.Scale.from_factors(jnp.asarray([2.0, 3.0, 4.0]))
    RF = cxfm.Reflect.from_normal([0.0, 0.0, 1.0])

    def test_mixed_pair_fuses_to_linear(self):
        got = cxfm.simplify(self.R | self.S)
        assert isinstance(got, cxfm.Linear)

    def test_same_type_rules_still_win(self):
        """`Linear` is the fallback, not a replacement for the tighter types."""
        r2 = cxfm.Rotate.from_euler("x", u.Q(45, "deg"))
        assert isinstance(cxfm.simplify(self.R | r2), cxfm.Rotate)
        s2 = cxfm.Scale.from_factors(jnp.asarray([5.0, 7.0, 11.0]))
        assert isinstance(cxfm.simplify(self.S | s2), cxfm.Scale)

    def test_fused_matches_sequential_application(self):
        p = cx.Point.from_([1.0, 2.0, 3.0], "m")
        chain = self.R | self.S | self.RF
        fused = cxfm.simplify(chain)
        seq = self.RF(self.S(self.R(p)))
        for k in ("x", "y", "z"):
            assert jnp.allclose(fused(p)[k].value, seq[k].value, atol=1e-12)

    def test_chain_collapses_to_one_op(self):
        got = cxfm.simplify(self.R | self.S | self.RF)
        assert isinstance(got, cxfm.Linear)

    def test_group_is_the_least_common_supergroup(self):
        """A rotation with a reflection is still orthogonal, not merely affine."""
        got = cxfm.simplify(self.R | self.RF)
        names = {g.__name__ for g in got.groups()}
        assert "OrthogonalGroup" in names
        assert "AffineGroup" not in names

    def test_group_widens_only_as_far_as_needed(self):
        got = cxfm.simplify(self.R | self.S)
        assert {g.__name__ for g in got.groups()} == {
            "AffineGroup",
            "DiffeomorphismGroup",
        }

    def test_inverse_round_trips_and_keeps_the_group(self):
        p = cx.Point.from_([1.0, 2.0, 3.0], "m")
        fused = cxfm.simplify(self.R | self.S)
        back = fused.inverse(fused(p))
        for k, want in zip(("x", "y", "z"), (1.0, 2.0, 3.0), strict=True):
            assert jnp.allclose(back[k].value, want, atol=1e-12)
        assert fused.inverse.groups() == fused.groups()

    def test_mismatched_dimensions_do_not_merge(self):
        """A 3x3 beside a 4x4 has no product, so the pair stays separate."""
        got = cxfm.simplify(self.R | cxfm.LorentzBoost([0.6, 0.0, 0.0]))
        assert isinstance(got, cxfm.Composed)
        assert len(got.transforms) == 2

    def test_identity_matrix_collapses_to_identity(self):
        assert cxfm.simplify(cxfm.Linear(jnp.eye(3))) is cxfm.identity

    def test_identity_matrix_is_kept_when_not_approx(self):
        """The identity check inspects values, so `approx=False` must skip it."""
        got = cxfm.simplify(cxfm.Linear(jnp.eye(3)), approx=False)
        assert isinstance(got, cxfm.Linear)

    def test_from_array_builds_a_linear(self):
        got = cxfm.Linear.from_(jnp.eye(3) * 2.0)
        assert isinstance(got, cxfm.Linear)
        assert jnp.allclose(got.matrix, jnp.eye(3) * 2.0)


class TestScaleIsDiagonal:
    """`Scale` scales the axes; an off-diagonal matrix belongs to `Linear`.

    Before `Linear` existed there was nowhere else to put a general matrix, so
    `Scale` accepted one and its name over-promised.
    """

    def test_from_factors_is_accepted(self):
        op = cxfm.Scale.from_factors(jnp.asarray([2.0, 3.0, 4.0]))
        assert jnp.allclose(jnp.diagonal(op.matrix), jnp.asarray([2.0, 3.0, 4.0]))

    def test_an_already_diagonal_matrix_is_accepted(self):
        op = cxfm.Scale(jnp.diag(jnp.asarray([2.0, 1.0, 0.5])))
        assert jnp.allclose(jnp.diagonal(op.matrix), jnp.asarray([2.0, 1.0, 0.5]))

    def test_an_off_diagonal_matrix_is_refused(self):
        shear = jnp.asarray([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(Exception, match="diagonal"):
            _ = cxfm.Scale(shear).matrix

    def test_a_rotation_is_refused(self):
        """The case that motivated the narrowing: a rotation is not a scaling."""
        rot = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
        with pytest.raises(Exception, match="diagonal"):
            _ = cxfm.Scale(rot.matrix).matrix

    def test_inverse_is_reciprocals_and_round_trips(self):
        op = cxfm.Scale.from_factors(jnp.asarray([2.0, 4.0, 5.0]))
        assert jnp.allclose(
            jnp.diagonal(op.inverse.matrix), jnp.asarray([0.5, 0.25, 0.2])
        )
        p = cx.Point.from_([1.0, 2.0, 3.0], "m")
        back = op.inverse(op(p))
        for k, want in zip(("x", "y", "z"), (1.0, 2.0, 3.0), strict=True):
            assert jnp.allclose(back[k].value, want, atol=1e-12)

    def test_inverse_of_a_malformed_scale_reports_the_real_problem(self):
        """It used to surface a raw `jnp.linalg.inv` shape error (see #726)."""
        with pytest.raises(Exception, match="square"):
            _ = cxfm.Scale(jnp.asarray([2.0, 3.0, 4.0])).inverse

    def test_merged_scales_stay_diagonal(self):
        s1 = cxfm.Scale.from_factors(jnp.asarray([2.0, 3.0, 4.0]))
        s2 = cxfm.Scale.from_factors(jnp.asarray([5.0, 7.0, 11.0]))
        got = cxfm.simplify(s1 | s2)
        assert isinstance(got, cxfm.Scale)
        assert jnp.allclose(jnp.diagonal(got.matrix), jnp.asarray([10.0, 21.0, 44.0]))


class TestLinearValidatesItsGroup:
    """`Linear` refuses a `group` that is not a transform group."""

    def test_a_non_group_type_is_refused(self):
        """Silently dropped otherwise: `groups()` would report only the widest.

        `most_specific_group` filters to `AbstractTransformGroup` subclasses, so
        an unrelated type never raises -- it just vanishes, and the operator
        fuses as though it had claimed nothing.
        """

        class NotAGroup:
            pass

        with pytest.raises(TypeError, match="AbstractTransformGroup"):
            cxfm.Linear(jnp.eye(3), NotAGroup)

    def test_a_non_type_is_refused(self):
        with pytest.raises(TypeError, match="AbstractTransformGroup"):
            cxfm.Linear(jnp.eye(3), "AffineGroup")

    def test_a_real_group_is_accepted(self):
        op = cxfm.Linear(jnp.eye(3), cxfm.groups.OrthogonalGroup)
        assert cxfm.groups.OrthogonalGroup in op.groups()

    def test_the_guard_itself_rejects_it(self):
        """Exercise the `__init__` guard, not the runtime type-checker.

        The suite runs under `COORDINAX_ENABLE_RUNTIME_TYPECHECKING`, so
        jaxtyping rejects a bad `group` at the decorator boundary and the tests
        above never reach the body. Installations do not typecheck at runtime,
        which is exactly where the guard has to hold, so unwrap the decorator
        and call the real `__init__`.
        """

        class NotAGroup:
            pass

        raw_init = inspect.unwrap(cxfm.Linear.__init__)
        with pytest.raises(TypeError, match="must be an AbstractTransformGroup"):
            raw_init(object.__new__(cxfm.Linear), jnp.eye(3), NotAGroup)


class TestMalformedMatricesReportTheSharedMessage:
    """Every route to a malformed matrix reports the base's square-check message.

    `Scale.inverse` already did (#731). These are the routes that still reached
    a raw JAX error first: a non-*square* 2D matrix slipped past the diagonal
    check's `ndim` guard, and `simplify`/`_merge`/`Linear.inverse` read the raw
    field rather than the validated `matrix`.
    """

    NONSQUARE = (3, 2)

    def _nonsquare(self):
        return jnp.ones(self.NONSQUARE)

    @pytest.mark.parametrize("op_type", [cxfm.Scale, cxfm.Linear])
    def test_matrix_reports_square(self, op_type):
        with pytest.raises(Exception, match="square"):
            _ = op_type(self._nonsquare()).matrix

    @pytest.mark.parametrize("op_type", [cxfm.Scale, cxfm.Linear])
    def test_inverse_reports_square(self, op_type):
        with pytest.raises(Exception, match="square"):
            _ = op_type(self._nonsquare()).inverse.matrix

    @pytest.mark.parametrize("op_type", [cxfm.Scale, cxfm.Linear])
    def test_simplify_reports_square(self, op_type):
        with pytest.raises(Exception, match="square"):
            cxfm.simplify(op_type(self._nonsquare()))

    def test_a_malformed_scale_cannot_be_built_at_all(self):
        """Was a `_merge` test; a malformed `Scale` no longer survives `__init__`.

        `Scale` stores the diagonal, and extracting it consumes the checked
        value, so the square error now fires at construction. There is no
        malformed `Scale` left to reach `_merge` with -- which is the point.
        `Linear` still stores its matrix, so its own routes are covered above.
        """
        with pytest.raises(Exception, match="square"):
            cxfm.Scale(self._nonsquare())

    def test_a_square_non_diagonal_still_reports_diagonal(self):
        """The square check must not swallow the diagonal one."""
        shear = jnp.asarray([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(Exception, match="diagonal"):
            _ = cxfm.Scale(shear).matrix
