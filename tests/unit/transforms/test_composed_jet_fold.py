"""`Composed`'s tangent `act` is the `act_jet` fold (#536, second half).

`Composed` used to hand-roll a shadow 2-jet: it threaded `at`/`at_vel` through
the per-sub-op `act` fold, advancing each anchor after every step (velocity
first, since it needs the old base point). That duplicated the `act_jet` fold
sitting in the same module, and only one of the two generalised -- the shadow
knew about exactly two anchors, so `at_jet` was passed through *unadvanced* and
came out with a different, wrong answer, and no error.
"""

__all__: tuple[str, ...] = ()

import dataclasses

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from coordinax.representations._src.semantics import (
    _TANGENT_TIME_ORDER_LADDER as _LADDER,
    AbstractTangentSemanticKind,
)

_ZHAT = jnp.asarray([0.0, 0.0, 1.0])
_TAU = u.Q(0.0, "s")
_ACC_REP = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.acc)

_AT = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
_VEL = {"x": u.Q(0.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
_ACC = {"x": u.Q(0.0, "m/s2"), "y": u.Q(0.0, "m/s2"), "z": u.Q(0.0, "m/s2")}


def _pipe() -> cxfm.Composed:
    r"""Translate 10 m along x, then spin at 1 rad/s about z.

    The translation is what makes the anchor advance observable: the rotation's
    centripetal term is $-\omega^2 r$, so it reports the base point it saw.
    An anchor left at the pipeline's *input* point reports r = 1 m instead of
    the 11 m the rotation actually acts at.
    """
    return cxfm.Composed(
        (
            cxfm.Translate.from_([10.0, 0.0, 0.0], "m"),
            cxfm.TimeDep(
                cxfm.builders.RotationAboutAxis(u.Q(1.0, "rad/s"), axis=_ZHAT)
            ),
        )
    )


def _x(cdict: dict) -> float:
    return float(u.ustrip("m/s2", cdict["x"]))


def test_at_jet_matches_the_sugar_through_a_pipeline() -> None:
    """The regression: `at_jet` used to skip the anchor advance and give -1."""
    pipe = _pipe()
    via_sugar = cxfm.act(pipe, _TAU, _ACC, cxc.cart3d, _ACC_REP, at=_AT, at_vel=_VEL)
    via_jet = cxfm.act(pipe, _TAU, _ACC, cxc.cart3d, _ACC_REP, at_jet={0: _AT, 1: _VEL})
    assert _x(via_sugar) == pytest.approx(_x(via_jet))


def test_both_spellings_match_the_act_jet_fold() -> None:
    """`act_jet` on the full jet is the reference: -omega^2 * 11 m."""
    pipe = _pipe()
    reference = cxfm.act_jet(pipe, _TAU, {0: _AT, 1: _VEL, 2: _ACC}, cxc.cart3d)[2]
    assert _x(reference) == pytest.approx(-11.0)

    for kw in ({"at": _AT, "at_vel": _VEL}, {"at_jet": {0: _AT, 1: _VEL}}):
        out = cxfm.act(pipe, _TAU, _ACC, cxc.cart3d, _ACC_REP, **kw)
        assert _x(out) == pytest.approx(-11.0)


def test_a_time_independent_pipeline_takes_at_jet_slot_zero() -> None:
    """No prolongation here -- just the pushforward, anchored on `at_jet[0]`.

    A curved chart is what makes the base point matter: the Jacobian of the
    spherical->spherical map depends on where it is evaluated.
    """
    pipe = cxfm.Composed(
        (
            cxfm.Rotate.from_euler("z", u.Q(90, "deg")),
            cxfm.Rotate.from_euler("x", u.Q(30, "deg")),
        )
    )
    at = {"r": u.Q(2.0, "m"), "theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
    v = {"r": u.Q(1.0, "m/s"), "theta": u.Q(0.1, "rad/s"), "phi": u.Q(0.2, "rad/s")}
    rep = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.vel)

    via_at = cxfm.act(pipe, None, v, cxc.sph3d, rep, at=at)
    via_jet = cxfm.act(pipe, None, v, cxc.sph3d, rep, at_jet={0: at})
    for k in via_at:
        assert jnp.allclose(
            u.ustrip(u.unit_of(via_at[k]), via_at[k]),
            u.ustrip(u.unit_of(via_at[k]), via_jet[k]),
        )


def test_the_doubly_given_base_point_is_refused_through_a_pipeline() -> None:
    """`Composed` refuses the same way every other path does."""
    with pytest.raises(TypeError, match="given twice"):
        cxfm.act(
            _pipe(), _TAU, _ACC, cxc.cart3d, _ACC_REP, at=_AT, at_jet={0: _AT, 1: _VEL}
        )


def test_a_pipeline_reaches_order_three() -> None:
    """The order ceiling is gone through `Composed` too, not just a bare op.

    The kind is registered here and popped again: the time-order ladder is
    global and other tests assert it holds exactly the three shipped kinds.
    """
    import jax.tree_util as jtu

    @jtu.register_static
    @dataclasses.dataclass(frozen=True, slots=True, repr=False)
    class Snap(AbstractTangentSemanticKind):
        order = 3

    try:
        rep = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, Snap())
        pipe = _pipe()
        jerk = {"x": u.Q(1.0, "m/s3"), "y": u.Q(0.0, "m/s3"), "z": u.Q(0.0, "m/s3")}
        slots = {0: _AT, 1: _VEL, 2: _ACC}

        out = cxfm.act(pipe, _TAU, jerk, cxc.cart3d, rep, at_jet=slots)
        reference = cxfm.act_jet(pipe, _TAU, {**slots, 3: jerk}, cxc.cart3d)[3]
        for k in ("x", "y", "z"):
            assert jnp.allclose(
                u.ustrip("m/s3", out[k]), u.ustrip("m/s3", reference[k]), atol=1e-6
            )
    finally:
        _LADDER.pop(3, None)


def test_the_coordinate_bundle_route_still_agrees() -> None:
    """The bundle path already folded jets; it must not have moved."""
    pipe = _pipe()
    pv = cx.Coordinate(
        point=cx.Point.from_([1.0, 0.0, 0.0], "m"),
        velocity=cx.Tangent.from_([0.0, 0.0, 0.0], "m/s"),
    )
    out = cxfm.act(pipe, _TAU, pv)
    lone = cxfm.act(
        pipe,
        _TAU,
        dict(pv["velocity"].data),
        cxc.cart3d,
        cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.vel),
        at_jet={0: dict(pv.point.data)},
    )
    for k in ("x", "y", "z"):
        assert jnp.allclose(
            u.ustrip("m/s", out["velocity"].data[k]),
            u.ustrip("m/s", lone[k]),
            atol=1e-6,
        )


def test_at_jet_slots_accept_point_and_tangent_bundles() -> None:
    """`at=` took a `Point`; `at_jet={0: Point}` has to as well.

    The public `Tangent` wrapper unwrapped (and chart-checked) `at`/`at_vel`
    only, so a `Point` in an `at_jet` slot reached the engine as a vector
    object and died on a component mismatch it could not explain.
    """
    op = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(1.0, "rad/s"), axis=_ZHAT))
    at = cx.Point.from_([1.0, 0.0, 0.0], "m")
    v = cx.Tangent.from_([0.0, 0.0, 0.0], "m/s")

    via_at = cx.act(op, _TAU, v, at=at)
    via_jet = cx.act(op, _TAU, v, at_jet={0: at})
    for k in ("x", "y", "z"):
        assert jnp.allclose(
            u.ustrip("m/s", via_at.data[k]), u.ustrip("m/s", via_jet.data[k])
        )
    # and the anchor was actually read: the omega x r term is 1 m/s along y
    assert float(u.ustrip("m/s", via_jet.data["y"])) == pytest.approx(1.0)


def test_a_mismatched_chart_in_an_at_jet_slot_says_which_slot() -> None:
    """The chart check travels with the unwrap, and names the slot it failed on."""
    op = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(1.0, "rad/s"), axis=_ZHAT))
    v = cx.Tangent.from_([0.0, 0.0, 0.0], "m/s")
    wrong = cx.Point.from_(
        {"r": u.Q(1.0, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.2, "rad")},
        cxc.sph3d,
    )
    with pytest.raises(ValueError, match=r"at_jet\[0\]"):
        cx.act(op, _TAU, v, at_jet={0: wrong})


def test_a_none_slot_reads_as_not_given() -> None:
    """`at_jet={0: None}` is "not given", exactly as `at=None` is.

    The engine tests slot *presence by key*, so a kept `None` would sail past
    the check and fail further down about a slot that looks supplied. Dropping
    it instead yields the engine's own missing-slot message, which names the
    slot and offers both spellings.
    """
    op = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(1.0, "rad/s"), axis=_ZHAT))
    v = cx.Tangent.from_([0.0, 0.0, 0.0], "m/s")

    with pytest.raises(TypeError, match=r"'at'.*or as slot 0 of 'at_jet'"):
        cx.act(op, _TAU, v, at_jet={0: None})

    # and a `None` alongside a real slot does not shadow it
    at = cx.Point.from_([1.0, 0.0, 0.0], "m")
    out = cx.act(op, _TAU, v, at_jet={0: at, 1: None})
    assert float(u.ustrip("m/s", out.data["y"])) == pytest.approx(1.0)


# --- `None` anchor slots, at the engine layer -----------------------------
#
# `at_jet` is caller-supplied input, so a slot can come out empty. The drop
# lives in the engine's two validators rather than in any one entrypoint, and
# these go in below the `cx.Tangent` wrapper to pin that. The runtime
# typechecker is not the gate: beartype samples one entry per container, so
# `{0: q, 1: None}` is only caught on the draws that pick slot 1.


def test_a_none_slot_reads_as_not_given_on_the_prolongation_path() -> None:
    """`_slot_jet`: the missing-slot error names the slot that came in `None`."""
    op = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(1.0, "rad/s"), axis=_ZHAT))
    with pytest.raises(TypeError, match=r"'at_vel'.*or as slot 1 of 'at_jet'"):
        cxfm.act(op, _TAU, _ACC, cxc.cart3d, _ACC_REP, at_jet={0: _AT, 1: None})


def test_a_none_slot_reads_as_not_given_on_the_pushforward_path() -> None:
    """`_merge_slot0`: no prolongation, but slot 0 still has to read as absent.

    A curved chart is what makes it observable -- the Jacobian pushforward
    refuses without a base point rather than quietly using the origin.
    """
    op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    v = {"r": u.Q(1.0, "m/s"), "theta": u.Q(0.1, "rad/s"), "phi": u.Q(0.2, "rad/s")}
    rep = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.vel)
    with pytest.raises(TypeError, match="requires 'at'"):
        cxfm.act(op, None, v, cxc.sph3d, rep, at_jet={0: None})


def test_a_none_slot_reads_as_not_given_through_a_pipeline() -> None:
    """`Composed` routes to the same validators, so it inherits this."""
    with pytest.raises(TypeError, match=r"'at'.*or as slot 0 of 'at_jet'"):
        cxfm.act(_pipe(), _TAU, _ACC, cxc.cart3d, _ACC_REP, at_jet={0: None, 1: _VEL})


def test_a_none_slot_does_not_shadow_a_real_one() -> None:
    """Dropping the empty slots must not take the supplied ones with them."""
    op = cxfm.TimeDep(cxfm.builders.RotationAboutAxis(u.Q(1.0, "rad/s"), axis=_ZHAT))
    vel = {"x": u.Q(0.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
    rep = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.vel)

    out = cxfm.act(op, _TAU, vel, cxc.cart3d, rep, at_jet={0: _AT, 1: None})
    reference = cxfm.act(op, _TAU, vel, cxc.cart3d, rep, at=_AT)
    for k in ("x", "y", "z"):
        assert jnp.allclose(u.ustrip("m/s", out[k]), u.ustrip("m/s", reference[k]))
    # and the anchor was read: the omega x r term is 1 m/s along y
    assert float(u.ustrip("m/s", out["y"])) == pytest.approx(1.0)
