"""`act` on tangent data has no order ceiling any more (#536).

`act` used to assemble the jet it needs from an `at`/`at_vel` keyword ladder,
which reached jet slots 0 and 1 and no further. Order-3 data needs slot 2 as
well, so it hard-failed with "use act_jet with a full jet instead" -- the API
admitting the ladder did not scale. `at_jet` is the general form; `at_vel` has
since been dropped, leaving `at` as sugar for slot 0 alone.
"""

__all__: tuple[str, ...] = ()

import dataclasses

import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from coordinax.representations._src.semantics import (
    _TANGENT_TIME_ORDER_LADDER as _LADDER,
    AbstractTangentSemanticKind,
)


@pytest.fixture(scope="module", autouse=True)
def _jerk_kind():
    """Register an order-3 kind for this module only, then unregister it.

    Defining the class registers it in the module-level time-order ladder, and
    that ladder is global: other tests assert it holds exactly the three kinds
    the library ships. Leaving order 3 behind broke five of them, so the entry
    is removed again on the way out.
    """
    import jax.tree_util as jtu

    @jtu.register_static
    @dataclasses.dataclass(frozen=True, slots=True, repr=False)
    class Jerk(AbstractTangentSemanticKind):
        """An order-3 kind, defined purely to exercise the ladder."""

        order = 3

    global _JERK_REP  # noqa: PLW0603
    _JERK_REP = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, Jerk())
    yield
    _LADDER.pop(3, None)
    _JERK_REP = None


_JERK_REP = None

# Time-dependent, so the *prolongation* path runs rather than the frozen-tau
# pushforward -- the prolongation is what needed the lower slots.
_OP = cxfm.TimeDep.from_(
    lambda tau: cxfm.Translate(
        {"x": u.Q(1.0, "km/s") * tau, "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")},
        cxc.cart3d,
    )
)
_TAU = u.Q(1.0, "s")
_DATA = {k: u.Q(1.0, "km/s3") for k in ("x", "y", "z")}
_SLOTS = {
    0: {k: u.Q(0.0, "km") for k in ("x", "y", "z")},
    1: {k: u.Q(0.0, "km/s") for k in ("x", "y", "z")},
    2: {k: u.Q(0.0, "km/s2") for k in ("x", "y", "z")},
}


def test_order_three_transforms_through_act() -> None:
    """The ceiling is gone: no `act_jet` detour required.

    The offset is linear in tau, so its third derivative is zero and the jerk
    comes back unchanged. That alone would also pass if the machinery quietly
    did nothing, which is what the cubic case below is for.
    """
    out = cxfm.act(_OP, _TAU, _DATA, cxc.cart3d, _JERK_REP, at_jet=_SLOTS)
    assert set(out) == {"x", "y", "z"}
    for k in ("x", "y", "z"):
        assert jnp.allclose(u.ustrip("km/s3", out[k]), 1.0, atol=1e-6)


def test_order_three_picks_up_the_third_derivative() -> None:
    r"""The value check: an order-3 slot gains $d^3\delta/d\tau^3$.

    With $\delta_x = \tau^3\,\mathrm{km/s^3}$ the third derivative is
    $6\,\mathrm{km/s^3}$, so `x` goes 1 -> 7 while the untouched axes stay at
    1. A path that ran but dropped the derivative would return 1 everywhere.
    """
    cubic = cxfm.TimeDep.from_(
        lambda tau: cxfm.Translate(
            {"x": u.Q(1.0, "km/s3") * tau**3, "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")},
            cxc.cart3d,
        )
    )
    out = cxfm.act(cubic, _TAU, _DATA, cxc.cart3d, _JERK_REP, at_jet=_SLOTS)
    assert jnp.allclose(u.ustrip("km/s3", out["x"]), 7.0, atol=1e-6)
    for k in ("y", "z"):
        assert jnp.allclose(u.ustrip("km/s3", out[k]), 1.0, atol=1e-6)


def test_a_missing_middle_slot_says_which_one() -> None:
    """Slots 0 and 1 alone do not reach order 3, and the error says so."""
    with pytest.raises(TypeError, match=r"slot\(s\) \[2\] are missing"):
        cxfm.act(
            _OP, _TAU, _DATA, cxc.cart3d, _JERK_REP, at_jet={0: _SLOTS[0], 1: _SLOTS[1]}
        )


def test_the_slot_zero_sugar_and_at_jet_agree() -> None:
    """`at=` is an alias for slot 0, so both spellings give the same answer."""
    acc = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.acc)
    data = {k: u.Q(1.0, "km/s2") for k in ("x", "y", "z")}
    via_sugar = cxfm.act(
        _OP, _TAU, data, cxc.cart3d, acc, at=_SLOTS[0], at_jet={1: _SLOTS[1]}
    )
    via_jet = cxfm.act(
        _OP, _TAU, data, cxc.cart3d, acc, at_jet={0: _SLOTS[0], 1: _SLOTS[1]}
    )
    for k in ("x", "y", "z"):
        assert u.ustrip("km/s2", via_sugar[k]) == u.ustrip("km/s2", via_jet[k])


def test_giving_a_slot_twice_is_refused() -> None:
    """Silently preferring one would be a wrong anchor, not a convenience."""
    with pytest.raises(TypeError, match="given twice"):
        cxfm.act(_OP, _TAU, _DATA, cxc.cart3d, _JERK_REP, at=_SLOTS[0], at_jet=_SLOTS)


def test_the_fibre_ladder_path_takes_at_jet_too() -> None:
    """`at_jet` is the general anchor form, so every path must accept it.

    A `TimeDep` holding a fibre offset routes through the ladder rule rather
    than the generic prolongation, and that rule read only `at=`. A caller
    passing the base point as slot 0 of `at_jet` hit a downstream "pass 'at'".
    """
    op = cxfm.TimeDep.from_(
        lambda tau: cxfm.Translate(
            {
                "x": u.Q(1.0, "km/s") * (tau / u.Q(1.0, "s")),
                "y": u.Q(0.0, "km/s"),
                "z": u.Q(0.0, "km/s"),
            },
            cxc.cart3d,
            semantic_kind=cxr.vel,
        )
    )
    rep = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.vel)
    data = {k: u.Q(1.0, "km/s") for k in ("x", "y", "z")}
    at = {k: u.Q(1.0, "km") for k in ("x", "y", "z")}

    via_at = cxfm.act(op, _TAU, data, cxc.cart3d, rep, at=at)
    via_jet = cxfm.act(op, _TAU, data, cxc.cart3d, rep, at_jet={0: at})
    for k in ("x", "y", "z"):
        assert jnp.allclose(u.ustrip("km/s", via_at[k]), u.ustrip("km/s", via_jet[k]))
    # and the ladder actually did something: 1 km/s + the offset's 1 km/s
    assert jnp.allclose(u.ustrip("km/s", via_jet["x"]), 2.0, atol=1e-6)


def test_the_missing_slot_errors_name_at_jet() -> None:
    """Both spellings are offered, since both now work.

    The messages previously named only the `at`/`at_vel` ladder. Asserting on
    the text is the point here: I once "fixed" this by adding the new
    constants and left the raise sites pointing at the old ones, which a
    constants-only check would not have caught.
    """
    acc = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.acc)
    data = {k: u.Q(1.0, "km/s2") for k in ("x", "y", "z")}

    with pytest.raises(TypeError, match=r"'at'.*or as slot 0 of 'at_jet'"):
        cxfm.act(_OP, _TAU, data, cxc.cart3d, acc, at_jet={1: _SLOTS[1]})

    with pytest.raises(TypeError, match=r"slot\(s\) \[1\] are missing"):
        cxfm.act(_OP, _TAU, data, cxc.cart3d, acc, at=_SLOTS[0])


def test_the_ladder_path_refuses_a_doubly_given_base_point() -> None:
    """Matching the generic path: two spellings of one slot is an error.

    Silently preferring `at=` would hide a wrong anchor.
    """
    op = cxfm.TimeDep.from_(
        lambda tau: cxfm.Translate(
            {
                "x": u.Q(1.0, "km/s") * (tau / u.Q(1.0, "s")),
                "y": u.Q(0.0, "km/s"),
                "z": u.Q(0.0, "km/s"),
            },
            cxc.cart3d,
            semantic_kind=cxr.vel,
        )
    )
    rep = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.vel)
    data = {k: u.Q(1.0, "km/s") for k in ("x", "y", "z")}
    at = {k: u.Q(1.0, "km") for k in ("x", "y", "z")}

    with pytest.raises(TypeError, match="given twice"):
        cxfm.act(op, _TAU, data, cxc.cart3d, rep, at=at, at_jet={0: at})
