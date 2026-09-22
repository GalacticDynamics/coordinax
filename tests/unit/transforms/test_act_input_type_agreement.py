"""Cross-input-type agreement for ``act`` (#942, #948, #955, #972).

The `CDict` methods are the reference implementation of every operator: they
cover the full (representation, semantic kind) ladder. So the same data
written as a bare array, a `unxt.Quantity`, a `unxts.linalg.QuantityMatrix`,
a `coordinax.Point`, a `coordinax.Tangent` or a `coordinax.Coordinate` bundle
must give the same answer as the `CDict` spelling.

``UNSUPPORTED`` lists the cells that stay an error, pinned by
`test_unsupported_cells_raise`.
"""

__all__: tuple[str, ...] = ()

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u
import unxts.linalg as ul
from dataclassish import replace

import coordinax as cx
import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from coordinax.frames import frame_transition
from coordinax.transforms._src.actions.utils import act_array_via_cdict

USYS = u.unitsystem("km", "s", "kg", "rad")

POS = (1.0, 0.0, 0.0)  # km
VEL = (10.0, 20.0, 30.0)  # km/s

# (kind, values, unit, rep) — the two data kinds every spelling is built for.
KINDS = [("pos", POS, "km", cxr.point), ("vel", VEL, "km/s", cxr.coord_vel)]


def _ops():
    """The operators under test: {position Translate, fibre kick, Boost, Composed}."""
    from coordinaxs.astro import ICRS, Galactocentric

    translate = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
    return {
        # A position translate: shifts points, identity on every tangent order.
        "pos_translate": (translate, None),
        # A fibre kick: identity on points, shifts matching-order tangents.
        "vel_kick": (
            replace(
                cxfm.Translate.from_([1.0, 2.0, 3.0], "km/s"), semantic_kind=cxr.vel
            ),
            None,
        ),
        # An operator with NO typed ArrayLike fast path (#948).
        "boost": (
            cxfm.Boost(
                {"x": u.Q(1.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")},
                chart=cxc.cart3d,
            ),
            u.Q(2.0, "s"),
        ),
        # The common astro pipeline: contains a position Translate AND a
        # fibre-kick Translate (the solar motion) under a Rotate.
        "icrs_to_gc": (frame_transition(ICRS(), Galactocentric()), None),
        # The pure linear maps (#972): `x -> M x` on points, `v -> M v` on any
        # tangent order, so every spelling of either kind must be served.
        "rotate": (cx.Rotate.from_euler("z", u.Q(37.0, "deg")), None),
        "scale": (cxfm.Scale.from_factors([2.0, 3.0, 1.0]), None),
        "shear": (
            cxfm.Shear.from_([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0, 0, 1.0]]),
            None,
        ),
        "reflect": (cxfm.Reflect.from_normal([0.0, 0.0, 1.0]), None),
    }


OPS = _ops()

# Cells that are intentionally still an error. A bare array has no units, so
# it cannot distinguish a position from tangent data; operators that treat the
# two differently refuse it instead of silently picking one. (`icrs_to_gc`
# contains such a fibre kick.) That ambiguity is a design question, not a
# dispatch hole.
#
# Note both surviving cells are ``rep=point`` under a fibre kick: THAT is the
# ambiguous spelling, because a unitless array tagged `point` could equally be
# the kick's own tangent data. An explicit tangent ``rep`` says what the data
# is, so it is served -- including through the `Rotate` inside `icrs_to_gc`,
# whose array path used to refuse it (#972).
UNSUPPORTED = {("vel_kick", "pos", "array"), ("icrs_to_gc", "pos", "array")}


# ===================================================================
# Spelling the same data every way, and reading the answer back out


def _spellings(values, unit, rep):
    """Build every accepted input spelling of ``values`` in ``unit``."""
    arr = jnp.asarray(values)
    vals = list(values)
    out = {
        # A bare array needs the chart and rep spelled out: it carries neither.
        "array": (arr, (cxc.cart3d, rep)),
        "quantity": (u.Q(vals, unit), ()),
        "cdict": ({k: u.Q(v, unit) for k, v in zip("xyz", values, strict=True)}, ()),
        "qmatrix": (ul.QuantityMatrix(arr, unit=(u.unit(unit),) * 3), ()),
    }
    zero_vel = cx.Tangent.from_([0.0, 0.0, 0.0], "km/s", cxc.cart3d, cxr.coord_vel)
    if rep == cxr.point:
        point = cx.Point.from_(vals, unit)
        out["point"] = (point, ())
        out["coordinate"] = (cx.Coordinate(point=point, velocity=zero_vel), ())
    else:
        tangent = cx.Tangent.from_(vals, unit, cxc.cart3d, rep)
        out["tangent"] = (tangent, ())
        out["coordinate"] = (
            cx.Coordinate(point=cx.Point.from_(list(POS), "km"), velocity=tangent),
            (),
        )
    return out


def _extract(result, kind, unit):
    """Read the acted-on ``kind`` component out of any result type, in ``unit``."""
    if isinstance(result, cx.Coordinate):
        # Explicit, so a kind this helper cannot place fails loudly rather
        # than silently reading the velocity slot.
        result = {"pos": lambda r: r.point, "vel": lambda r: r["velocity"]}[kind](
            result
        )
    if isinstance(result, cx.Point | cx.Tangent):
        result = result.data
    if isinstance(result, dict):
        return np.asarray([float(u.ustrip(unit, result[k])) for k in "xyz"])
    if isinstance(result, ul.QuantityMatrix):
        return np.asarray(
            [
                float(u.ustrip(unit, u.Q(result.value[i], result.unit[i])))
                for i in range(3)
            ]
        )
    if isinstance(result, u.AbstractQuantity):
        return np.asarray(u.ustrip(unit, result), dtype=float)
    return np.asarray(result, dtype=float)  # bare array, already in USYS units


def _act(op_name, spelling, values, unit, rep, kind):
    """Apply ``op_name`` to one spelling and return the answer as a float array."""
    op, tau = OPS[op_name]
    x, extra = _spellings(values, unit, rep)[spelling]
    return _extract(cxfm.act(op, tau, x, *extra, usys=USYS), kind, unit)


SPELLINGS = ["array", "quantity", "qmatrix", "point", "tangent", "coordinate"]


# ===================================================================
# The matrix: {spelling} x {operator} x {position, velocity data}


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(("kind", "values", "unit", "rep"), KINDS, ids=["pos", "vel"])
def test_spelling_agrees_with_cdict(op_name, spelling, kind, values, unit, rep):
    """Every input spelling gives the `CDict` answer, or is skipped as unsupported."""
    if spelling not in _spellings(values, unit, rep):
        pytest.skip(f"{spelling!r} does not spell {kind} data")
    if (op_name, kind, spelling) in UNSUPPORTED:
        pytest.skip(f"{(op_name, kind, spelling)} is intentionally an error")

    reference = _act(op_name, "cdict", values, unit, rep, kind)
    got = _act(op_name, spelling, values, unit, rep, kind)
    np.testing.assert_allclose(got, reference, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(("op_name", "kind", "spelling"), sorted(UNSUPPORTED))
def test_unsupported_cells_raise(op_name, kind, spelling):
    """The unit-less-array ambiguity is refused loudly, not guessed at."""
    values, unit, rep = next((v, un, r) for (k, v, un, r) in KINDS if k == kind)
    with pytest.raises(TypeError, match=r"representation|ambiguous"):
        _act(op_name, spelling, values, unit, rep, kind)


# ===================================================================
# Absolute anchors, so the agreement test above cannot pass vacuously


def test_position_translate_is_identity_on_velocities():
    """A position `Translate` does not move velocities (#942)."""
    shift = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")
    got = cxfm.act(shift, None, u.Q([10.0, 20.0, 30.0], "km/s"))
    assert isinstance(got, u.AbstractQuantity)
    np.testing.assert_allclose(np.asarray(u.ustrip("km/s", got)), VEL)


def test_fibre_kick_moves_only_matching_order():
    """A ``semantic_kind=vel`` `Translate` is the identity on points."""
    kick = replace(cxfm.Translate.from_([1.0, 2.0, 3.0], "km/s"), semantic_kind=cxr.vel)
    pos = cxfm.act(kick, None, u.Q([1.0, 0.0, 0.0], "km"))
    np.testing.assert_allclose(np.asarray(u.ustrip("km", pos)), POS)
    vel = cxfm.act(kick, None, u.Q([10.0, 20.0, 30.0], "km/s"))
    np.testing.assert_allclose(np.asarray(u.ustrip("km/s", vel)), [11.0, 22.0, 33.0])


@pytest.mark.parametrize("unit", ["km/s", "km/s2"])
def test_icrs_to_galactocentric_accepts_a_bare_tangent_quantity(unit):
    """ICRS -> Galactocentric composes a position `Translate` (#942)."""
    op, _ = OPS["icrs_to_gc"]
    got = op(u.Q([10.0, 20.0, 30.0], unit))
    assert isinstance(got, u.AbstractQuantity)
    assert u.unit_of(got) == u.unit(unit)


def test_boost_acts_on_a_bare_array():
    """An operator with no typed array fast path still serves arrays (#948)."""
    boost = cxfm.Boost(
        {"x": u.Q(1.0, "kpc/Myr"), "y": u.Q(0.0, "kpc/Myr"), "z": u.Q(0.0, "kpc/Myr")},
        chart=cxc.cart3d,
    )
    got = cxfm.act(
        boost,
        u.Q(1.0, "Myr"),
        jnp.asarray([1.0, 0.0, 0.0]),
        usys=u.unitsystems.galactic,
    )
    np.testing.assert_allclose(np.asarray(got), [2.0, 0.0, 0.0])


def test_bare_array_without_usys_is_refused():
    """A bare array has no units, so ``usys`` must supply them."""
    boost = cxfm.Boost(
        {"x": u.Q(1.0, "kpc/Myr"), "y": u.Q(0.0, "kpc/Myr"), "z": u.Q(0.0, "kpc/Myr")},
        chart=cxc.cart3d,
    )
    with pytest.raises(TypeError, match="usys"):
        cxfm.act(boost, u.Q(1.0, "Myr"), jnp.asarray([1.0, 0.0, 0.0]))


def test_velocity_quantitymatrix_is_kicked():
    """A velocity `QuantityMatrix` infers a tangent rep, so a kick lands (#955)."""
    kick = replace(cxfm.Translate.from_([1.0, 2.0, 3.0], "km/s"), semantic_kind=cxr.vel)
    qm = ul.QuantityMatrix(jnp.asarray([10.0, 20.0, 30.0]), unit=("km/s",) * 3)
    got = cxfm.act(kick, None, qm, usys=USYS)
    assert isinstance(got, ul.QuantityMatrix)
    np.testing.assert_allclose(np.asarray(got.value), [11.0, 22.0, 33.0])


def test_heterogeneous_quantitymatrix_rep_is_an_error_not_a_guess():
    """Mixed component dimensions have no single role, so say so (#955)."""
    kick = replace(cxfm.Translate.from_([1.0, 2.0, 3.0], "km/s"), semantic_kind=cxr.vel)
    qm = ul.QuantityMatrix(jnp.asarray([1.0, 2.0, 3.0]), unit=("km", "km/s", "km"))
    with pytest.raises(ValueError, match="do not agree"):
        cxfm.act(kick, None, qm, usys=USYS)
    # ... but an explicit `rep` is still honoured.
    assert isinstance(
        cxfm.act(kick, None, qm, cxc.cart3d, cxr.point, usys=USYS), ul.QuantityMatrix
    )


class TestABareArrayIsRefusedInAPhysicalBasis:
    """A bare array's units come from the coordinate basis, so only it is read.

    In `sph3d` the angular components are angular speeds in the coordinate
    basis and speeds in a physical one; a bare array cannot say which.
    """

    _OP = cxfm.Translate.from_([1.0, 2.0, 3.0], "kpc")
    _AT: ClassVar = {
        "r": u.Q(2.0, "kpc"),
        "theta": u.Q(1.0, "rad"),
        "phi": u.Q(0.5, "rad"),
    }
    _X = jnp.asarray([0.1, 0.2, 0.3])
    _USYS = u.unitsystem("kpc", "Myr", "Msun", "rad")

    def test_the_coordinate_basis_still_acts(self) -> None:
        got = act_array_via_cdict(
            self._OP,
            None,
            self._X,
            cxc.sph3d,
            cxr.coord_vel,
            at=self._AT,
            usys=self._USYS,
        )
        assert np.asarray(got).shape == (3,)

    def test_a_physical_basis_is_refused(self) -> None:
        with pytest.raises(TypeError, match="PhysicalBasis"):
            act_array_via_cdict(
                self._OP,
                None,
                self._X,
                cxc.sph3d,
                cxr.phys_vel,
                at=self._AT,
                usys=self._USYS,
            )


class TestTheArrayFunnelNamesABadShape:
    """A wrong-shaped array is named in the caller's terms, not QuantityMatrix's.

    A 0-D array has no last axis at all, which is as wrong as a mismatched one.
    """

    _OP = cxfm.Translate.from_([1.0, 2.0, 3.0], "km")

    @pytest.mark.parametrize(
        ("x", "expected"),
        [(jnp.asarray(1.0), "no axes"), (jnp.asarray([1.0, 0.0]), "is 2")],
        ids=["scalar", "too-few"],
    )
    def test_a_bad_shape_is_named(self, x, expected: str) -> None:
        with pytest.raises(ValueError, match=expected):
            act_array_via_cdict(self._OP, None, x, cxc.cart3d, cxr.coord_vel, usys=USYS)

    def test_a_good_shape_still_acts(self) -> None:
        got = act_array_via_cdict(
            self._OP,
            None,
            jnp.asarray([1.0, 0.0, 0.0]),
            cxc.cart3d,
            cxr.coord_vel,
            usys=USYS,
        )
        assert np.asarray(got).shape == (3,)


def test_the_missing_usys_message_names_every_unit_carrying_input() -> None:
    """The suggestion list must match what the funnel actually accepts.

    `act_array_via_cdict` refuses a bare array without ``usys`` and offers the
    inputs that carry their own units. `QuantityMatrix` was missing from that
    list even though it goes through this same funnel -- a message that
    enumerates supported types rots as soon as one is added, so pin it.
    """
    rate = {k: u.Q(v, "kpc/Myr") for k, v in [("x", 1.0), ("y", 0.0), ("z", 0.0)]}
    boost = cxfm.Boost(rate, chart=cxc.cart3d)
    x = jnp.asarray([1.0, 0.0, 0.0])

    with pytest.raises(TypeError, match="requires 'usys'") as excinfo:
        cxfm.act(boost, u.Q(1.0, "Myr"), x)

    msg = str(excinfo.value)
    for name in ("Quantity", "QuantityMatrix", "component dict", "typed vector"):
        assert name in msg, f"{name!r} missing from the suggestion list"

    # ... and each named input really does work through the same funnel.
    qm = ul.QuantityMatrix(x, unit=ul.UnitsMatrix(("kpc", "kpc", "kpc")))
    got = cxfm.act(boost, u.Q(1.0, "Myr"), qm, cxc.cart3d)
    assert np.allclose(np.asarray(u.ustrip("kpc", got)), [2.0, 0.0, 0.0])


def test_the_physical_basis_message_names_every_unit_carrying_input() -> None:
    """The other suggestion list must match what the funnel accepts too.

    Sibling of the missing-`usys` guard: a bare array cannot be read in a
    non-coordinate basis, and the message offers the inputs that carry their
    own per-component units. A component dict is one of them and was missing.
    """
    rate = {k: u.Q(v, "kpc/Myr") for k, v in [("x", 1.0), ("y", 0.0), ("z", 0.0)]}
    boost = cxfm.Boost(rate, chart=cxc.cart3d)

    with pytest.raises(TypeError, match="cannot act on a bare array") as excinfo:
        cxfm.act(
            boost,
            u.Q(1.0, "Myr"),
            jnp.asarray([1.0, 2.0, 3.0]),
            cxc.sph3d,
            cxr.phys_vel,
            usys=u.unitsystems.galactic,
        )

    msg = str(excinfo.value)
    for name in ("Quantity", "QuantityMatrix", "component dict", "typed vector"):
        assert name in msg, f"{name!r} missing from the suggestion list"

    # ... and the component dict it now names really does act in that basis.
    # `Rotate`, not the `Boost` above: a boost's rate is a cart3d velocity and
    # does not convert into a spherical physical basis. The point here is that
    # a CDict is accepted where a bare array is not, which any linear op shows.
    rot = cxfm.Rotate.from_euler("z", u.Q(37.0, "deg"))
    v = {k: u.Q(val, "km/s") for k, val in [("r", 1.0), ("theta", 2.0), ("phi", 3.0)]}
    at = {"r": u.Q(1.0, "kpc"), "theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
    got = cxfm.act(rot, None, v, cxc.sph3d, cxr.phys_vel, at=at)
    assert set(got) == set(v)


# ===================================================================
# #972 -- a linear map acts on a tangent given as a raw array
#
# `Rotate`, `Scale`, `Shear` and `Reflect` share one `ArrayLike` ``act``
# (`AbstractLinearTransform`), which refused any non-point ``rep`` outright.
# Unlike a `Translate` -- where a position shift is the identity on a velocity,
# so refusing merely withheld a no-op -- a linear map genuinely acts on a
# tangent, by the same matrix, so the refusal withheld a real computation that
# the `CDict` path next door already performed.

LINEAR_OPS = ["rotate", "scale", "shear", "reflect"]


@pytest.mark.parametrize("op_name", LINEAR_OPS)
@pytest.mark.parametrize(
    ("rep", "unit", "kind"),
    [(cxr.coord_vel, "km/s", "vel"), (cxr.coord_acc, "km/s2", "acc")],
)
def test_linear_op_tangent_array_agrees_with_every_spelling(op_name, rep, unit, kind):
    """A tangent array through a linear map matches CDict/Tangent/Quantity (#972)."""
    op, tau = OPS[op_name]
    vals = [10.0, 20.0, 30.0]
    arr = jnp.asarray(vals)

    # Act once; indexing the result per component would re-run the operator.
    acted_cdict = cxfm.act(
        op,
        tau,
        {k: u.Q(v, unit) for k, v in zip("xyz", vals, strict=True)},
        cxc.cart3d,
        rep,
        usys=USYS,
    )
    reference = np.asarray([float(u.ustrip(unit, acted_cdict[k])) for k in "xyz"])
    # The answer is not the input: the matrix really is applied.
    assert not np.allclose(reference, vals)

    got_array = np.asarray(cxfm.act(op, tau, arr, cxc.cart3d, rep, usys=USYS))
    np.testing.assert_allclose(got_array, reference, rtol=1e-12, atol=1e-12)

    tangent = cxfm.act(
        op, tau, cx.Tangent.from_(vals, unit, cxc.cart3d, rep), usys=USYS
    )
    np.testing.assert_allclose(
        _extract(tangent, kind, unit), reference, rtol=1e-12, atol=1e-12
    )

    quantity = cxfm.act(op, tau, u.Q(vals, unit), cxc.cart3d, rep, usys=USYS)
    np.testing.assert_allclose(
        _extract(quantity, kind, unit), reference, rtol=1e-12, atol=1e-12
    )


def test_tangent_array_on_a_non_cartesian_chart_with_an_anchor():
    """The CDict ladder serves a curved chart too, given its ``at`` anchor (#972)."""
    # A rotation about x, so the spherical velocity components genuinely move
    # (a rotation about z is a phi shift, hence the identity on them).
    op = cx.Rotate.from_euler("x", u.Q(37.0, "deg"))
    at = {"r": u.Q(2.0, "km"), "theta": u.Q(0.7, "rad"), "phi": u.Q(0.3, "rad")}
    comps = ("r", "theta", "phi")
    units = ("km/s", "rad/s", "rad/s")
    vals = [1.0, 0.1, 0.2]
    v = {k: u.Q(x, un) for k, x, un in zip(comps, vals, units, strict=True)}

    reference = cxfm.act(op, None, v, cxc.sph3d, cxr.coord_vel, at=at, usys=USYS)
    ref = np.asarray(
        [float(u.ustrip(un, reference[k])) for k, un in zip(comps, units, strict=True)]
    )
    assert not np.allclose(ref, vals)  # not a vacuous identity

    got = cxfm.act(
        op, None, jnp.asarray(vals), cxc.sph3d, cxr.coord_vel, at=at, usys=USYS
    )
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("op_name", LINEAR_OPS)
def test_bare_array_without_a_rep_is_still_a_point(op_name):
    """No ``rep`` still means `point`: the fix does not re-read unitless data (#972)."""
    op, tau = OPS[op_name]
    arr = jnp.asarray([1.0, 2.0, 3.0])
    implicit = cxfm.act(op, tau, arr, usys=USYS)
    explicit = cxfm.act(op, tau, arr, cxc.cart3d, cxr.point, usys=USYS)
    np.testing.assert_array_equal(np.asarray(implicit), np.asarray(explicit))


def test_fibre_kick_ambiguity_survives_the_linear_fix():
    """A no-``rep`` array under a fibre kick is still refused, not guessed (#972).

    The companion to `test_unsupported_cells_raise`: the `Rotate` inside
    ICRS -> Galactocentric no longer refuses a tangent array, but the solar-motion
    fibre kick still cannot tell a unitless `point` array from its own tangent
    data, so the pipeline keeps saying so.
    """
    op, _ = OPS["icrs_to_gc"]
    with pytest.raises(TypeError, match=r"ambiguous|representation"):
        cxfm.act(op, None, jnp.asarray([1.0, 2.0, 3.0]), usys=USYS)


class TestTheCallerSuppliedChartIsHonoured:
    """A bare array is read in the chart the caller named, or refused (#977).

    The array path used to overwrite `chart` with `guess_chart(x)`, which
    always answers Cartesian for a bare array -- so the Cartesian check below
    it could never fire, and `act(op, tau, x, sph3d, point)` silently treated
    the data as Cartesian.
    """

    _OP = cxfm.Rotate.from_euler("z", u.Q(90.0, "deg"))
    _X = jnp.asarray([1.0, 0.0, 0.0])

    def test_a_curvilinear_chart_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires a Cartesian chart"):
            cxfm.act(self._OP, None, self._X, cxc.sph3d, cxr.point, usys=USYS)

    def test_a_cartesian_chart_still_acts(self) -> None:
        got = cxfm.act(self._OP, None, self._X, cxc.cart3d, cxr.point, usys=USYS)
        np.testing.assert_allclose(np.asarray(got), [0.0, 1.0, 0.0], atol=1e-12)

    def test_a_component_count_mismatch_is_named(self) -> None:
        with pytest.raises(ValueError, match="last axis of x is 2"):
            cxfm.act(
                self._OP,
                None,
                jnp.asarray([1.0, 0.0]),
                cxc.cart3d,
                cxr.point,
                usys=USYS,
            )

    def test_a_scalar_is_named_too_not_an_index_error(self) -> None:
        """A 0-D array has no last axis to index; asking is not indexing."""
        with pytest.raises(ValueError, match="no axes"):
            cxfm.act(self._OP, None, jnp.asarray(1.0), cxc.cart3d, cxr.point, usys=USYS)


class TestLorentzBoostSharesTheLinearArrayPath:
    """The 4-D operator on the same `AbstractLinearTransform` path (#977).

    The fix is on the shared base, so `LorentzBoost` inherits it; the 3-D
    operators above cannot show that the 4-D Minkowski case also works.
    """

    _OP = cxfm.LorentzBoost.from_(u.Q([0.1, 0.0, 0.0], ""))
    _CHART = cxc.minkowskict
    _X = jnp.asarray([1.0, 0.5, 0.0, 0.0])

    @staticmethod
    def _units(rep):
        """The per-component units the array path reads from `USYS`.

        Derived from `rep`, not hard-coded: a displacement is in `km` and a
        velocity in `km / s`, so a fixed unit would compare the two paths in
        different units -- and pass anyway, because a Lorentz boost acts on
        components linearly and is indifferent to which unit they carry.
        """
        return tuple(
            USYS[d] for d in rep.semantic_kind.coord_dimensions(cxc.minkowskict)
        )

    @pytest.mark.parametrize(
        "rep", [cxr.coord_disp, cxr.coord_vel], ids=["disp", "vel"]
    )
    def test_a_tangent_rep_array_matches_the_cdict_spelling(self, rep) -> None:
        units = self._units(rep)
        cdict = {
            k: u.Q(float(v), unit)
            for k, v, unit in zip(self._CHART.components, self._X, units, strict=True)
        }

        got = cxfm.act(self._OP, None, self._X, self._CHART, rep, usys=USYS)
        ref = cxfm.act(self._OP, None, cdict, self._CHART, rep, usys=USYS)

        np.testing.assert_allclose(
            np.asarray(got),
            [
                float(u.ustrip(unit, ref[k]))
                for k, unit in zip(self._CHART.components, units, strict=True)
            ],
            rtol=0,
            atol=1e-12,
        )

    def test_the_two_reps_are_not_trivially_the_same_comparison(self) -> None:
        """The units really do differ between the reps, so the test has teeth."""
        assert self._units(cxr.coord_disp) != self._units(cxr.coord_vel)
