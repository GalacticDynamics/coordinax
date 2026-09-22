"""Cross-input-type agreement for ``act`` (#942, #948, #955).

The `CDict` methods are the reference implementation of every operator: they
cover the full (representation, semantic kind) ladder. So the same data
written as a bare array, a `unxt.Quantity`, a `unxts.linalg.QuantityMatrix`,
a `coordinax.Point`, a `coordinax.Tangent` or a `coordinax.Coordinate` bundle
must give the same answer as the `CDict` spelling. Three holes in the dispatch
table used to break that:

- #942: `Translate` refused a tangent-rep Quantity (and a tangent-rep array)
  outright, so a bare velocity Quantity could not pass through any operator
  containing a position `Translate` — including ICRS -> Galactocentric.
- #948: the generic arity-5 `ArrayLike` method re-dispatched to a 6-argument
  `act` that only ever exists for `CDict`, so an operator without its own
  typed array fast path (e.g. `Boost`) found no method at all.
- #955: a `QuantityMatrix` defaulted to ``rep=point`` even though it stores
  per-component units, making a fibre-only operator a silent no-op on it.

The intentionally-unsupported cells are listed in ``UNSUPPORTED`` and pinned
by `test_unsupported_cells_raise`: a bare array carries no units, so it cannot
say whether it is a position or tangent data, and that ambiguity is rejected
loudly rather than guessed at.
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
    }


OPS = _ops()

# Cells that are intentionally still an error. A bare array has no units, so
# it cannot distinguish a position from tangent data; operators that treat the
# two differently refuse it instead of silently picking one. (`icrs_to_gc`
# contains such a fibre kick, and its `Rotate` has its own point-only array
# fast path.) These are a separate design question, not a dispatch hole.
UNSUPPORTED = {
    ("vel_kick", "pos", "array"),
    ("icrs_to_gc", "pos", "array"),
    ("icrs_to_gc", "vel", "array"),
}


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
        result = result.point if kind == "pos" else result["velocity"]
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
    op = frame_transition(*_gc_frames())
    got = op(u.Q([10.0, 20.0, 30.0], unit))
    assert isinstance(got, u.AbstractQuantity)
    assert u.unit_of(got) == u.unit(unit)


def _gc_frames():
    from coordinaxs.astro import ICRS, Galactocentric

    return ICRS(), Galactocentric()


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
    """Units for a bare array come from the coordinate basis (#970 review).

    `rep.semantic_kind.coord_dimensions(chart)` describes the *coordinate*
    basis: in `sph3d` it reports angular speed for the angular components. A
    physical (orthonormal) basis has every velocity component a speed, so
    reading a bare array that way would silently mis-unit two of its three
    components. Both bases used to return the same numbers, which was the tell.
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
