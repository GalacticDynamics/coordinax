"""Two-argument curves in the frame builders.

A builder is called with a *single* parameter. For a one-argument curve
that is the curve parameter; for a two-argument ``gamma(tau, t)`` it is
the time, and the station must then be pinned as a field. With neither
pinned, two unknowns face one slot, so construction raises.

With a station pinned, `builder(t)` is the frame of the time-t slice at
that station. A slice at fixed t is a one-argument curve -- what `AtTime`
produces -- so the existing machinery applies unchanged, and the
equivalence with the hand-built slice is the correctness claim these tests
turn on.

Before this, a two-argument curve constructed happily and then failed on
*every* call with a `TypeError` raised from inside the ODE solve, nowhere
near the construction that caused it.

The routing and the guard both live on `AbstractCurveFrameBuilder`, so
both concrete builders inherit them. Also pinned here are the two
one-argument idioms that must keep working: #712 found both misread when
arity was decided by counting parameters rather than checking whether the
second one is required.
"""

import functools as ft

import astropy.units as apyu
import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc
from coordinaxs.curveframes._src.arclength import _is_two_argument


def curve1(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """An ordinary one-argument curve."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


def curve_gyr(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """A curve genuinely parametrised in Gyr, in kpc -- the galactic case.

    It *converts* its argument rather than reading the magnitude, which is how
    every curve in this package's docs is written and what makes a dimensionally
    compatible mismatch harmless.
    """
    t = tau.ustrip("Gyr")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "kpc")


def curve2(s: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """A time-dependent curve: it bends and stretches with ``t``."""
    sv, tv = s.ustrip("km"), t.ustrip("s")
    x = sv * (1.0 + 0.5 * tv)
    y = 0.1 * tv * sv**2
    return u.Q(jnp.stack([x, y, jnp.zeros_like(x)]), "km")


def curve_with_knob(
    tau: u.AbstractQuantity, smoothing: float = 0.1
) -> u.AbstractQuantity:
    """One-argument, but carrying a tuning knob with a default."""
    del smoothing
    return curve1(tau)


@pytest.mark.parametrize(
    "builder", [cxfc.BishopBuilder, cxfc.FrenetSerretBuilder], ids=["bishop", "frenet"]
)
def test_two_argument_curve_without_a_station_is_rejected(builder) -> None:
    """Both builders inherit the guard from `AbstractCurveFrameBuilder`."""
    with pytest.raises(ValueError, match="must be pinned"):
        builder(curve2, "km")


def test_the_error_names_the_remedy() -> None:
    """A message that only says "no" costs the reader the fix."""
    with pytest.raises(ValueError, match="AtTime"):
        cxfc.BishopBuilder(curve2, "km")


@pytest.mark.parametrize(
    "builder", [cxfc.BishopBuilder, cxfc.FrenetSerretBuilder], ids=["bishop", "frenet"]
)
def test_a_pinned_station_makes_the_call_time_parameter_the_time(builder) -> None:
    """The whole correctness claim: routing equals the hand-built slice.

    `builder(t)` on a two-argument curve must give the frame of the time-t
    slice at the pinned station -- which is what
    ``builder(AtTime(curve, t), ...)`` builds directly. Fails if `_resolve`
    binds the wrong slot, or slices at the wrong time.
    """
    s0 = u.Q(1.3, "km")
    b = builder(curve2, "km", station=s0)
    # t = 0 is excluded deliberately: `curve2(s, 0)` is the straight line
    # ``(s, 0, 0)``, where Frenet--Serret is singular (zero curvature, so the
    # normal is undefined and the rotation is NaN). That is the apparatus's
    # own degeneracy, not a routing question, and `BishopBuilder` exists
    # precisely because it does not have it.
    for t_val in (0.5, 1.7, 2.3):
        t = u.Q(t_val, "s")
        manual = builder(cxfc.AtTime(curve2, t), "km", station=s0)
        assert jnp.allclose(
            b.location(t).ustrip("km"), manual.location(t).ustrip("km"), atol=1e-12
        ), t_val
        assert jnp.allclose(
            b.tangent(t).ustrip(""), manual.tangent(t).ustrip(""), atol=1e-12
        ), t_val
        assert jnp.allclose(b.rotation_matrix(t), manual.rotation_matrix(t), atol=1e-12)


def test_the_frame_actually_evolves_with_time() -> None:
    """Guards against `_resolve` being an expensive no-op.

    A routing bug that ignored `t` would still satisfy the equivalence test
    above if the *manual* side ignored it too. This pins that the frame
    genuinely differs between two times -- on a curve that bends, so the
    rotation moves and not merely the origin.
    """
    b = cxfc.BishopBuilder(curve2, "km", station=u.Q(1.3, "km"))
    l0 = b.location(u.Q(0.0, "s")).ustrip("km")
    l1 = b.location(u.Q(1.7, "s")).ustrip("km")
    t0 = b.tangent(u.Q(0.0, "s")).ustrip("")
    t1 = b.tangent(u.Q(1.7, "s")).ustrip("")
    assert float(jnp.linalg.norm(l1 - l0)) > 1.0, (l0, l1)
    assert float(jnp.linalg.norm(t1 - t0)) > 0.1, (t0, t1)


def test_gradients_flow_to_time_and_to_the_station() -> None:
    """Both slots stay differentiable: `t` at call time, the station as a leaf.

    Checked against the closed form for
    ``gamma(s, t) = (s(1 + t/2), t s^2/10, 0)``:
    ``d(y)/dt = s^2/10`` and ``d(y)/ds = t s / 5``.
    """
    s0_val, t_val = 1.3, 1.0
    b = cxfc.BishopBuilder(curve2, "km", station=u.Q(s0_val, "km"))

    d_dt = jax.grad(lambda tv: b.location(u.Q(tv, "s")).ustrip("km")[1])(t_val)
    assert jnp.allclose(d_dt, 0.1 * s0_val**2, atol=1e-10), d_dt

    def loc_of_station(sv: float) -> float:
        moved = cxfc.BishopBuilder(curve2, "km", station=u.Q(sv, "km"))
        return moved.location(u.Q(t_val, "s")).ustrip("km")[1]

    d_ds = jax.grad(loc_of_station)(s0_val)
    assert jnp.allclose(d_ds, 0.2 * t_val * s0_val, atol=1e-10), d_ds


def test_at_time_makes_a_two_argument_curve_usable() -> None:
    """The remedy the message names has to actually work."""
    frozen = cxfc.AtTime(curve2, u.Q(0.5, "s"))
    b = cxfc.BishopBuilder(frozen, "km")
    # gamma(s=1.3, t=0.5) = (1.3 * 1.25, 0.1 * 0.5 * 1.69, 0)
    got = b.location(u.Q(1.3, "km")).ustrip("km")
    assert jnp.allclose(got, jnp.array([1.625, 0.0845, 0.0]), atol=1e-8), got


def test_a_defaulted_second_parameter_is_still_one_argument() -> None:
    """``def curve(tau, smoothing=0.1)`` is a one-argument curve."""
    cxfc.BishopBuilder(curve_with_knob, "s")


def test_a_variadic_second_parameter_is_still_one_argument() -> None:
    """``*args``/``**kw`` have no default, but ``curve(tau)`` binds them empty.

    Checking only "the second parameter is required" read both as
    time-dependent, sending an ordinary one-argument curve down the
    two-argument path. ``**kw`` is the sharper case: it cannot be called as
    ``curve(tau, t)`` at all, so that reading was not merely pessimistic but
    unusable.
    """

    def with_args(tau: u.AbstractQuantity, *args: object) -> u.AbstractQuantity:
        del args
        return curve1(tau)

    def with_kwargs(tau: u.AbstractQuantity, **kw: object) -> u.AbstractQuantity:
        del kw
        return curve1(tau)

    cxfc.BishopBuilder(with_args, "s")
    cxfc.BishopBuilder(with_kwargs, "s")


def test_a_required_keyword_only_second_parameter_is_rejected() -> None:
    """Callable neither way, so it gets its own error rather than a reading.

    ``def curve(tau, *, resolution)`` cannot be reached positionally, so
    ``curve(tau, t)`` fails, and has no default, so ``curve(tau)`` fails too.
    Reading it as two-argument named `AtTime(curve, t)` as the remedy, which
    would not have worked.
    """

    def kw_only(tau: u.AbstractQuantity, *, resolution: float) -> u.AbstractQuantity:
        del resolution
        return curve1(tau)

    with pytest.raises(TypeError, match="keyword-only"):
        cxfc.BishopBuilder(kw_only, "s")


def test_a_partial_frozen_time_is_still_one_argument() -> None:
    """`ft.partial` leaves the bound parameter visible, with a default."""
    frozen = ft.partial(curve2, t=u.Q(0.5, "s"))
    cxfc.BishopBuilder(frozen, "km")


# --------------------------------------------------------------------------
# #748: a wrapper reports the arity of what it wraps, not of its own
# convenience signature.


def test_arclength_reports_the_arity_of_the_curve_it_wraps() -> None:
    """`ArcLength.__call__` defaults ``t``; that is about the wrapper, not the curve.

    Reading the signature concluded "one-argument" for a wrapper that
    genuinely needs a time, so a builder called ``arc(station)`` and
    `ArcLength` raised. Fails if `_is_two_argument` goes back to inspecting
    the wrapper's own signature.
    """
    assert _is_two_argument(cxfc.ArcLength(curve2, "km")) is True
    assert _is_two_argument(cxfc.ArcLength(curve1, "s")) is False
    # `AtTime` binds the time, so one-argument is the truth about it.
    assert _is_two_argument(cxfc.AtTime(curve2, u.Q(0.5, "s"))) is False


def test_the_eulerian_station_composes_into_a_builder() -> None:
    """The composition the docs promise: a frame at a fixed *arc length*.

    This raised `TypeError: ... must be called as arc(s, t)` before #748.
    """
    s0 = u.Q(1.3, "km")
    b = cxfc.BishopBuilder(cxfc.ArcLength(curve2, "km"), "km", station=s0)

    # at t = 0 the curve is the straight line (s, 0, 0), so arc length 1.3
    # lands exactly on x = 1.3.
    got = b.location(u.Q(0.0, "s")).ustrip("km")
    assert jnp.allclose(got, jnp.array([1.3, 0.0, 0.0]), atol=1e-8), got

    # and the frame evolves with t rather than being pinned at construction
    t0 = b.tangent(u.Q(0.0, "s")).ustrip("")
    t1 = b.tangent(u.Q(1.7, "s")).ustrip("")
    assert float(jnp.linalg.norm(t1 - t0)) > 0.1, (t0, t1)


def test_eulerian_holds_arc_length_where_lagrangian_follows_the_material_point() -> (
    None
):
    """The physical distinction, on a curve that genuinely stretches.

    An Eulerian station sits at a fixed arc length, so its distance from the
    origin barely moves; a Lagrangian one keeps its material label and is
    carried outwards. Fails if the two wrappers are conflated -- which a
    rigid motion could not detect.
    """
    s0 = u.Q(1.3, "km")
    eul = cxfc.BishopBuilder(cxfc.ArcLength(curve2, "km"), "km", station=s0)
    lag = cxfc.BishopBuilder(
        cxfc.LagrangianArcLength(curve2, u.Q(0.0, "s"), "km"), "km", station=s0
    )

    def radius(b, t_val: float) -> float:
        return float(jnp.linalg.norm(b.location(u.Q(t_val, "s")).ustrip("km")))

    # both start at the same place, since t0 = 0 is the reference slice
    assert abs(radius(eul, 0.0) - radius(lag, 0.0)) < 1e-8

    # then they separate: the Eulerian one stays put, the Lagrangian one does not
    assert abs(radius(eul, 1.7) - 1.3) < 0.01, radius(eul, 1.7)
    assert radius(lag, 1.7) > 2.0, radius(lag, 1.7)


# --------------------------------------------------------------------------
# #718: a curve that knows what it exposes is checked against `tau_unit` at
# construction, rather than failing unevenly later.


def test_tau_unit_must_match_the_dimension_the_curve_exposes() -> None:
    """Passing a time where the curve exposes a length.

    `tau_unit` is required now, so plain omission is a `TypeError` before any
    object exists. This covers the other half: a unit supplied and *wrong*,
    which no amount of required-ness catches.
    """
    arc = cxfc.ArcLength(curve1, "s")  # exposes arc length: a *length*
    with pytest.raises(ValueError, match="dimension length"):
        cxfc.BishopBuilder(arc, "s")
    with pytest.raises(ValueError, match="dimension length"):
        cxfc.FrenetSerretBuilder(arc, "s")
    cxfc.BishopBuilder(arc, "km")  # correct, and still fine


def test_the_wrong_unit_is_only_half_visible_when_unguarded() -> None:
    """Why this is checked at construction, and what the guard does not reach.

    With the wrong `tau_unit`, `location` returns *correct* positions -- it
    never consults the unit -- while the autodiff paths raise. Anyone who
    sanity-checks positions first would conclude it works.

    A plain function cannot advertise what it exposes, so a length-parametrised
    one still slips through when "s" is passed. That is the residual case the
    `_param_dimension` guard does not cover; the wrappers do advertise, which
    is where the mistake is easiest to make.
    """

    def by_length(tau: u.AbstractQuantity) -> u.AbstractQuantity:
        d = tau.ustrip("km")
        return u.Q(jnp.stack([d, jnp.zeros_like(d), jnp.zeros_like(d)]), "km")

    b = cxfc.BishopBuilder(by_length, "s")  # a time, but the curve wants a length
    s_val = u.Q(1.0, "km")

    got = b.location(s_val).ustrip("km")
    assert jnp.allclose(got, jnp.array([1.0, 0.0, 0.0])), got

    with pytest.raises(apyu.UnitConversionError, match="not convertible"):
        b.tangent(s_val)


def test_a_wrapper_forwards_the_dimension_it_wraps() -> None:
    """`AtTime(ArcLength(curve), t)` still exposes an arc length.

    Binding the time changes the arity, not the dimension of the remaining
    parameter. Reading `_param_dimension` off the *type* missed this, so the
    guard did not fire and `tangent` failed later with a conversion error --
    and this is the composition the docs recommend as the remedy for a
    two-argument curve, so it is the one most worth catching.
    """
    frozen = cxfc.AtTime(cxfc.ArcLength(curve2, "km"), u.Q(0.5, "s"))
    assert frozen._param_dimension == "length"
    with pytest.raises(ValueError, match="dimension length"):
        cxfc.BishopBuilder(frozen, "s")
    cxfc.BishopBuilder(frozen, "km")  # correct unit is unaffected

    # A wrapper over a curve that claims nothing must claim nothing itself.
    assert cxfc.AtTime(curve2, u.Q(0.5, "s"))._param_dimension is None


def test_tau_unit_is_inferred_from_the_parameter_when_undeclared() -> None:
    """An undeclared unit is read off the `Quantity` the builder is called with.

    A declared unit is a second, independent statement of a fact the parameter
    already carries, and two statements can disagree. Reading it off the
    parameter removes the second statement, so there is nothing to disagree.
    """
    tau = u.Q(2.0, "Gyr")
    for cls in (cxfc.BishopBuilder, cxfc.FrenetSerretBuilder):
        declared = cls(curve_gyr, "Gyr").rotation_matrix(tau)
        inferred = cls(curve_gyr).rotation_matrix(tau)
        assert jnp.allclose(declared, inferred), (cls.__name__, declared, inferred)


def test_a_dimensionally_compatible_mismatch_is_absorbed_by_a_converting_curve() -> (
    None
):
    """Declaring "s" for a Gyr curve is *not* wrong when the curve converts.

    `unxt.experimental.jacfwd` strips with ``ustrip(unit, arg)`` -- a
    conversion, not a reinterpretation -- so 2 Gyr declared as "s" reaches the
    curve as 6.3e16 s, the curve's own ``ustrip("Gyr")`` converts it back, and
    the derivative comes out in ``kpc/s`` instead of ``kpc/Gyr``: the same
    physical quantity, differently labelled.

    Recorded because the opposite was believed: that such a curve came back
    "off by 3.15e16 with no error anywhere". It does not, and the distinction
    decides what `tau_unit` is actually for -- see the next test for the case
    where the mismatch really does corrupt the answer.
    """
    tau = u.Q(2.0, "Gyr")
    right = cxfc.BishopBuilder(curve_gyr, "Gyr").rotation_matrix(tau)
    wrong = cxfc.BishopBuilder(curve_gyr, "s").rotation_matrix(tau)
    assert jnp.allclose(right, wrong)


def test_a_value_reading_curve_is_the_case_that_needs_the_unit() -> None:
    """A curve that reads ``.value`` trusts the declared unit, so a wrong one lies.

    This is the one place `tau_unit` is load-bearing: the curve never converts,
    so whatever magnitude the declaration produces is taken at face value.
    Inference cannot be wrong here, because it hands the curve back exactly the
    numbers the caller passed.
    """

    def by_value(tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.value  # assumes it was handed its own unit
        return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "kpc")

    tau = u.Q(2.0, "Gyr")
    truth = cxfc.BishopBuilder(by_value, "Gyr").tangent(tau)
    assert jnp.allclose(cxfc.BishopBuilder(by_value).tangent(tau).value, truth.value)

    # Silently wrong, and finite: `tangent` returns a plausible unit vector
    # that is simply not this curve's. The parallel-transport path is louder --
    # the ODE gives up on a curve made 3.15e16 times stiffer -- so `tangent`
    # is what pins the *silent* failure.
    assert not jnp.allclose(
        cxfc.BishopBuilder(by_value, "s").tangent(tau).value, truth.value
    )


def test_a_unitless_parameter_has_nothing_to_infer_from() -> None:
    """The other case that must declare: a raw parameter carries no unit.

    It fails with a message naming both ways out, rather than an
    `AttributeError` from inside the derivative.
    """
    for cls in (cxfc.BishopBuilder, cxfc.FrenetSerretBuilder):
        # Both accessors, because they resolve the unit on separate paths:
        # `FrenetSerretBuilder` reached `.astype(float)` first and reported the
        # accident rather than the cause.
        for meth in ("tangent", "rotation_matrix"):
            with pytest.raises(TypeError, match="call-time parameter"):
                getattr(cls(curve_gyr), meth)(2.0)


def test_a_unitless_station_blames_the_station_not_the_call() -> None:
    """A pinned `station` is what the builder evaluates at, so it is the culprit.

    The call-time parameter can be a perfectly good `Quantity` and the builder
    still has nothing to read, because it never looks at it. Advising "pass a
    `Quantity` parameter" here sends the reader to the one place that is
    already correct, so the message names the station and shows how to fix
    *that*.
    """
    b = cxfc.BishopBuilder(curve_gyr, station=jnp.asarray(0.3))
    with pytest.raises(TypeError, match="pinned `station`"):
        b.tangent(u.Q(1.0, "Gyr"))  # a Quantity, and it does not help

    # Declaring the unit is the other way out, and does work.
    pinned = cxfc.BishopBuilder(curve_gyr, "Gyr", station=jnp.asarray(0.3))
    assert jnp.allclose(
        pinned.tangent(u.Q(1.0, "Gyr")).value,
        cxfc.BishopBuilder(curve_gyr, station=u.Q(0.3, "Gyr"))
        .tangent(u.Q(1.0, "Gyr"))
        .value,
    )


def test_a_raw_array_parameter_works_when_the_unit_is_declared() -> None:
    """The array fastpath: bare arrays, given meaning by the declared unit.

    This is what `tau_unit` is *for* once inference covers the `Quantity`
    case. It is wrapped at `_param`, the single funnel every accessor takes
    its curve parameter from, so all of them accept a raw parameter and every
    one agrees with the `Quantity` call to the bit.

    Before this, a raw parameter died with `NotFoundLookupError` from inside
    `unxt`'s `ustrip` dispatch, several frames below the call -- declaring the
    unit did not help, so the fastpath was documented but unreachable.
    """
    raw, quantity = jnp.asarray(0.7), u.Q(0.7, "s")
    for cls in (cxfc.BishopBuilder, cxfc.FrenetSerretBuilder):
        b = cls(curve1, "s")
        for meth in ("location", "tangent", "rotation_matrix"):
            got, want = getattr(b, meth)(raw), getattr(b, meth)(quantity)
            got, want = getattr(got, "value", got), getattr(want, "value", want)
            assert jnp.allclose(jnp.asarray(got), jnp.asarray(want)), (cls, meth)

    # A raw `station` takes the same funnel, so it is covered by the same wrap.
    pinned = cxfc.BishopBuilder(curve1, "s", station=jnp.asarray(0.3))
    assert jnp.allclose(
        pinned.tangent(quantity).value,
        cxfc.BishopBuilder(curve1, "s", station=u.Q(0.3, "s")).tangent(quantity).value,
    )


def test_the_dimension_guard_still_fires_on_an_inferred_unit() -> None:
    """Nothing is given up by inferring: the #718 guard runs on it too.

    With no declaration there is nothing to check at construction, so the check
    moves to the first call that supplies a parameter -- still before any wrong
    number is returned.
    """
    arc = cxfc.ArcLength(curve1, "s")  # exposes a *length*
    b = cxfc.BishopBuilder(arc)
    with pytest.raises(ValueError, match="dimension length"):
        b.tangent(u.Q(1.0, "s"))
    b.tangent(u.Q(1.0, "km"))  # a length is what it wants, and needs no declaring


def test_declaring_the_unit_is_still_accepted() -> None:
    """The explicit form is unchanged -- inference is a default, not a removal."""
    cxfc.BishopBuilder(curve1, "s")
    cxfc.BishopBuilder(curve1, "Gyr")
    cxfc.FrenetSerretBuilder(curve1, "s")
