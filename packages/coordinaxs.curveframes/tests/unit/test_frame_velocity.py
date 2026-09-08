r"""The frame velocity comes from the curve, and only from the curve.

A time-dependent curve frame's connection is $\partial\gamma/\partial t$ at
fixed first argument. Nothing in the library selects the Eulerian or the
Lagrangian reading; the parametrisation the caller hands in already fixes it,
and these tests pin that the library never overrides it.
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc

S0 = 1.3
T0 = 1.0


def stretch_and_bend(
    sigma: u.AbstractQuantity, t: u.AbstractQuantity
) -> u.AbstractQuantity:
    """gamma(sigma, t) = (sigma (1 + t/2), t sigma^2 / 10, 0).

    `sigma` is a *material* label: it names the same piece of the curve on
    every slice. The curve both stretches and bends, so arc length is not a
    reparametrisation-invariant relabelling of it.
    """
    sv, tv = sigma.ustrip("km"), t.ustrip("s")
    x = sv * (1.0 + 0.5 * tv)
    y = 0.1 * tv * sv**2
    return u.Q(jnp.stack([x, y, jnp.zeros_like(x)]), "km")


#: d/dt of `stretch_and_bend` at fixed `sigma`, in km/s.
MATERIAL_VELOCITY = jnp.array([0.5 * S0, 0.1 * S0**2, 0.0])


def dt_at_fixed_label(curve, label: float, t: float) -> jax.Array:
    """d(curve)/dt holding the first argument fixed -- whatever it labels."""

    def f(tv: float) -> jax.Array:
        return curve(u.Q(label, "km"), u.Q(tv, "s")).ustrip("km")

    return jax.jacfwd(f)(t)


def test_plain_curve_is_material() -> None:
    """An unwrapped curve's first argument is whatever the caller made it."""
    got = dt_at_fixed_label(stretch_and_bend, S0, T0)
    assert jnp.allclose(got, MATERIAL_VELOCITY)


def test_arclength_is_eulerian() -> None:
    """Wrapping in `ArcLength` re-measures per slice, so the label advects.

    Asserted as the size of the departure rather than as `not allclose`: a NaN
    compares False against everything, so the negative form alone would pass
    on an all-NaN result -- the exact failure mode the guards in this package
    exist to prevent.
    """
    arc = cxfc.ArcLength(stretch_and_bend, "km")
    got = dt_at_fixed_label(arc, S0, T0)
    assert jnp.isfinite(got).all()
    advection = float(jnp.linalg.norm(got - MATERIAL_VELOCITY))
    assert advection == pytest.approx(0.6677, abs=1e-3)


def test_lagrangian_arclength_restores_the_material_velocity() -> None:
    """`LagrangianArcLength`'s label names a material point, so it does not."""
    lag = cxfc.LagrangianArcLength(stretch_and_bend, u.Q(0.0, "s"), "km")
    got = dt_at_fixed_label(lag, S0, T0)
    assert jnp.allclose(got, MATERIAL_VELOCITY, atol=1e-6)


# --------------------------------------------------------------------------
# `builder.velocity` reports that connection, on whichever section it is given.


def test_a_worldtube_reports_the_material_velocity() -> None:
    """A pinned station makes `tau` the time, so `velocity` differentiates it."""
    b = cxfc.BishopBuilder(stretch_and_bend, "km", station=u.Q(S0, "km"))
    got = b.velocity(u.Q(T0, "s"))
    assert str(u.unit_of(got)) == "km / s"
    assert jnp.allclose(got.ustrip("km/s"), MATERIAL_VELOCITY, atol=1e-5)


def test_an_attime_slice_reports_the_same_velocity() -> None:
    """The two sections are cuts of one object, so they agree where they cross."""
    b = cxfc.BishopBuilder(cxfc.AtTime(stretch_and_bend, u.Q(T0, "s")), "km")
    got = b.velocity(u.Q(S0, "km"))
    assert jnp.allclose(got.ustrip("km/s"), MATERIAL_VELOCITY, atol=1e-5)


def test_the_attime_slice_is_not_the_tangent() -> None:
    """The trap this accessor exists to avoid.

    On the `AtTime` branch `location` takes the *station*, so differentiating
    it with respect to its argument gives the tangent -- same shape, same
    units, entirely plausible, and wrong. Asserted against both values so an
    implementation written as "differentiate `location`" fails here rather
    than passing quietly.
    """
    b = cxfc.BishopBuilder(cxfc.AtTime(stretch_and_bend, u.Q(T0, "s")), "km")
    tangent = jax.jacfwd(lambda sv: b.location(u.Q(sv, "km")).ustrip("km"))(S0)

    assert not jnp.allclose(tangent, MATERIAL_VELOCITY, atol=1e-3)  # they differ
    got = b.velocity(u.Q(S0, "km")).ustrip("km/s")
    assert jnp.allclose(got, MATERIAL_VELOCITY, atol=1e-5)
    assert not jnp.allclose(got, tangent, atol=1e-3)


@pytest.mark.parametrize("bare", [True, False], ids=["bare", "quantity"])
def test_the_array_fastpath_reaches_velocity(bare: bool) -> None:
    """A bare station is wrapped by `_param`, exactly as `location` wraps it.

    Bare parameters plus a declared `tau_unit` are the array fastpath. Reading
    the `station` field directly instead of going through the funnel handed
    the curve a raw float, which died inside the curve's own `ustrip` with
    nothing to say why.
    """
    station = S0 if bare else u.Q(S0, "km")
    worldtube = cxfc.BishopBuilder(stretch_and_bend, "km", station=station)
    assert jnp.allclose(
        worldtube.velocity(u.Q(T0, "s")).ustrip("km/s"), MATERIAL_VELOCITY, atol=1e-5
    )

    tau = S0 if bare else u.Q(S0, "km")
    slice_ = cxfc.BishopBuilder(cxfc.AtTime(stretch_and_bend, u.Q(T0, "s")), "km")
    assert jnp.allclose(
        slice_.velocity(tau).ustrip("km/s"), MATERIAL_VELOCITY, atol=1e-5
    )


def test_a_one_argument_curve_has_a_frame_that_does_not_move() -> None:
    r"""No *second* argument, so there is no time to differentiate against.

    Parametrised by arc length rather than by anything time-like, because the
    distinction being tested is easy to blur: this curve has a perfectly good
    `d(gamma)/d(arc)`, and it is the **tangent**. The frame velocity is
    $\partial\gamma/\partial t$ at fixed label, and with no `t` in the
    signature there is nothing for it to be but zero.
    """

    def circle(arc: u.AbstractQuantity) -> u.AbstractQuantity:
        a = arc.ustrip("km")
        return u.Q(jnp.stack([jnp.cos(a), jnp.sin(a), jnp.zeros_like(a)]), "km")

    got = cxfc.BishopBuilder(circle, "km").velocity(u.Q(0.3, "km"))
    assert str(u.unit_of(got)) == "km / s"
    assert jnp.allclose(got.ustrip("km/s"), jnp.zeros(3))


def test_the_reading_follows_the_curve_not_the_accessor() -> None:
    """`velocity` reports; it never chooses Eulerian or Lagrangian.

    The same underlying curve, wrapped two ways, gives two different answers
    -- which is the whole point of #779.
    """
    eulerian = cxfc.BishopBuilder(
        cxfc.AtTime(cxfc.ArcLength(stretch_and_bend, "km"), u.Q(T0, "s")), "km"
    ).velocity(u.Q(S0, "km"))
    lagrangian = cxfc.BishopBuilder(
        cxfc.AtTime(stretch_and_bend, u.Q(T0, "s")), "km"
    ).velocity(u.Q(S0, "km"))

    assert jnp.allclose(lagrangian.ustrip("km/s"), MATERIAL_VELOCITY, atol=1e-5)
    assert not jnp.allclose(eulerian.ustrip("km/s"), MATERIAL_VELOCITY, atol=1e-3)


# --------------------------------------------------------------------------
# Transporting a velocity is `TimeDep`'s job, and it already does it.


def _ground_truth(op, x0: jnp.ndarray, v_lab: jnp.ndarray) -> jax.Array:
    """d/dt of the lab trajectory pushed through the frame map at that same t."""

    def mapped(tv: float) -> jax.Array:
        x = x0 + v_lab * (tv - T0)
        q = {k: u.Q(x[i], "km") for i, k in enumerate("xyz")}
        out = op(u.Q(tv, "s"), q)
        return jnp.stack([out[k].ustrip("km") for k in "xyz"])

    return jax.jacfwd(mapped)(T0)


@pytest.mark.parametrize("offset", [0.0, 0.5, 2.0], ids=["on-axis", "n=0.5", "n=2"])
def test_timedep_transports_a_velocity_correctly_off_axis(offset: float) -> None:
    r"""`TimeDep` wraps a curve-frame builder unmodified, and is right everywhere.

    This is why there is no hand-rolled `relative_velocity`. The closed form
    $R(v-\beta)$ is exact only *on* the axis; off it the neglected
    $\dot R\,\mathbf{n}$ term grows with the offset. A tubular chart exists to
    describe points off the axis, so the closed form would be wrong exactly
    where the chart is used. Differentiating the whole map is not.
    """
    import coordinax.transforms as cxfm

    b = cxfc.BishopBuilder(stretch_and_bend, "km", station=u.Q(S0, "km"))
    op = cxfm.TimeDep(b)

    beta = b.velocity(u.Q(T0, "s")).ustrip("km/s")
    rot = jnp.asarray(b.rotation_matrix(u.Q(T0, "s")))
    gamma = b.location(u.Q(T0, "s")).ustrip("km")

    v_lab = jnp.array([1.0, 0.0, 0.0])
    x0 = gamma + jnp.array([0.0, offset, 0.0])

    truth = _ground_truth(op, x0, v_lab)
    closed_form = rot @ (v_lab - beta)

    assert jnp.allclose(truth, _ground_truth(op, x0, v_lab))  # deterministic
    if offset == 0.0:
        assert jnp.allclose(truth, closed_form, atol=1e-4)
    else:
        # the closed form drifts, and further the further out you go
        assert not jnp.allclose(truth, closed_form, atol=1e-2)
