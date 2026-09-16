r"""`rate_of_strain` is the ADM $K_{ij}$ of a family of slices.

$K_{ij} = \tfrac{1}{2\alpha}(\partial_t\gamma_{ij} - (\mathcal{L}_\beta
\gamma)_{ij})$ reduces to $\tfrac12\partial_t\gamma_{ij}$ here: the lapse is 1
under absolute time, and the Lie-drag term vanishes because a chart coordinate
is what $\partial_t$ holds fixed, so the coordinate shift is zero. The shift
that does *not* vanish is the ambient one `velocity` reports; the two are
different objects and these pin that they behave differently.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coordinaxs.api.manifolds import metric_matrix

import unxt as u

import coordinaxs.curveframes as cxfc

S0, T0 = 1.3, 1.0
BOUNDS = (u.Q(0.0, "km"), u.Q(3.0, "km"))
#: Off the axis, where `gamma` -- and so `K` -- depends on each slice's
#: transport seed. Valid only relative to a carried gauge; see #870.
POINT = {"tau": u.Q(S0, "km"), "n1": u.Q(0.2, "km"), "n2": u.Q(0.1, "km")}

#: On the axis, where the answer is gauge-free for any seed.
ON_AXIS = {"tau": u.Q(S0, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}


def stretch_and_bend(
    sigma: u.AbstractQuantity, t: u.AbstractQuantity
) -> u.AbstractQuantity:
    """The #778 curve: stretches along itself and bends."""
    sv, tv = sigma.ustrip("km"), t.ustrip("s")
    return u.Q(
        jnp.stack([sv * (1 + 0.5 * tv), 0.1 * tv * sv**2, jnp.zeros_like(sv)]), "km"
    )


def _family(curve):
    """The slice family: the tube's chart at each time."""
    return lambda t: cxfc.TubularChart(
        cxfc.BishopBuilder(cxfc.AtTime(curve, t), "km"), tau_bounds=BOUNDS
    )


@pytest.fixture(scope="module")
def k_deforming() -> np.ndarray:
    """`K_ij` for the deforming curve -- asserted about from two directions.

    A `jacfwd` through a diffrax solve, and the same tensor both times, so a
    worker computes it once rather than per test.
    """
    return np.asarray(
        cxfc.rate_of_strain(
            _family(stretch_and_bend), POINT, u.Q(T0, "s"), assume_gauge_carried=True
        ).value
    )


def test_a_material_labelling_shows_the_tube_stretching(
    k_deforming: np.ndarray,
) -> None:
    """The curve genuinely deforms, so the rate of strain is non-zero."""
    assert k_deforming[0, 0] == pytest.approx(0.77937, abs=1e-4)


def test_arc_length_holds_the_metric_so_its_strain_is_near_zero() -> None:
    """The same curve, relabelled, deforms hardly at all.

    `ArcLength` keeps the parametrisation near unit-speed by construction, so
    `gamma_tau_tau` stays near 1 and its rate of change near 0. The value
    follows the labelling the caller chose -- the same choice `velocity`
    reports, and this never overrides it.
    """
    k = cxfc.rate_of_strain(
        _family(cxfc.ArcLength(stretch_and_bend, "km")),
        POINT,
        u.Q(T0, "s"),
        assume_gauge_carried=True,
    )
    assert np.asarray(k.value)[0, 0] == pytest.approx(-0.00548, abs=1e-4)


# `test_it_is_symmetric` stood here and was vacuous: `K = 1/2 d_t(J^T J)` is
# symmetric by construction for every builder, so the assertion could not fail
# however wrong `K` was. What it should have been checking is below -- a rigid
# motion is an isometry, so `K` must vanish. That is the property the n-plane
# gauge defect actually breaks, and the one a regression test needs (#870).


def _rotation(axis: list[float], theta):
    """Rodrigues, so the motion below is exactly rigid."""
    a = jnp.asarray(axis) / jnp.linalg.norm(jnp.asarray(axis))
    k = jnp.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return jnp.eye(3) + jnp.sin(theta) * k + (1 - jnp.cos(theta)) * (k @ k)


def _helix_family(axis: list[float]):
    """A helix carried rigidly about ``axis``, sliced at each time."""

    def curve(tau: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
        arc = tau.ustrip("km")
        base = jnp.stack([jnp.cos(arc), jnp.sin(arc), 0.4 * arc])
        return u.Q(_rotation(axis, t.ustrip("s")) @ base, "km")

    return lambda t: cxfc.TubularChart(
        cxfc.BishopBuilder(cxfc.AtTime(curve, t), "km"),
        tau_bounds=(u.Q(-1.0, "km"), u.Q(2.0, "km")),
    )


HELIX_OFF = {"tau": u.Q(0.5, "km"), "n1": u.Q(0.2, "km"), "n2": u.Q(0.1, "km")}
HELIX_ON = {"tau": u.Q(0.5, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}


@pytest.mark.parametrize("axis", [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], ids=["z", "y"])
def test_a_rigid_rotation_gives_no_strain_on_axis(axis: list[float]) -> None:
    """A rigid motion is an isometry, so nothing deforms.

    Two traps this avoids. A rigid *translation* proves nothing: `T0` does not
    move, so the world-anchored transport seed does not either. And the axis
    matters -- only rotations fixing `argmin |T0|` are equivariant, so `x-hat`
    passes even when the gauge is broken. Both axes here are generic.
    """
    k = cxfc.rate_of_strain(_helix_family(axis), HELIX_ON, u.Q(0.3, "s"))
    assert np.abs(np.asarray(k.value)).max() < 1e-5


def test_an_off_axis_rate_is_refused_as_gauge_dependent() -> None:
    """Off the axis the answer turns on a seed the caller never chose."""
    with pytest.raises(eqx.EquinoxRuntimeError, match="gauge-dependent"):
        cxfc.rate_of_strain(_helix_family([0.0, 0.0, 1.0]), HELIX_OFF, u.Q(0.3, "s"))


def test_the_gauge_guard_leaves_the_point_traceable() -> None:
    """The guard defers to runtime rather than reading the point in Python.

    A `float(n)` refuses every tracer, on-axis ones included, which would put
    the whole function out of reach of `jit` and `vmap` -- the chart itself
    traces fine. `vmap` also pins that the deferred check still fires, for the
    off-axis member alone.
    """
    fam, t = _helix_family([0.0, 0.0, 1.0]), u.Q(0.3, "s")
    k = lambda n1: cxfc.rate_of_strain(fam, {**HELIX_ON, "n1": u.Q(n1, "km")}, t).value

    on_axis = np.asarray(jax.jit(k)(0.0))
    assert np.abs(on_axis).max() < 1e-5
    assert np.allclose(np.asarray(jax.vmap(k)(jnp.zeros(2))), on_axis)

    with pytest.raises(eqx.EquinoxRuntimeError, match="gauge-dependent"):
        jax.vmap(k)(jnp.array([0.0, 0.2]))


@pytest.mark.xfail(
    strict=True, reason="#870: the n-plane gauge drifts and is reported as strain"
)
def test_a_rigid_rotation_gives_no_strain_off_axis() -> None:
    """The requirement the guard defers rather than meets.

    Opting in returns `|K|max = 0.017405` about `z-hat`, where an isometry
    demands zero. Strict, so that fixing #870 -- by carrying one director
    across the family -- turns this green and says so.
    """
    k = cxfc.rate_of_strain(
        _helix_family([0.0, 0.0, 1.0]),
        HELIX_OFF,
        u.Q(0.3, "s"),
        assume_gauge_carried=True,
    )
    assert np.abs(np.asarray(k.value)).max() < 1e-5


def test_only_the_longitudinal_component_can_be_nonzero(k_deforming) -> None:
    """`K` is longitudinal, whatever the name suggests (#870).

    `n1`, `n2` are Cartesian in an orthonormal normal plane, so
    `gamma_nn = delta_ij` at every offset, and `d_t` of a constant is zero. A
    tube doubling in radius reports the same `K` as one holding it, which the
    docstring now says out loud.
    """
    assert np.abs(k_deforming[0, 0]) > 0.1  # the one that carries the strain
    assert np.abs(k_deforming[1:, 1:]).max() < 1e-15
    assert np.abs(k_deforming[0, 1:]).max() < 1e-10


def test_the_normal_plane_metric_is_the_identity() -> None:
    """The reason the block above is zero, pinned at the source."""
    ch = _family(stretch_and_bend)(u.Q(T0, "s"))
    for n1, n2 in [(0.0, 0.0), (0.2, 0.1), (0.5, 0.4)]:
        at = {"tau": u.Q(S0, "km"), "n1": u.Q(n1, "km"), "n2": u.Q(n2, "km")}
        g = np.asarray(metric_matrix(ch.M, at, ch).matrix.ustrip(""))

        # One ulp, not exact: the block is built from the Jacobian rather
        # than read off symbolically. Measured 2.22e-16 at all three offsets.
        assert np.abs(g[1:, 1:] - np.eye(2)).max() < 1e-15


def test_a_static_curve_does_not_deform() -> None:
    """No time in the curve, so the slice geometry does not change."""

    def circle(arc: u.AbstractQuantity) -> u.AbstractQuantity:
        a = arc.ustrip("km")
        return u.Q(jnp.stack([jnp.cos(a), jnp.sin(a), jnp.zeros_like(a)]), "km")

    family = lambda t: cxfc.TubularChart(
        cxfc.BishopBuilder(circle, "km"), tau_bounds=BOUNDS
    )
    k = cxfc.rate_of_strain(family, ON_AXIS, u.Q(T0, "s"))
    assert np.allclose(np.asarray(k.value), np.zeros((3, 3)), atol=1e-6)


def test_the_rate_is_per_the_unit_of_the_time_it_was_given() -> None:
    """Seconds give `1 / s`; milliseconds give `1 / ms`, a thousand times smaller."""
    fam = _family(stretch_and_bend)
    in_s = cxfc.rate_of_strain(fam, ON_AXIS, u.Q(T0, "s"))
    in_ms = cxfc.rate_of_strain(fam, ON_AXIS, u.Q(T0 * 1000, "ms"))

    assert "1 / s" in in_s.unit.to_string()
    assert "1 / ms" in in_ms.unit.to_string()
    assert np.asarray(in_ms.value)[0, 0] == pytest.approx(
        np.asarray(in_s.value)[0, 0] / 1000, rel=1e-3
    )


def test_a_batched_time_is_refused() -> None:
    """`K_ij` is a 2-tensor, so the time it is taken at must be a scalar.

    A batched `t` raises the Jacobian's rank above 2; `TubularChart` is
    single-point for the same reason. Unguarded it failed inside the chart
    with `All input arrays must have the same shape`, naming nothing.
    """
    with pytest.raises(ValueError, match="must be a scalar"):
        cxfc.rate_of_strain(
            _family(stretch_and_bend), ON_AXIS, u.Q(jnp.asarray([0.5, T0]), "s")
        )


def test_a_bare_time_is_refused() -> None:
    """Nothing else states what the rate is *per*."""
    with pytest.raises(TypeError, match="must carry a unit"):
        cxfc.rate_of_strain(_family(stretch_and_bend), ON_AXIS, T0)
