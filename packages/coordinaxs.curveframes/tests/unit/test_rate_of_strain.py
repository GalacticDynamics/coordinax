r"""`rate_of_strain` is the ADM $K_{ij}$ of a family of slices.

$K_{ij} = \tfrac{1}{2\alpha}(\partial_t\gamma_{ij} - (\mathcal{L}_\beta
\gamma)_{ij})$ reduces to $\tfrac12\partial_t\gamma_{ij}$ here: the lapse is 1
under absolute time, and the Lie-drag term vanishes because a chart coordinate
is what $\partial_t$ holds fixed, so the coordinate shift is zero. The shift
that does *not* vanish is the ambient one `velocity` reports; the two are
different objects and these pin that they behave differently.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc

S0, T0 = 1.3, 1.0
BOUNDS = (u.Q(0.0, "km"), u.Q(3.0, "km"))
POINT = {"tau": u.Q(S0, "km"), "n1": u.Q(0.2, "km"), "n2": u.Q(0.1, "km")}


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


def test_a_material_labelling_shows_the_tube_stretching() -> None:
    """The curve genuinely deforms, so the rate of strain is non-zero."""
    k = cxfc.rate_of_strain(_family(stretch_and_bend), POINT, u.Q(T0, "s"))
    assert np.asarray(k.value)[0, 0] == pytest.approx(0.77937, abs=1e-4)


def test_arc_length_holds_the_metric_so_its_strain_is_near_zero() -> None:
    """The same curve, relabelled, deforms hardly at all.

    `ArcLength` keeps the parametrisation near unit-speed by construction, so
    `gamma_tau_tau` stays near 1 and its rate of change near 0. The value
    follows the labelling the caller chose -- the same choice `velocity`
    reports, and this never overrides it.
    """
    k = cxfc.rate_of_strain(
        _family(cxfc.ArcLength(stretch_and_bend, "km")), POINT, u.Q(T0, "s")
    )
    assert np.asarray(k.value)[0, 0] == pytest.approx(-0.00548, abs=1e-4)


def test_it_is_symmetric() -> None:
    """`K_ij` is a symmetric 2-tensor, being a derivative of one."""
    k = np.asarray(
        cxfc.rate_of_strain(_family(stretch_and_bend), POINT, u.Q(T0, "s")).value
    )
    assert np.allclose(k, k.T, atol=1e-6)


def test_a_static_curve_does_not_deform() -> None:
    """No time in the curve, so the slice geometry does not change."""

    def circle(arc: u.AbstractQuantity) -> u.AbstractQuantity:
        a = arc.ustrip("km")
        return u.Q(jnp.stack([jnp.cos(a), jnp.sin(a), jnp.zeros_like(a)]), "km")

    family = lambda t: cxfc.TubularChart(
        cxfc.BishopBuilder(circle, "km"), tau_bounds=BOUNDS
    )
    k = cxfc.rate_of_strain(family, POINT, u.Q(T0, "s"))
    assert np.allclose(np.asarray(k.value), np.zeros((3, 3)), atol=1e-6)


def test_the_rate_is_per_the_unit_of_the_time_it_was_given() -> None:
    """Seconds give `1 / s`; milliseconds give `1 / ms`, a thousand times smaller."""
    fam = _family(stretch_and_bend)
    in_s = cxfc.rate_of_strain(fam, POINT, u.Q(T0, "s"))
    in_ms = cxfc.rate_of_strain(fam, POINT, u.Q(T0 * 1000, "ms"))

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
            _family(stretch_and_bend), POINT, u.Q(jnp.asarray([0.5, T0]), "s")
        )


def test_a_bare_time_is_refused() -> None:
    """Nothing else states what the rate is *per*."""
    with pytest.raises(TypeError, match="must carry a unit"):
        cxfc.rate_of_strain(_family(stretch_and_bend), POINT, T0)
