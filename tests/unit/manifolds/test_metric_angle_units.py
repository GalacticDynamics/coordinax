"""The analytic Euclidean metrics must not depend on how the caller spells an angle.

A metric component in a degree-parameterised basis is ``r²(π/180)²`` per ``deg²``,
which is the *same tensor* as ``r²`` per ``rad²``.  The analytic rules compute the
radian-convention value, so they must label it per ``rad²`` whatever unit the point
carries -- otherwise every metric quantity is wrong by ``(180/π)ⁿ`` for degree input.

`jac_pt_map` is the independent reference: it differentiates the chart transition
and cannot get the convention wrong.
"""

import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm

# (chart, point-in-radians, angular keys)
CASES = {
    "polar2d": (
        cxc.polar2d,
        {"r": u.Q(2.0, "m"), "theta": u.Angle(0.7, "rad")},
        ("theta",),
    ),
    "cyl3d": (
        cxc.cyl3d,
        {"rho": u.Q(2.0, "m"), "phi": u.Angle(0.7, "rad"), "z": u.Q(0.5, "m")},
        ("phi",),
    ),
    "sph3d": (
        cxc.sph3d,
        {"r": u.Q(2.0, "m"), "theta": u.Angle(0.7, "rad"), "phi": u.Angle(0.3, "rad")},
        ("theta", "phi"),
    ),
    "math_sph3d": (
        cxc.math_sph3d,
        {"r": u.Q(2.0, "m"), "theta": u.Angle(0.3, "rad"), "phi": u.Angle(0.7, "rad")},
        ("theta", "phi"),
    ),
    "lonlat_sph3d": (
        cxc.lonlat_sph3d,
        {
            "lon": u.Angle(0.7, "rad"),
            "lat": u.Angle(0.3, "rad"),
            "distance": u.Q(2.0, "m"),
        },
        ("lon", "lat"),
    ),
}


def _in_degrees(point, angular):
    """The same geometric point, with its angles spelled in degrees."""
    return {
        k: (
            u.Angle(np.rad2deg(np.asarray(v.value)).item(), "deg")
            if k in angular
            else v
        )
        for k, v in point.items()
    }


def _agree(analytic, pullback):
    """Compare diagonals in a common unit basis.

    The analytic rule labels its value per ``rad**2`` while the pullback labels
    the same tensor in whatever the point carried, so the raw ``.value`` arrays
    differ by ``(180/pi)**2`` for degree input even when both are right.  The
    conversion is the whole point of the fix, so the test must do it too.
    """
    diag = analytic.diagonal
    for i in range(len(diag)):
        entry = pullback[i, i]
        got = u.uconvert(entry.unit, diag[i]).value
        assert np.allclose(got, entry.value, rtol=1e-10, atol=1e-12), (
            f"component {i}: analytic {got} vs pullback {entry.value} [{entry.unit}]"
        )


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("spelling", ["rad", "deg"])
def test_analytic_metric_matches_the_jacobian_pullback(name, spelling):
    """The analytic rule must agree with `jac_pt_map` in either angle spelling."""
    chart, point_rad, angular = CASES[name]
    point = point_rad if spelling == "rad" else _in_degrees(point_rad, angular)

    analytic = cxm.metric_matrix(cxm.R3 if len(point) == 3 else cxm.R2, point, chart)
    jac = cxc.jac_pt_map(point, chart, chart.cartesian)
    _agree(analytic, jac.T @ jac)


def test_a_degree_step_has_the_length_a_degree_step_has():
    """One degree of longitude on a 1 km sphere is `pi/180` km, not 1 km."""
    at = {
        "lon": u.Angle(30.0, "deg"),
        "lat": u.Angle(0.0, "deg"),
        "distance": u.Q(1.0, "km"),
    }
    v = {
        "lon": u.Angle(1.0, "deg"),
        "lat": u.Angle(0.0, "deg"),
        "distance": u.Q(0.0, "km"),
    }
    got = float(cxm.norm(v, cxc.lonlat_sph3d, at=at).ustrip("km"))
    assert got == pytest.approx(np.pi / 180, rel=1e-10)
