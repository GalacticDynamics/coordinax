r"""`pt_map` registrations for `TubularChart`.

Forward is closed form. Inverse is the seeded nearest-point solve, then a
projection of the offset onto the triad -- the tangent component is zero at
the solution by construction, which is exactly the stationarity condition.
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import plum

import coordinax.charts as cxc
import unxt as u
from coordinax._src.custom_types import CDict, OptUSys
from coordinax.manifolds import Rn

from .chart import TubularChart
from .nearest import nearest_tau


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: TubularChart,
    to_M: Rn,
    to_chart: cxc.Cart3D,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""TubularChart -> Cart3D: $\gamma(\tau) + n_1U_1 + n_2U_2$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    >>> chart = cxfc.TubularChart(
    ...     cxfc.BishopBuilder(circle),
    ...     tau_bounds=(u.Q(-1.0, "s"), u.Q(7.0, "s")),
    ... )
    >>> p = {"tau": u.Q(0.0, "s"), "n1": u.Q(0.1, "km"), "n2": u.Q(0.0, "km")}
    >>> cxc.pt_map(p, chart.M, chart, chart.M, cxc.cart3d)
    {'x': Q(1.1, 'km'), 'y': Q(0., 'km'), 'z': Q(0., 'km')}

    """
    del to_M, usys
    assert from_M == from_chart.M  # noqa: S101
    b = from_chart.builder
    tau = p["tau"]
    R = b.rotation_matrix(tau)
    g = b.location(tau)
    unit = g.unit
    xyz = (
        jnp.asarray(g.ustrip(unit), dtype=float)
        + jnp.asarray(p["n1"].ustrip(unit), dtype=float) * R[1]
        + jnp.asarray(p["n2"].ustrip(unit), dtype=float) * R[2]
    )
    return {k: u.Q(xyz[i], unit) for i, k in enumerate(("x", "y", "z"))}


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: cxc.Cart3D,
    to_M: Rn,
    to_chart: TubularChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""Cart3D -> TubularChart, via the nearest-point projection.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    >>> chart = cxfc.TubularChart(
    ...     cxfc.BishopBuilder(circle),
    ...     tau_bounds=(u.Q(-1.0, "s"), u.Q(7.0, "s")),
    ... )

    A point exactly on the curve has zero offsets. Bishop runs an ODE solve
    internally, so the recovered values are checked with a tolerance rather
    than pinned to exact digits:

    >>> on_curve = chart.builder.location(u.Q(0.0, "s"))
    >>> p = {k: on_curve[i] for i, k in enumerate(("x", "y", "z"))}
    >>> back = cxc.pt_map(p, chart.M, cxc.cart3d, chart.M, chart)
    >>> bool(jnp.allclose(back["tau"].ustrip("s"), 0.0, atol=1e-6))
    True
    >>> bool(jnp.allclose(back["n1"].ustrip("km"), 0.0, atol=1e-6))
    True
    >>> bool(jnp.allclose(back["n2"].ustrip("km"), 0.0, atol=1e-6))
    True

    """
    del from_M, usys
    assert to_M == to_chart.M  # noqa: S101
    b = to_chart.builder
    unit = b.location(to_chart.tau_bounds[0]).unit
    x = u.Q(
        jnp.stack(
            [jnp.asarray(p[k].ustrip(unit), dtype=float) for k in ("x", "y", "z")]
        ),
        unit,
    )

    tau = nearest_tau(b, x, bounds=to_chart.tau_bounds, n_seed=to_chart.n_seed)
    R = b.rotation_matrix(tau)
    d = jnp.asarray(x.ustrip(unit), dtype=float) - jnp.asarray(
        b.location(tau).ustrip(unit), dtype=float
    )
    return {
        "tau": tau,
        "n1": u.Q(jnp.dot(d, R[1]), unit),
        "n2": u.Q(jnp.dot(d, R[2]), unit),
    }


@plum.dispatch
def pt_map(
    p: CDict,
    from_M: Rn,
    from_chart: TubularChart,
    to_M: Rn,
    to_chart: TubularChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """TubularChart -> TubularChart. Identity only for the *same* chart object.

    Parameterized charts compare conservatively (equal only when identical), so
    this declines to a Cartesian round trip whenever the two differ.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    >>> chart = cxfc.TubularChart(
    ...     cxfc.BishopBuilder(circle),
    ...     tau_bounds=(u.Q(-1.0, "s"), u.Q(7.0, "s")),
    ... )
    >>> p = {"tau": u.Q(0.0, "s"), "n1": u.Q(0.1, "km"), "n2": u.Q(0.0, "km")}
    >>> cxc.pt_map(p, chart.M, chart, chart.M, chart) is p
    True

    """
    del from_M, to_M, usys
    if from_chart is to_chart:
        return p
    xyz = pt_map(p, from_chart.M, from_chart, to_chart.M, cxc.cart3d)
    return pt_map(xyz, to_chart.M, cxc.cart3d, to_chart.M, to_chart)
