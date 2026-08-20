r"""`pt_map` registrations for `TubularChart`.

Forward is closed form. Inverse is the seeded nearest-point solve, then a
projection of the offset onto the triad -- the tangent component is zero at
the solution by construction, which is exactly the stationarity condition.
"""

__all__: tuple[str, ...] = ()

from typing import Any

import jax.numpy as jnp
import plum

import coordinax.charts as cxc
import unxt as u
from coordinax._src.charts.checks import check_manifold_matches_chart
from coordinax._src.custom_types import OptUSys
from coordinax.manifolds import Rn
from coordinaxs.api.custom_types import CDict

from .chart import TubularChart
from .nearest import nearest_tau

_MSG_RAW_NEEDS_USYS = (
    "coordinate {name!r} carries no unit, so it needs a `usys=` to say what "
    "its numbers mean -- e.g. "
    "`pt_map(p, ..., usys=u.unitsystem('km', 's', 'kg', 'rad'))`. This is the "
    "raw-array route; pass `unxt.Quantity` coordinates instead to skip it."
)


def _usys_unit(usys: OptUSys, name: str, dimension: str, /) -> Any:
    """``usys[dimension]``, or a message naming the coordinate that needed it.

    The single place the unit system is indexed, so the "raw coordinate with
    no `usys`" failure has exactly one wording and the type checker sees the
    `None` ruled out once.
    """
    if usys is None:
        raise ValueError(_MSG_RAW_NEEDS_USYS.format(name=name))
    return usys[dimension]


def _in_usys(value: Any, name: str, dimension: str, usys: OptUSys, /) -> Any:
    """Interpret a raw coordinate through ``usys``; pass a `Quantity` through.

    The raw-array route is the cheapest way in and carries no units, so the
    unit system is what gives its numbers meaning -- the same bargain
    `coordinax`'s own charts make (see `uconvert_to_rad`). A `Quantity` already
    states its unit and is returned untouched, so a mixed dict works.
    """
    if u.unit_of(value) is not None:
        return value
    return u.Q(value, _usys_unit(usys, name, dimension))


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
    ...     cxfc.BishopBuilder(circle, "s"),
    ...     tau_bounds=(u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s")),
    ... )
    >>> p = {"tau": u.Q(0.0, "s"), "n1": u.Q(0.1, "km"), "n2": u.Q(0.0, "km")}
    >>> cxc.pt_map(p, chart.M, chart, chart.M, cxc.cart3d)
    {'x': Q(1.1, 'km'), 'y': Q(0., 'km'), 'z': Q(0., 'km')}

    Raw coordinates take the same route, given a ``usys`` to say what their
    numbers mean, and come back raw:

    >>> usys = u.unitsystem("km", "s", "kg", "rad")
    >>> p = {"tau": 0.0, "n1": 0.1, "n2": 0.0}
    >>> cxc.pt_map(p, chart.M, chart, chart.M, cxc.cart3d, usys=usys)
    {'x': Array(1.1, dtype=float64...),
     'y': Array(0., dtype=float64...),
     'z': Array(0., dtype=float64...)}

    """
    del to_M
    check_manifold_matches_chart(from_M, from_chart, "from_M")
    b = from_chart.builder

    # Raw in -> raw out, as everywhere else in `pt_map`: the caller who came
    # in on the array route wants to leave on it, not be handed Quantities.
    tau_dim = from_chart.coord_dimensions[0]
    raw = u.unit_of(p["tau"]) is None
    tau = _in_usys(p["tau"], "tau", tau_dim, usys)
    n1 = _in_usys(p["n1"], "n1", "length", usys)
    n2 = _in_usys(p["n2"], "n2", "length", usys)

    R = b.rotation_matrix(tau)
    g = b.location(tau)
    # Read the output unit from the system, not from `n1`: in a mixed dict
    # `n1` may be a `Quantity` the caller supplied in some other unit (100 m
    # rather than 0.1 km), and the raw output is defined by the system.
    unit = _usys_unit(usys, "n1", "length") if raw else g.unit
    xyz = g.ustrip(unit) + n1.ustrip(unit) * R[1] + n2.ustrip(unit) * R[2]
    if raw:
        return dict(zip(("x", "y", "z"), xyz, strict=True))
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
    ...     cxfc.BishopBuilder(circle, "s"),
    ...     tau_bounds=(u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s")),
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
    del from_M
    check_manifold_matches_chart(to_M, to_chart, "to_M")
    b = to_chart.builder

    raw = u.unit_of(p["x"]) is None
    xyz = [_in_usys(p[k], k, "length", usys) for k in ("x", "y", "z")]

    unit = b.location(to_chart.tau_bounds[0]).unit
    x = u.Q(jnp.stack([c.ustrip(unit) for c in xyz]), unit)

    tau = nearest_tau(b, x, bounds=to_chart.tau_bounds, n_seed=to_chart.n_seed)
    R = b.rotation_matrix(tau)
    d = x.ustrip(unit) - b.location(tau).ustrip(unit)
    n1, n2 = jnp.dot(d, R[1]), jnp.dot(d, R[2])
    if raw:
        # `tau` comes back from `nearest_tau` as a `Quantity` in the bounds'
        # unit; strip it to the unit system so the dict is uniformly raw.
        tau_unit = _usys_unit(usys, "tau", to_chart.coord_dimensions[0])
        per_length = u.Q(1.0, unit).ustrip(_usys_unit(usys, "n1", "length"))
        return {
            "tau": tau.ustrip(tau_unit),
            "n1": n1 * per_length,
            "n2": n2 * per_length,
        }
    return {"tau": tau, "n1": u.Q(n1, unit), "n2": u.Q(n2, unit)}


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
    ...     cxfc.BishopBuilder(circle, "s"),
    ...     tau_bounds=(u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s")),
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
