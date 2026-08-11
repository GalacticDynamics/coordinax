r"""Induced metric for `TubularChart`.

$g = J^\top J$ with $J = \partial(x,y,z)/\partial(\tau,n_1,n_2)$. Taking it
from the Jacobian rather than a closed form is deliberate: it is correct for
both Frenet--Serret (which has $d\tau\,dn_i$ cross terms from torsion) and
Bishop (which does not), and it needs no curvature accessors that the
builders do not expose.
"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import plum

import unxt as u
from coordinax._src.metric.matrix import DenseMetric
from coordinax.manifolds import Rn
from coordinaxs.api.manifolds import metric_matrix

from .chart import TubularChart
from .register_ptmap import pt_map


@plum.dispatch
def metric_matrix(M: Rn, point: dict, chart: TubularChart, /) -> DenseMetric:
    r"""Induced metric on a tubular neighbourhood, as $J^\top J$.

    Bishop is rotation-minimising, so the metric is diagonal (no
    $d\tau\,dn_i$ cross terms) with unit blocks in the normal directions;
    $g_{\tau\tau}$ is not $(1-k_1n_1-k_2n_2)^2$ because the builders are
    $\tau$-parameterised rather than unit-speed, so it carries a
    $\|\gamma'\|^2$ speed factor:

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc
    >>> from coordinaxs.api.manifolds import metric_matrix

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    >>> chart = cxfc.TubularChart(
    ...     cxfc.BishopBuilder(circle),
    ...     tau_bounds=(u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s")),
    ... )
    >>> at = {"tau": u.Q(0.7, "s"), "n1": u.Q(0.13, "km"), "n2": u.Q(-0.21, "km")}
    >>> g = metric_matrix(chart.M, at, chart).matrix
    >>> bool(jnp.allclose(g[0, 1], 0.0, atol=1e-8))
    True
    >>> bool(jnp.allclose(g[1, 1], 1.0, atol=1e-8))
    True
    >>> bool(jnp.allclose(g[2, 2], 1.0, atol=1e-8))
    True

    Frenet--Serret has torsion, so it picks up nonzero $d\tau\,dn_i$ cross
    terms that Bishop does not:

    >>> def helix(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")

    >>> chart2 = cxfc.TubularChart(
    ...     cxfc.FrenetSerretBuilder(helix),
    ...     tau_bounds=(u.Q(-1.0, "s"), u.Q(6.0, "s")),
    ... )
    >>> g2 = metric_matrix(chart2.M, at, chart2).matrix
    >>> bool(jnp.allclose(g2[0, 1], 0.0, atol=1e-6))
    False

    """
    del M
    unit = chart.builder.location(chart.tau_bounds[0]).unit
    tau_unit = chart.builder.tau_unit

    def to_xyz(v: jax.Array) -> jax.Array:
        p = {"tau": u.Q(v[0], tau_unit), "n1": u.Q(v[1], unit), "n2": u.Q(v[2], unit)}
        out = pt_map(p, chart.M, chart, chart.M, chart.cartesian)
        return jnp.stack(
            [jnp.asarray(out[k].ustrip(unit), dtype=float) for k in ("x", "y", "z")]
        )

    v = jnp.stack(
        [
            jnp.asarray(point["tau"].ustrip(tau_unit), dtype=float),
            jnp.asarray(point["n1"].ustrip(unit), dtype=float),
            jnp.asarray(point["n2"].ustrip(unit), dtype=float),
        ]
    )
    J = jax.jacfwd(to_xyz)(v)
    return DenseMetric(J.T @ J)
