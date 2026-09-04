r"""A chart on a tubular neighbourhood of a curve.

Coordinates are $(\tau, n_1, n_2)$:

$$ \mathbf{x} = \boldsymbol{\gamma}(\tau)
   + n_1\mathbf{U}_1(\tau) + n_2\mathbf{U}_2(\tau) $$

where $(\mathbf{T},\mathbf{U}_1,\mathbf{U}_2)$ is the triad supplied by an
`AbstractCurveFrameBuilder`. The same class serves Frenet--Serret and Bishop;
they differ only in the builder handed in.

Note that $\tau$ is the *curve parameter*, not arc length. The builders are
$\tau$-parameterised, so $g_{\tau\tau}$ carries a $\|\gamma'\|^2$ speed factor
rather than reducing to $(1-k_1n_1-k_2n_2)^2$.
"""

__all__ = ("TubularChart",)

import dataclasses

from typing import Any, ClassVar, final, override

import equinox as eqx
import jax
import jax.numpy as jnp

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import unxt as u
from coordinax._src.base import AbstractParameterizedChart

from .arclength import _is_two_argument
from .base import AbstractCurveFrameBuilder

_MSG_BARE_TIME_BOUNDS = (
    "`TubularChart.tau_bounds` must carry a unit when the builder pins a "
    "station: `tau` is then the evaluation time, and the builder's `tau_unit` "
    "describes the station instead, so nothing else states this coordinate's "
    "dimension."
)


@final
class TubularChart(AbstractParameterizedChart):
    r"""Chart on a tubular neighbourhood of a curve.

    Differentiability is opt-in per instance, exactly as for any parameterized
    chart: a curve that is an `equinox.Module` holding `unxt.Quantity`
    parameters contributes leaves and can be differentiated through; a plain
    function closes over trace-time constants and contributes none.

    Coordinate data must be a single point, not a batch: the forward and
    inverse `pt_map`, and `check_data(..., values=True)`, all raise on
    batched `tau`/`n1`/`n2` (the Jacobian in `jacobian_factor` takes
    `jax.jacfwd` over `tau`, which is not batch-aware). Use `jax.vmap` over
    single-point calls instead -- see the "Working With Curve Charts" guide's
    Limitations section for this and the chart's other boundaries.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    >>> chart = cxfc.TubularChart(
    ...     cxfc.BishopBuilder(circle, "s"),
    ...     tau_bounds=(u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s")),
    ... )
    >>> chart.components
    ('tau', 'n1', 'n2')
    >>> chart.coord_dimensions
    ('time', 'length', 'length')

    """

    builder: AbstractCurveFrameBuilder
    """The curve-frame builder supplying gamma and the triad."""

    _: dataclasses.KW_ONLY

    tau_bounds: tuple[Any, Any]
    """Scan range for the inverse solve.

    Must cover the curve of interest, and for a **closed** curve must not span
    more than one period: a wider range ties the nearest-point solve between
    `gamma(tau)` and `gamma(tau + period)`, the same ambient point.

    A point whose true nearest curve point lies outside `tau_bounds` does not
    raise: the fallback solve can converge to a finite, low-residual `tau`
    outside `tau_bounds` instead. See the curve-charts guide's Limitations
    section for worked examples of both warnings above.
    """

    n_seed: int = eqx.field(static=True, default=64)
    """Seed points for the inverse scan. Static, since it is a loop bound."""

    M: ClassVar[Any]

    @override
    @property
    def M(self) -> Any:
        """The ambient manifold, always flat 3-space regardless of the curve."""
        return cxm.R3

    @property
    def components(self) -> tuple[str, str, str]:
        return ("tau", "n1", "n2")

    @property
    def coord_dimensions(self) -> tuple[str, str, str]:
        # The first coordinate inherits whatever the curve is parameterised by,
        # so this cannot be a class-level tuple the way most charts declare it.
        #
        # `tau_bounds` is the source rather than the builder: it is a required
        # field holding the tau range as a `Quantity`, so it carries the unit
        # structurally -- which is what this property needs and an inferring
        # builder, having no call parameter here, cannot supply.
        #
        # On a worldtube it is the *only* source. `_tau_unit_at` resolves the
        # curve *parameter*, and prefers a declared `tau_unit` over the value
        # handed to it -- but a pinned station makes `tau` the time, and
        # `tau_unit` describes the station, so asking the builder labels a time
        # coordinate `length`.
        if self.is_time_dependent:
            tau_unit = u.unit_of(self.tau_bounds[0])
            if tau_unit is None:
                raise TypeError(_MSG_BARE_TIME_BOUNDS)
        else:
            tau_unit = self.builder._tau_unit_at(self.tau_bounds[0])
        return (str(u.dimension_of(tau_unit)), "length", "length")

    @property
    def cartesian(self) -> cxc.Cart3D:
        return cxc.cart3d

    @property
    def is_time_dependent(self) -> bool:
        """Whether this chart's coordinates depend on a time supplied at call.

        True when the builder wraps a two-argument curve ``gamma(s, t)``.
        `AbstractCurveFrameBuilder._resolve` then reads the builder's single
        argument as the **time**, taking the station from ``builder.station``
        -- so this chart's ``tau`` coordinate is a time too, and only ``n1``
        and ``n2`` remain spatial.

        A spacetime chart needs to know: such a chart is a *fibre bundle* over
        time rather than a factor to multiply time by, and pairing it with a
        time axis would give two time coordinates (see
        `coordinax.charts.GalileanCT`).
        """
        return _is_two_argument(self.builder.curve)

    def check_data(self, data: dict, /, *, values: bool = False, **kw: Any) -> dict:
        # Forward `values`: the base class gates its coordinate-dimension check
        # on it, and binding it as a named parameter keeps it out of `**kw`.
        super().check_data(data, values=values, **kw)
        if values:
            # Inside the reach the Jacobian factor is positive; at the focal
            # distance it vanishes and the coordinates stop being *locally*
            # injective. Necessary, not sufficient, for global injectivity --
            # can't see a point mirrored across the curve or the curve's
            # global self-approach distance (see the curve-charts guide's
            # Limitations section).
            #
            # `~(f > 0)`, not `f <= 0`: a pinned-station builder makes the
            # on-curve speed (and factor) `0/0 = nan`, and `nan <= 0` is
            # False too -- negating `nan > 0` (also False) catches it.
            #
            # Hybrid form, matching ``_src/charts/checks.py`` (see ``nearest.py``
            # for the full mechanics): `eqx.error_if` under trace, plain
            # `ValueError` when concrete. The return value MUST be threaded
            # back into `data` -- an unused result silently vanishes under
            # `jit` (verified: it returned n1=-1.6, well outside the reach).
            pred = jnp.any(~(self.jacobian_factor(data) > 0))
            msg = (
                "point lies outside the reach of the curve: the tubular "
                "coordinates are not locally injective there"
            )
            if isinstance(pred, jax.core.Tracer):
                data = {**data, "n1": eqx.error_if(data["n1"], pred, msg)}
            elif bool(pred):
                raise ValueError(msg)
        return data

    def jacobian_factor(self, data: dict, /) -> Any:
        r"""$\partial\mathbf{x}/\partial\tau$ scaled by the on-curve speed.

        Equals $1-k_1n_1-k_2n_2$ at *any* parametrisation, not only a
        unit-speed one: $\partial\mathbf{x}/\partial\tau$ itself picks up a
        $\|\gamma'\|$ speed factor away from unit speed, but dividing it out
        below cancels that factor, leaving the same dimensionless quantity a
        unit-speed curve would give directly. It is positive inside the
        reach and vanishes at the focal distance, which is the test that
        matters.
        """
        tau, n1, n2 = data["tau"], data["n1"], data["n2"]
        unit = self.builder._tau_unit_at(tau)
        # Derive the unit from the curve (as ``nearest.py`` and
        # ``register_ptmap.py`` do), not hardcode `"km"`: the scale cancels in
        # `dot(dx,T)/speed`, but a hardcoded unit raises `UnitConversionError`
        # for a dimensionless curve.
        ambient_unit = self.builder.location(tau).unit

        def gamma_v(t: jax.Array) -> jax.Array:
            return jnp.asarray(
                self.builder.location(u.Q(t, unit)).ustrip(ambient_unit), dtype=float
            )

        def offset_v(t: jax.Array) -> jax.Array:
            R = self.builder.rotation_matrix(u.Q(t, unit))
            n1_v = n1.ustrip(ambient_unit)
            n2_v = n2.ustrip(ambient_unit)
            return gamma_v(t) + n1_v * R[1] + n2_v * R[2]

        tau_v = tau.ustrip(unit)
        dx = jax.jacfwd(offset_v)(tau_v)
        speed = jnp.linalg.norm(jax.jacfwd(gamma_v)(tau_v))
        # Project onto the tangent: dx is parallel to T, and past the focal
        # distance it REVERSES. `norm(dx)` is sign-blind and bounces back up
        # (measured: at n1=-1.1 on the unit circle, norm=+0.1 but the true
        # factor is -0.1), so a `<= 0` guard built on the norm can only fire
        # exactly at the focal point -- a measure-zero set it will never hit.
        T = self.builder.rotation_matrix(u.Q(tau_v, unit))[0]
        return jnp.dot(dx, T) / speed
