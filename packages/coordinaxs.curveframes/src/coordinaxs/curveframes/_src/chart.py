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

from typing import Any, ClassVar, cast, final, override

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

#: Both messages name `tau_bounds`, which is the whole point of raising here:
#: left to `nearest_tau`, the same mistakes surface as a bare
#: `UnitConversionError` or `AttributeError` from inside the scan setup, with
#: nothing to say which field was wrong (measured, both cases).
_MSG_BOUNDS_DISAGREE = (
    "`TubularChart.tau_bounds` must state one dimension, but the lower bound "
    "is {lo} and the upper is {hi}. The scan strips both ends to the lower "
    "bound's unit, which a different dimension cannot convert to."
)

_MSG_WORLDTUBE_BOUNDS_NOT_TIME = (
    "`TubularChart.tau_bounds` must be times when the builder pins a station, "
    "but they are {lo}. `tau` is the evaluation time on this branch, so "
    "bounds of another dimension label the coordinate one way while the "
    "builder evaluates it the other -- leaving no value the chart both "
    "declares and accepts."
)

_MSG_OUTSIDE_REACH = (
    "point lies outside the reach of the curve: the tubular "
    "coordinates are not locally injective there"
)

_MSG_DEGENERATE_FRAME = (
    "the frame is degenerate at the tube axis, so no offset gives a chart "
    "here: `dx/dtau` has no component along the curve's own spatial tangent, "
    "leaving it in the normal plane `span(U1, U2)` and the Jacobian singular. "
    "This is *not* a reach failure -- moving `n1`, `n2` inward cannot help. "
    "Reparametrise so `tau` advances along the curve, or chart the slice at a "
    "fixed time with `AtTime(curve, t)` instead of the worldtube."
)

_MSG_PINNED_STATION_ON_ONE_ARGUMENT = (
    "the builder pins `station=` on a one-argument curve, so this chart's "
    "`tau` has nothing left to vary: every `tau` maps to the same ambient "
    "point and `jacobian_factor` is `nan`. Such a builder is a frame *field*, "
    "which is fine on its own -- it just cannot parameterise a chart. Drop "
    "`station=` for a chart whose `tau` moves along the curve, or pass a "
    "two-argument `gamma(s, t)`, where the station pins `s` and leaves `tau` "
    "the time."
)


_MSG_BOUNDS_HALF_BARE = (
    "`TubularChart.tau_bounds` must be both `Quantity` or both bare, but got "
    "{lo} and {hi}. A bare bound takes its unit from the builder's declared "
    "`tau_unit` and a `Quantity` one carries its own, so a mixed pair has no "
    "single answer to what the bounds mean."
)


def _bounds_dimension(bound: Any, /) -> str | None:
    """Return the dimension ``bound`` states, or `None` if it is bare."""
    unit = u.unit_of(bound)
    return None if unit is None else str(u.dimension_of(unit))


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
    ...     cxfc.BishopBuilder(circle, "s", normal_0="auto"),
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

    Exactly one period is not safe either, for a different reason. A
    rotation-minimising frame is not periodic: carried once around a closed
    *space* curve it returns rotated about the tangent, so the chart is torn
    at the seam -- `tau` just inside either end names the same station and the
    same ``(n1, n2)``, but a different ambient point. Measured on a trefoil
    over ``(0, 2*pi)``: the normal comes back ``-2.225041 rad`` (``-127.485
    deg``) round, putting the two sides of the seam ``0.358726 km`` apart at
    ``n1 = 0.2 km``. A **planar** closed curve has no such tear, which is why
    the obvious probe misses this. `TubularChart.holonomy` measures it.

    A point whose true nearest curve point lies outside `tau_bounds` does not
    raise: the fallback solve can converge to a finite, low-residual `tau`
    outside `tau_bounds` instead. See the curve-charts guide's Limitations
    section for worked examples of both warnings above.

    That degradation needs the curve to be *evaluable* past `tau_bounds`, which
    an `ArcLength` carrying a finite `s_max` is not: its interpolation covers
    only ``[-m, s_max + m]``, so a query whose answer lies beyond that raises
    rather than degrading. Setting `s_max` to `tau_bounds[1]`, which is all
    `ArcLength.s_max` asks for in-bounds queries, is *not* enough for
    out-of-bounds ones -- size it against the answers the solve may return, not
    against the scan range.
    """

    n_seed: int = eqx.field(static=True, default=64)
    """Seed points for the inverse scan. Static, since it is a loop bound."""

    M: ClassVar[Any]

    @override
    @property
    def M(self) -> Any:
        """The ambient manifold, always flat 3-space regardless of the curve."""
        return cxm.R3

    def __check_init__(self) -> None:
        """Require `tau_bounds` to state one dimension, state it, and mean it.

        `tau_bounds[0]` alone is what the unit is read from -- it labels the
        coordinate, and `nearest_tau` strips both ends to it -- so a
        disagreeing `tau_bounds[1]` is not caught until the inverse solve
        runs, and not named when it is. Measured on an unguarded chart:
        ``(Q(0, "s"), Q(2, "km"))`` builds, reports `('time', ...)`, and dies
        in the scan with `UnitConversionError`; a mixed bare/`Quantity` pair
        dies with ``'float' object has no attribute 'ustrip'``. Neither
        mentions `tau_bounds`, and neither fires until someone maps a point.

        A *converting* pair is fine and stays fine -- ``(Q(0, "s"), Q(2000,
        "ms"))`` strips to 2 s -- so this checks the dimension, not the unit.

        Bare bounds are the array fastpath and stay legal on the static
        branch, where the builder's declared `tau_unit` says what the numbers
        mean. On a **worldtube** they never are: `tau_unit` describes the
        pinned station, so nothing states this coordinate's dimension (see
        `_tau_unit`). That was `coord_dimensions`' own check until it moved
        here -- it is a pure function of the builder's curve, knowable the
        moment the chart is built, and leaving it downstream is the shape
        this method exists to remove. Reading the arity again costs nothing:
        `AbstractCurveFrameBuilder.__check_init__` already inspected the same
        curve, and the result is cached.

        Carrying a unit is still not enough on that branch: it has to be a
        *time*. Length bounds on a worldtube built a chart that declared
        `('length', 'length', 'length')` and accepted only `time` -- its own
        `check_data` refused a `tau` in `s`, and a `tau` in `km` died in the
        scan with `UnitConversionError`. No value satisfied both (#820).

        Rejecting the bounds rather than reconciling the label with the
        runtime, because the library reserves a two-argument curve's second
        argument for the time: `AtTime` binds it, `GalileanCT` refuses a chart
        that has one, and `TimeDep` sends every other parameter to a builder
        field rather than a call-time argument.
        """
        lo, hi = (_bounds_dimension(b) for b in self.tau_bounds)
        if lo != hi:
            msg = _MSG_BOUNDS_HALF_BARE if None in (lo, hi) else _MSG_BOUNDS_DISAGREE
            raise ValueError(msg.format(lo=lo or "bare", hi=hi or "bare"))
        # `TypeError`, matching the sibling in `base.py`: a missing unit is
        # the wrong *kind* of value, where two disagreeing ones above are the
        # wrong value. `_tau_unit` raised this same message from the property.
        if lo is None and self.is_time_dependent:
            raise TypeError(_MSG_BARE_TIME_BOUNDS)
        # And having a unit is not enough: it has to be a *time*. `_tau_unit`
        # labels this coordinate from the bounds while `_resolve` evaluates it
        # as the time regardless, so any other dimension leaves the two
        # disagreeing with no value satisfying both (see #820).
        if self.is_time_dependent and lo != "time":
            raise ValueError(_MSG_WORLDTUBE_BOUNDS_NOT_TIME.format(lo=lo))

        # Here and not on the builder: a pinned station is legitimate there,
        # and the builder does not know it is about to become a chart. The
        # worldtube guards above miss it because `is_time_dependent` reads the
        # curve's *arity*, which a station does not change.
        if self.builder.station is not None and not self.is_time_dependent:
            raise ValueError(_MSG_PINNED_STATION_ON_ONE_ARGUMENT)

    @property
    def components(self) -> tuple[str, str, str]:
        return ("tau", "n1", "n2")

    @property
    def _tau_unit(self) -> u.AbstractUnit:
        """The unit of *this chart's* ``tau``, which is not the builder's.

        The two coincide on the static branch and part on a worldtube, which
        is the whole of what this property exists to say. Everything the chart
        does with ``tau`` -- label it, strip it, seed a scan over it -- routes
        through here rather than the builder, or it gets the curve
        *parameter's* unit where this *coordinate's* was wanted.

        `tau_bounds` is the source rather than the builder: it is a required
        field holding the tau range as a `Quantity`, so it carries the unit
        structurally -- which is what this needs, and an inferring builder,
        having no call parameter here, cannot supply.

        On a worldtube it is the *only* source. `_tau_unit_at` resolves the
        curve *parameter*, and prefers a declared `tau_unit` over the value
        handed to it -- but a pinned station makes `tau` the time, and
        `tau_unit` describes the station. Asking the builder therefore labels
        a time coordinate `length`, and strips a time in kilometres.

        Bare bounds cannot reach the worldtube branch: `__check_init__`
        rejects that pair, so the `unit_of` below is never `None` there.
        """
        if self.is_time_dependent:
            return cast("u.AbstractUnit", u.unit_of(self.tau_bounds[0]))
        return self.builder._tau_unit_at(self.tau_bounds[0])

    @property
    def coord_dimensions(self) -> tuple[str, str, str]:
        # The first coordinate inherits whatever the curve is parameterised by,
        # so this cannot be a class-level tuple the way most charts declare it.
        return (str(u.dimension_of(self._tau_unit)), "length", "length")

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

    def holonomy(self, *, n_scale: int = 16) -> u.AbstractQuantity:
        r"""Angle by which the frame fails to close around a closed curve.

        A rotation-minimising (Bishop) frame is not periodic: carried once
        around a closed space curve it comes back rotated about the tangent by
        an angle that is a property of the curve, not of the solver. The chart
        is then **torn at the seam** -- ``tau`` just above ``tau_bounds[0]``
        and just below ``tau_bounds[1]`` name the same station and the same
        ``(n1, n2)`` offset but different ambient points.

        Returns the size of that tear. Zero when the frame closes, and zero
        when the curve does not: holonomy belongs to a loop. Closure is judged
        relative to the curve's own size, at the working dtype's ``sqrt(eps)``.

        An open curve never evaluates the frame, so one whose frame is
        undefined answers 0 rather than raising -- a straight line under
        `FrenetSerretBuilder` has no normal, but no seam either.

        Examples
        --------
        A trefoil over exactly one period closes, but its frame does not::

            >>> import jax.numpy as jnp, numpy as np, unxt as u
            >>> import coordinaxs.curveframes as cxfc
            >>> def trefoil(tau):
            ...     t = tau.ustrip("s")
            ...     return u.Q(jnp.stack([jnp.sin(t) + 2 * jnp.sin(2 * t),
            ...                           jnp.cos(t) - 2 * jnp.cos(2 * t),
            ...                           -jnp.sin(3 * t)]), "km")
            >>> bounds = (u.Q(0.0, "s"), u.Q(float(2 * np.pi), "s"))
            >>> b = cxfc.BishopBuilder(trefoil, "s", normal_0="auto")
            >>> ch = cxfc.TubularChart(b, tau_bounds=bounds)
            >>> bool(abs(ch.holonomy().ustrip("rad")) > 2.2)
            True

        A *planar* closed curve has none, which is why the obvious probe
        misses this::

            >>> def circle(tau):
            ...     t = tau.ustrip("s")
            ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
            ...                           jnp.zeros_like(t)]), "km")
            >>> b = cxfc.BishopBuilder(circle, "s", normal_0="auto")
            >>> ch = cxfc.TubularChart(b, tau_bounds=bounds)
            >>> bool(abs(ch.holonomy().ustrip("rad")) < 1e-8)
            True

        """
        # Refused, not clamped: both smaller values are silently wrong, in
        # opposite directions.
        if n_scale < 3:
            msg = (
                f"`n_scale` must be at least 3, got {n_scale}: it sizes the "
                "curve so closure can be judged relative to it, and a closed "
                "curve's two endpoints coincide -- so 2 samples span nothing "
                "and report no seam where there is one, while 1 compares a "
                "point with itself and reports a seam on any curve at all."
            )
            raise ValueError(msg)

        unit = self._tau_unit
        lo = jnp.asarray(self.tau_bounds[0].ustrip(unit))
        hi = jnp.asarray(self.tau_bounds[1].ustrip(unit))

        # Spread about the mean, not a radius from the origin, which fails for
        # a curve passing through it. `location`, not `curve`: it routes
        # through `_resolve`, so a two-argument worldtube gets `(station,
        # time)` rather than a time where a station belongs.
        seeds = jnp.linspace(lo, hi, n_scale)
        len_unit = u.unit_of(self.builder.location(u.Q(lo, unit)))
        pts = jax.vmap(lambda t: self.builder.location(u.Q(t, unit)).ustrip(len_unit))(
            seeds
        )
        scale = jnp.max(jnp.linalg.norm(pts - jnp.mean(pts, axis=0), axis=-1))

        g_lo, g_hi = pts[0], pts[-1]
        gap = jnp.linalg.norm(g_hi - g_lo)
        eps = jnp.sqrt(jnp.finfo(gap.dtype).eps)
        closed = gap <= eps * scale

        def _angle(_: Any = None) -> jax.Array:
            """Measure the normal's rotation in the start frame's `(N, B)`.

            `atan2`, not `arccos`: the sign says which way it twists.
            """
            r_lo = self.builder.rotation_matrix(u.Q(lo, unit))
            r_hi = self.builder.rotation_matrix(u.Q(hi, unit))
            n_lo, b_lo = r_lo[1], r_lo[2]
            n_hi = r_hi[1]
            return jnp.arctan2(jnp.dot(n_hi, b_lo), jnp.dot(n_hi, n_lo))

        # Branched, not `jnp.where`: those two calls are a parallel-transport
        # ODE solve and the whole cost (6.9 s against 5.3 ms for the scan), and
        # an open curve discards them. Hybrid because a Python branch cannot
        # test a tracer, and `lax.cond` eagerly would trace the solve it skips.
        zero = jnp.zeros((), dtype=gap.dtype)
        if isinstance(closed, jax.core.Tracer):
            angle = jax.lax.cond(closed, _angle, lambda _: zero, None)
        else:
            angle = _angle() if bool(closed) else zero

        return u.Q(angle, "rad")

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
            factor = self.jacobian_factor(data)
            # Evaluated at the tube *axis* too, because the two ways this can
            # vanish want different words: a focal failure needs an offset and
            # leaves the axis healthy, while a transversely-moving worldtube is
            # singular at the axis, where no offset helps. Costs a second
            # `jacfwd`, and cannot be deferred to the failing branch -- under
            # `jit` there is no branch to defer it to.
            axis_data = {**data, "n1": data["n1"] * 0, "n2": data["n2"] * 0}
            axis = self.jacobian_factor(axis_data)

            # Against `sqrt(eps)`, not a bare `0`: a degenerate chart does not
            # come back exactly zero (`0.0` eagerly, `1.9469e-17` compiled), so
            # `> 0` let the guard pass a singular chart once compiled.
            # `~(x > tol)` rather than `x <= tol` so a NaN still refuses.
            tol = jnp.sqrt(jnp.finfo(jnp.asarray(factor).dtype).eps)
            bad = jnp.any(~(factor > tol))
            axis_bad = jnp.any(~(axis > tol))

            for pred, msg in (
                (bad & axis_bad, _MSG_DEGENERATE_FRAME),
                (bad & ~axis_bad, _MSG_OUTSIDE_REACH),
            ):
                if isinstance(pred, jax.core.Tracer):
                    data = {**data, "n1": eqx.error_if(data["n1"], pred, msg)}
                elif bool(pred):
                    raise ValueError(msg)
        return data

    def jacobian_factor(self, data: dict, /) -> Any:
        r"""$\partial\mathbf{x}/\partial\tau$ scaled by the on-curve speed.

        On the **static** branch this equals $1-k_1n_1-k_2n_2$ at *any*
        parametrisation, not only a unit-speed one:
        $\partial\mathbf{x}/\partial\tau$ itself picks up a $\|\gamma'\|$
        speed factor away from unit speed, but dividing it out below cancels
        that factor, leaving the same dimensionless quantity a unit-speed
        curve would give directly. It is positive inside the reach and
        vanishes at the focal distance, which is the test that matters.

        **Not on a worldtube**, where $\tau$ is a time rather than the curve
        parameter. There this is $\cos$ of the angle between the station's
        velocity and the curve's spatial tangent at $n=0$, so it is already
        below 1 on the axis and can be 0 there: a rod spun about its end
        (``s * (cos t, sin t, 0)``) moves purely transversely, giving exactly
        ``0.0`` on the whole $n_2=0$ plane -- the axis included -- and
        $n_2/s_0$ off it. That is a degenerate *frame*, not a focal distance,
        and `check_data` says so separately.
        """
        tau, n1, n2 = data["tau"], data["n1"], data["n2"]
        # The chart's `tau`, not the builder's curve parameter: on a worldtube
        # those are a time and a station respectively, and asking the builder
        # strips seconds in kilometres. See `_tau_unit`.
        unit = self._tau_unit
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
