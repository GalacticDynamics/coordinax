# Working With Curve Charts

`coordinaxs.curveframes` provides `TubularChart`: a `coordinax.charts` chart whose coordinates $(\tau, n_1, n_2)$ locate a point relative to a curve,

$$
\mathbf{x} = \boldsymbol{\gamma}(\tau) + n_1\mathbf{U}_1(\tau) + n_2\mathbf{U}_2(\tau)
$$

where $(\mathbf{T}, \mathbf{U}_1, \mathbf{U}_2)$ is the triad supplied by a `FrenetSerretBuilder` or `BishopBuilder`. This guide covers construction, the forward and inverse maps, differentiating through a fitted curve, the induced metric, and where the chart stops being valid.

For the frame-based moving-frame machinery `TubularChart` builds on, see {doc}`Working With Curve Frames <../packages/coordinaxs.curveframes/guide>`. For the chart system in general, see [Working With Charts](charts.md).

## Why A Chart, Not A Frame

`coordinaxs.curveframes` already lets you ride along a curve: wrap a builder in `coordinax.transforms.TimeDep` and call `act(op, tau, x)`. There, $\tau$ is an argument you supply at evaluation time — a _parameter_ of the transform, external to the point being transformed.

```{code-block} python
>>> import jax.numpy as jnp
>>> import unxt as u
>>> import coordinax.transforms as cxfm
>>> import coordinaxs.curveframes as cxfc

>>> def circle(tau):
...     t = tau.ustrip("s")
...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

>>> op = cxfm.TimeDep(cxfc.BishopBuilder(circle))
>>> x = u.Q(jnp.array([0.5, 0.0, 0.0]), "km")
>>> tau = u.Q(0.7, "s")
>>> cxfm.act(op, tau, x)  # tau supplied separately from x
Q([...], 'km')

```

`TubularChart` makes $\tau$ a _coordinate_ instead: one of the three components of the point itself, recovered the same way `n1` and `n2` are.

```{code-block} python
>>> import coordinax.charts as cxc

>>> BOUNDS = (u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s"))
>>> chart = cxfc.TubularChart(cxfc.BishopBuilder(circle), tau_bounds=BOUNDS)
>>> p = {"x": u.Q(0.5, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
>>> got = cxc.pt_map(p, chart.M, cxc.cart3d, chart.M, chart)
>>> sorted(got)
['n1', 'n2', 'tau']

```

That is a categorical difference, not a stylistic one. A frame transform has no notion of "the $\tau$ of this point" — you either already know it or you don't ask. A chart does: $\tau$ is data carried by the point, so the general chart machinery (`pt_map`, its inverse, Jacobians, the induced metric) applies to it exactly as it does to `n1` and `n2`. That is what makes a fitted curve's position _along_ the curve a quantity you can solve for, differentiate, and feed a metric — the subject of the rest of this guide.

## Constructing A Chart

`TubularChart` wraps either builder unchanged — it is agnostic to which triad supplies $\mathbf{U}_1,\mathbf{U}_2$:

```{code-block} python
>>> ch_bishop = cxfc.TubularChart(cxfc.BishopBuilder(circle), tau_bounds=BOUNDS)
>>> ch_bishop.components
('tau', 'n1', 'n2')
>>> ch_bishop.coord_dimensions
('time', 'length', 'length')

>>> ch_frenet = cxfc.TubularChart(cxfc.FrenetSerretBuilder(circle), tau_bounds=BOUNDS)
>>> ch_frenet.components
('tau', 'n1', 'n2')

```

`coord_dimensions` follows whatever the curve is parameterised by, not a fixed `"time"` — a curve parameterised by arc length reports `"length"` for $\tau$ too:

```{code-block} python
>>> def by_length(tau):
...     s = tau.ustrip("km")
...     return u.Q(jnp.stack([s, jnp.zeros_like(s), jnp.zeros_like(s)]), "km")

>>> ch_len = cxfc.TubularChart(
...     cxfc.BishopBuilder(by_length, "km"),
...     tau_bounds=(u.Q(0.0, "km"), u.Q(1.0, "km")),
... )
>>> ch_len.coord_dimensions
('length', 'length', 'length')

```

`tau_bounds` sets the scan range the inverse solve seeds from (below); it must cover the $\tau$ values you intend to query. For a curve that closes on itself, it must also cover **no more than one period** — see [Limitations](#limitations).

## Both Directions

The forward map is the closed-form definition: $\gamma(\tau)$ plus the normal-plane offset. The inverse projects an ambient point onto the curve (`nearest_tau`, a seeded Newton solve on the stationarity condition $\mathbf{T}\cdot(\mathbf{x}-\boldsymbol{\gamma}) = 0$) and reads off the offset. Round-tripping recovers the original coordinates:

```{code-block} python
>>> p = {"tau": u.Q(0.7, "s"), "n1": u.Q(0.13, "km"), "n2": u.Q(-0.21, "km")}
>>> xyz = cxc.pt_map(p, ch_bishop.M, ch_bishop, ch_bishop.M, cxc.cart3d)
>>> xyz
{'x': Q(0.86427167, 'km'), 'y': Q(0.72796599, 'km'), 'z': Q(0.21, 'km')}

>>> back = cxc.pt_map(xyz, ch_bishop.M, cxc.cart3d, ch_bishop.M, ch_bishop)
>>> bool(jnp.allclose(back["tau"].ustrip("s"), 0.7, atol=1e-6))
True
>>> bool(jnp.allclose(back["n1"].ustrip("km"), 0.13, atol=1e-6))
True
>>> bool(jnp.allclose(back["n2"].ustrip("km"), -0.21, atol=1e-6))
True

```

## Differentiability

Treating $\tau$ as a coordinate rather than a fixed parameter is what makes this reachable: fitting a curve to data means finding the curve parameters for which the chart's coordinates match observations, which needs a gradient through the _inverse_ transition — through the root-find, and for Bishop, through the parallel-transport ODE nested inside it.

Make the curve's radius a live, fittable parameter by holding it in an `equinox.Module`:

```{code-block} python
>>> import equinox as eqx
>>> import jax

>>> class Helix(eqx.Module):
...     radius: u.AbstractQuantity
...     def __call__(self, tau):
...         t = tau.ustrip("s")
...         r = self.radius.ustrip("km")
...         return u.Q(jnp.stack([r * jnp.cos(t), r * jnp.sin(t), 0.3 * t]), "km")

>>> HELIX_BOUNDS = (u.Q(-1.0, "s"), u.Q(6.0, "s"))
>>> def chart_for(radius_km):
...     curve = Helix(radius=u.Q(radius_km, "km"))
...     return cxfc.TubularChart(cxfc.BishopBuilder(curve), tau_bounds=HELIX_BOUNDS)

>>> import coordinax.manifolds as cxm
>>> x = {"x": u.Q(1.1, "km"), "y": u.Q(0.4, "km"), "z": u.Q(0.2, "km")}
>>> def n1_of_radius(radius_km):
...     return cxc.pt_map(x, cxm.R3, cxc.cart3d, cxm.R3, chart_for(radius_km))["n1"].ustrip("km")

```

`jax.grad` differentiates straight through `optimistix`'s implicit root-find (and, for Bishop, the `diffrax` ODE solve for the parallel-transported normal) and matches a central finite difference:

```{code-block} python
>>> analytic = jax.grad(n1_of_radius)(1.0)
>>> h = 1e-5
>>> numeric = (n1_of_radius(1.0 + h) - n1_of_radius(1.0 - h)) / (2 * h)
>>> round(float(analytic), 10), round(float(numeric), 10)
(-1.0015358319, -1.0015358319)

```

`x` is an observed point, `radius` (or any other differentiable curve parameter) is what you're solving for, and `n1_of_radius` is the residual whose gradient a fit needs.

The builder you fit through matters for how much that gradient costs, not only for the metric's cross terms below: Bishop's gradient goes through `diffrax`'s `DirectAdjoint` (see `bishop.py`'s _Choosing an adjoint_ section), which is roughly three orders of magnitude slower per point than Frenet–Serret's closed-form gradient, and that cost scales roughly linearly with the number of points. Prefer Frenet–Serret for gradient-based fitting over thousands of points when the metric's torsion cross-terms don't matter for the fit.

## The Metric

No `metric_matrix` rule is registered for `TubularChart` — it has no closed form better than the Jacobian pullback $g = J^\top J$, so it falls through to `coordinax`'s generic Euclidean rule, unchanged by which builder supplies the triad. The two builders still give visibly different metrics, because the Jacobian itself differs. A **torsion-carrying** curve is required to see it: a planar curve has zero torsion, and Frenet–Serret is diagonal there too, same as Bishop.

```{code-block} python
>>> from coordinaxs.api.manifolds import metric_matrix

>>> def helix(tau):
...     t = tau.ustrip("s")
...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")

>>> at = {"tau": u.Q(0.7, "s"), "n1": u.Q(0.13, "km"), "n2": u.Q(-0.21, "km")}

>>> ch_b = cxfc.TubularChart(cxfc.BishopBuilder(helix), tau_bounds=HELIX_BOUNDS)
>>> g_b = metric_matrix(ch_b.M, at, ch_b).matrix
>>> bool(jnp.abs(g_b[0, 1].ustrip("km / s")) < 1e-8)  # Bishop: no d(tau).d(n1) cross term
True

>>> ch_f = cxfc.TubularChart(cxfc.FrenetSerretBuilder(helix), tau_bounds=HELIX_BOUNDS)
>>> g_f = metric_matrix(ch_f.M, at, ch_f).matrix
>>> bool(jnp.abs(g_f[0, 1].ustrip("km / s")) < 1e-8)  # Frenet-Serret: torsion cross term
False

```

At this point, on this curve, Bishop's off-diagonal is $\sim 2\times10^{-12}$ (zero to floating-point precision — Bishop is rotation-minimising, so there is no twist in the normal plane to couple $\tau$ to $n_1, n_2$), and Frenet–Serret's is $\sim 6\times10^{-2}$ — a real cross term coming from the curve's torsion. Ten orders of magnitude apart, same curve, same point:

```{code-block} python
>>> float(g_b[0, 1].ustrip("km / s")) < 1e-10
True
>>> float(g_f[0, 1].ustrip("km / s")) > 1e-2
True

```

Prefer Bishop for this chart when the metric matters and the diagonal structure is convenient — it holds regardless of torsion.

## Arc-Length Reparametrisation

`coordinaxs.curveframes` provides `ArcLength`, which wraps a curve $\gamma(\tau)$ and returns $s \mapsto \gamma(\tau(s))$ with $\|\gamma'(s)\| = 1$ everywhere, by solving $d\tau/ds = 1/\|\gamma'(\tau)\|$ rather than integrating speed and inverting it. Because a curve is consumed purely as a callable throughout `coordinaxs.curveframes`, `ArcLength(curve)` is itself a curve: it wraps into `BishopBuilder` or `FrenetSerretBuilder`, and from there into `TubularChart`, unchanged.

```{code-block} python
>>> arc = cxfc.ArcLength(helix)
>>> ch_arc = cxfc.TubularChart(
...     cxfc.BishopBuilder(arc, "km"), tau_bounds=(u.Q(0.0, "km"), u.Q(5.0, "km"))
... )

```

A builder over an arc-length curve takes a **length** `tau_unit` — `"km"` above, not `"s"` — because the wrapped curve's parameter now is arc length rather than time. `tau_bounds` follow: they are lengths too, and `coord_dimensions` reports it:

```{code-block} python
>>> ch_arc.coord_dimensions
('length', 'length', 'length')

```

### The Payoff

On the curve itself ($n_1 = n_2 = 0$), $g_{ss}$ reduces to the textbook tubular-coordinate form $(1-k_1n_1-k_2n_2)^2 = 1$ — no speed factor. On the plain $\tau$-parameterised `helix`, $g_{\tau\tau}$ still carries the curve's squared speed, $1 + 0.3^2$:

```{code-block} python
>>> at0 = {"tau": u.Q(0.7, "s"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}
>>> g_plain = metric_matrix(ch_b.M, at0, ch_b).matrix
>>> f"{g_plain[0, 0].ustrip('km2 / s2'):.10f}"
'1.0900000000'

>>> at_arc = {"tau": u.Q(1.3, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}
>>> g_arc = metric_matrix(ch_arc.M, at_arc, ch_arc).matrix
>>> f"{g_arc[0, 0].ustrip(''):.10f}"
'1.0000000000'

```

No metric code changes between the two calls: `metric_matrix` falls through to the same generic Jacobian pullback described in [The Metric](#the-metric) either way. The only difference is which curve the chart wraps.

### Time-Dependent Curves: Eulerian Versus Lagrangian

Over a two-argument curve $\gamma(\tau, t)$, `ArcLength` stays two-argument: `ArcLength(curve)(s, t)` measures arc length on the slice at `t` — the slice being evaluated. This is the **Eulerian** reading: a label is the arc length of the _current_ curve, so a fixed material point's label drifts as the curve moves.

`LagrangianArcLength(curve, t0)` measures arc length on the fixed reference slice `t0` instead, always — never on the `t` supplied at call time — then evaluates the wrapped curve at that _supplied_ `t`. A label therefore names the same material point at every `t`, but it stops being the current arc length once the curve has moved.

The two readings agree wherever the curve moves rigidly — a rotation or translation leaves every arc length untouched — and differ only where the curve stretches or compresses. A uniformly stretching line, $\gamma(\tau, t) = (\tau(1+0.5t), 0, 0)$, makes the difference concrete: at label $s=1\,\mathrm{km}$, $t=1\,\mathrm{s}$, the Eulerian reading gives one unit of arc length along the current (already-stretched) line; the Lagrangian reading gives the position of the material point that was one unit along the line at $t_0=0$:

```{code-block} python
>>> def stretch(tau, t):
...     x = tau.ustrip("s") * (1.0 + 0.5 * t.ustrip("s"))
...     z = jnp.zeros_like(x)
...     return u.Q(jnp.stack([x, z, z]), "km")

>>> eulerian = cxfc.AtTime(cxfc.ArcLength(stretch), u.Q(1.0, "s"))
>>> eulerian(u.Q(1.0, "km"))
Q([1., 0., 0.], 'km')

>>> lagrangian = cxfc.AtTime(
...     cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s")), u.Q(1.0, "s")
... )
>>> lagrangian(u.Q(1.0, "km"))
Q([1.5, 0. , 0. ], 'km')

```

`AtTime(curve, t)` binds the evaluation time of a two-argument curve, turning it into a one-argument one; it is what makes `ArcLength`'s otherwise-two-argument result callable with `s` alone above. Where `AtTime` sits relative to `ArcLength` changes what is being asked: `ArcLength(AtTime(curve, t))` binds `t` first, so `ArcLength` sees a one-argument curve and freezes arc length to that one slice permanently — there is no Eulerian/Lagrangian distinction left to make. `AtTime(ArcLength(curve), t)`, used above, keeps `ArcLength` two-argument and only fixes which slice a given call reads; a later call with a different `t` re-measures on that slice instead.

### Cost

Every call into `ArcLength` or `LagrangianArcLength` solves the reparametrisation ODE from $s=0$ to the requested $s$. Under `BishopBuilder` that sits inside Bishop's own parallel-transport solve, which evaluates the curve many times per call, so the costs multiply: measured on a helix, a forward `pt_map` is ~46x slower than over the un-reparametrised curve, and ~86x under `jax.grad`. Frenet--Serret is far cheaper, having no ODE of its own.

Amortising that with a precomputed $\tau(s)$ interpolation is [tracked separately](https://github.com/GalacticDynamics/coordinax/issues) -- doing it without breaking gradients with respect to the curve's own parameters needs more than caching the solve.

## Limitations

The builders themselves stay $\tau$-parameterised, not unit-speed: $g_{\tau\tau}$ carries a $\|\gamma'\|^2$ speed factor by default, as seen in [The Metric](#the-metric) above. Wrap the curve in `ArcLength` first (see [Arc-Length Reparametrisation](#arc-length-reparametrisation)) to get a unit-speed $\tau$ instead.

**`tau_bounds` must not span more than one period for a closed curve.** `tau_bounds` seeds the inverse's coarse scan (see [Both Directions](#both-directions) above); for a closed curve, $\gamma(\tau)$ and $\gamma(\tau + \text{period})$ are the same ambient point, so a range wider than one period turns the nearest-point solve into an exact tie:

```{code-block} python
>>> float(jnp.max(jnp.abs((circle(u.Q(0.7, "s")) - circle(u.Q(0.7 + 2 * jnp.pi, "s"))).ustrip("km"))))
0.0

```

A bounds range spanning two periods recovers the _wrong_ branch for a point that was originally at $\tau=0.7\,\mathrm{s}$:

```{code-block} python
>>> wide = cxfc.TubularChart(cxfc.BishopBuilder(circle), tau_bounds=(u.Q(0.0, "s"), u.Q(4 * jnp.pi, "s")))
>>> on_curve = circle(u.Q(0.7, "s"))
>>> d = {"x": on_curve[0], "y": on_curve[1], "z": on_curve[2]}
>>> recovered = cxc.pt_map(d, wide.M, cxc.cart3d, wide.M, wide)["tau"]
>>> bool(jnp.allclose(recovered.ustrip("s"), 0.7, atol=1e-6))
False

```

Even a correctly one-period range still has the two endpoints coincide (they are the same seam point); the scan's tie-break resolves that seam to the lower bound, not the upper one:

```{code-block} python
>>> seam = circle(u.Q(0.0, "s"))
>>> d_seam = {"x": seam[0], "y": seam[1], "z": seam[2]}
>>> recovered_seam = cxc.pt_map(d_seam, ch_bishop.M, cxc.cart3d, ch_bishop.M, ch_bishop)["tau"]
>>> bool(jnp.allclose(recovered_seam.ustrip("s"), 0.0, atol=1e-8))
True

```

`n_seed` (default 64) sets how finely that same scan samples `tau_bounds` before the root-find polish; it is what makes the inverse pick the _nearest_ point rather than merely _a_ stationary one. On a curve that doubles back on itself, too coarse a scan can pick the wrong basin outright and lock onto the wrong branch — raise `n_seed` if the curve has tight folds relative to the sampling density `tau_bounds` implies. The polish itself is confined to one seed spacing either side of the scan's chosen basin, so once the scan has the right basin the polish cannot leave it for a competing stationary point (a local _maximum_ of the distance also satisfies the stationarity condition, and an unconstrained polish could converge there instead — see `nearest_tau`'s docstring for a worked counterexample).

**The chart is injective only _locally_ inside the curve's reach, and that is checked only when you ask for it.** Past the _focal distance_ — where the normal-plane offset cancels the local curvature exactly — nearby ambient points map to the same $(\tau, n_1, n_2)$. `check_data(..., values=True)` raises rather than return coordinates that aren't unique, but `values` defaults to `False`, and every `check_data` call inside `coordinax` itself passes `values=False` or omits it. The inverse `pt_map` does not call `check_data` at all, so it happily returns focal-point coordinates without complaint — the guard below is opt-in, something _you_ must call, not a protection the chart applies for you. On the unit circle with `BishopBuilder`, $+\mathbf{U}_1$ points outward, so the Jacobian factor is $1+n_1$ and the focal point sits at $n_1=-1.0$ — one radius toward the center:

```{code-block} python
>>> at_inside = {"tau": u.Q(0.7, "s"), "n1": u.Q(0.2, "km"), "n2": u.Q(0.0, "km")}
>>> float(ch_bishop.jacobian_factor(at_inside)) > 0
True

>>> at_focal = {"tau": u.Q(0.0, "s"), "n1": u.Q(-1.0, "km"), "n2": u.Q(0.0, "km")}
>>> bool(jnp.abs(ch_bishop.jacobian_factor(at_focal)) < 1e-6)
True

>>> at_outside = {"tau": u.Q(0.0, "s"), "n1": u.Q(-1.6, "km"), "n2": u.Q(0.0, "km")}
>>> ch_bishop.check_data(at_outside, values=True)
Traceback (most recent call last):
    ...
ValueError: point lies outside the reach of the curve: the tubular coordinates are not locally injective there

```

The factor is a _local_ test only: it says nothing about a point mirrored across the curve at the same offset (same factor, same ambient point, different $\tau$), and nothing about the curve's global self-approach distance either — a passing factor does not rule either out.

**The inverse does not detect a point whose nearest curve point lies outside `tau_bounds`.** The coarse scan is confined to `tau_bounds`; if the bracketed polish above finds no root there (because the true nearest point is further out), `nearest_tau` falls back to an unconstrained root-find from the scan's edge, which can walk the solution arbitrarily far outside `tau_bounds`. The result is a finite $\tau$ outside `tau_bounds`, with residual near zero and no error or `NaN` -- clipping `tau` to the bounds is not a fix, since it would break the stationarity condition the inverse solves. A helix queried well past the end of its intended range shows this:

```{code-block} python
>>> far = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(100.0, "km")}
>>> got_far = cxc.pt_map(far, ch_f.M, cxc.cart3d, ch_f.M, ch_f)
>>> float(got_far["tau"].ustrip("s"))
333.3333333333333

```

`HELIX_BOUNDS` is `(-1, 6)` s, so `333.33` is nowhere near it. Callers who cannot rule out querying past the fitted range should check the returned `tau` against `tau_bounds` themselves -- that converged-but-out-of-bounds case is not an error.

A genuinely degenerate query is different, and _is_ caught: a point equidistant from the whole closed circle -- its centre -- leaves the stationarity condition satisfied everywhere, not by any particular $\tau$, so that same fallback root-find does not converge (its step divides by a derivative that vanishes identically). `nearest_tau` checks the solver's own result and raises rather than return an arbitrary, meaningless $\tau$:

```{code-block} python
>>> centre = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
>>> cxc.pt_map(centre, ch_frenet.M, cxc.cart3d, ch_frenet.M, ch_frenet)
Traceback (most recent call last):
    ...
RuntimeError: nearest-point solve did not converge

```

That check cannot see the periodic-aliasing case from [Limitations](#limitations) above -- a wide `tau_bounds` still converges, just to the wrong branch -- so a one-period `tau_bounds` remains the caller's responsibility for closed curves.

### Coordinate Data Must Be Scalar Per Point

Unlike the stock charts, `TubularChart` does not accept batched coordinate arrays: the inverse solve and `jacobian_factor` are written for a scalar `tau`, and `jax.jacfwd` over a batched one would build an N x N Jacobian. Passing arrays raises rather than returning something wrong.

Use `jax.vmap` over single points, which works and is the fast path anyway -- the inverse batches well, dropping from ~40 us to well under 1 us per point between one query and a thousand:

```{code-block} python
>>> import jax

>>> tubular = cxfc.TubularChart(
...     cxfc.BishopBuilder(circle),
...     tau_bounds=(u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s")),
... )
>>> def to_cart(tau, n1):
...     p = {"tau": u.Q(tau, "s"), "n1": u.Q(n1, "km"), "n2": u.Q(0.0, "km")}
...     return cxc.pt_map(p, tubular.M, tubular, tubular.M, cxc.cart3d)["x"]

>>> jax.vmap(to_cart)(jnp.array([0.5, 1.0]), jnp.array([0.1, 0.2]))
Q([0.96534082, 0.64836277], 'km')
```

:::{seealso}

[Working With Charts](charts.md)

{doc}`Working With Curve Frames <../packages/coordinaxs.curveframes/guide>`

:::
