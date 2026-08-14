# Tutorial: Bring Your Own Curve

A curve is consumed purely as a callable throughout `coordinaxs.curveframes` — `AbstractCurveFrameBuilder.__call__` and `.location` are the only two call sites that touch it. Nothing requires a plain function: an `equinox.Module` works exactly the same way, and gets differentiable curve parameters for free, because its fields are pytree leaves. This tutorial builds one `equinox.Module` curve for each of the parametrisation shapes `ArcLength` and the builders accept, then differentiates through one of them.

**Prerequisites**: {doc}`Working With Curve Frames <guide>`, {doc}`Working With Curve Charts <curve-charts>`.

```pycon
>>> import equinox as eqx
>>> import jax
>>> import jax.numpy as jnp

>>> import unxt as u

>>> import coordinaxs.curveframes as cxfc
>>> from coordinaxs.api.manifolds import metric_matrix
```

A `speed_at` helper reused from Step 2 onward: the norm of the curve's Jacobian at a given arc-length station, for curves whose parameter is already a length.

```pycon
>>> def speed_at(curve, s_val):
...     return float(
...         jnp.linalg.norm(jax.jacfwd(lambda v: curve(u.Q(v, "km")).ustrip("km"))(s_val))
...     )
...
```

## Step 1: Parametrised by Time

A circle whose radius is a pytree field, parametrised by time:

```pycon
>>> class Circle(eqx.Module):
...     radius: u.AbstractQuantity
...     def __call__(self, tau):
...         t = tau.ustrip("s")
...         r = self.radius.ustrip("km")
...         return u.Q(jnp.stack([r * jnp.cos(t), r * jnp.sin(t), jnp.zeros_like(t)]), "km")
...

>>> time_circle = Circle(radius=u.Q(2.0, "km"))
```

Its speed equals the radius, not 1 — a time parametrisation has no reason to be unit-speed:

```pycon
>>> speed_t = float(
...     jnp.linalg.norm(jax.jacfwd(lambda t: time_circle(u.Q(t, "s")).ustrip("km"))(0.7))
... )
>>> round(speed_t, 6)
2.0
```

`ArcLength(time_circle, "s")` reparametrises it by arc length; `"s"` is `time_circle`'s own parameter unit, not the arc length's. The wrapped curve feeds a builder and a chart exactly like any other curve, and the metric's $g_{ss}$ component comes out exactly 1 on the curve:

```pycon
>>> arc_time = cxfc.ArcLength(time_circle, "s")
>>> ch_time = cxfc.TubularChart(
...     cxfc.BishopBuilder(arc_time, "km"), tau_bounds=(u.Q(0.0, "km"), u.Q(10.0, "km"))
... )
>>> at_time = {"tau": u.Q(1.3, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}
>>> round(float(metric_matrix(ch_time.M, at_time, ch_time).matrix[0, 0].ustrip("")), 6)
1.0
```

## Step 2: Parametrised by Arc Length

A circle already parametrised by arc length needs no wrapper:

```pycon
>>> class ArcCircle(eqx.Module):
...     radius: u.AbstractQuantity
...     def __call__(self, s):
...         v = s.ustrip("km")
...         r = self.radius.ustrip("km")
...         return u.Q(
...             jnp.stack([r * jnp.cos(v / r), r * jnp.sin(v / r), jnp.zeros_like(v)]), "km"
...         )
...

>>> arc_circle = ArcCircle(radius=u.Q(2.0, "km"))
>>> round(speed_at(arc_circle, 1.3), 6)
1.0
```

It goes straight into a builder, with no `ArcLength` wrap:

```pycon
>>> ch_direct = cxfc.TubularChart(
...     cxfc.BishopBuilder(arc_circle, "km"),
...     tau_bounds=(u.Q(0.0, "km"), u.Q(4 * jnp.pi, "km")),
... )
>>> at_direct = {"tau": u.Q(1.3, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}
>>> round(
...     float(metric_matrix(ch_direct.M, at_direct, ch_direct).matrix[0, 0].ustrip("")), 6
... )
1.0
```

`ArcLength`'s `tau_unit` has no default, so wrapping `arc_circle` without one fails at construction, before any call:

```pycon
>>> try:
...     cxfc.ArcLength(arc_circle)
... except TypeError as e:
...     print(e)
...
missing a required argument: 'tau_unit'
```

With the unit supplied, `arc_circle`'s own parameter is already a length, so `ArcLength(arc_circle, "km")` reproduces it to ODE solver tolerance, not exactly:

```pycon
>>> arc_on_arc = cxfc.ArcLength(arc_circle, "km")
>>> diff = float(
...     jnp.max(
...         jnp.abs(
...             arc_circle(u.Q(1.3, "km")).ustrip("km")
...             - arc_on_arc(u.Q(1.3, "km")).ustrip("km")
...         )
...     )
... )
>>> diff < 1e-8
True
```

## Step 3: A Two-Argument Module $\gamma(s, t)$

A module whose `__call__` takes both a station and a time is unit-speed in $s$ at every slice, without needing `ArcLength` at any slice. `AtTime` binds one slice into a one-argument curve, reading `inspect.signature` on the bound `__call__` the same way it reads a plain function's:

```pycon
>>> class SeriesCircle(eqx.Module):
...     def __call__(self, s, t):
...         r = 2.0 + t.ustrip("s")
...         v = s.ustrip("km")
...         return u.Q(
...             jnp.stack([r * jnp.cos(v / r), r * jnp.sin(v / r), jnp.zeros_like(v)]), "km"
...         )
...

>>> combo = cxfc.AtTime(SeriesCircle(), u.Q(1.0, "s"))
>>> round(speed_at(combo, 1.3), 6)
1.0
```

## Step 4: Backed by Sampled Data

A module reconstructed from knots and positions via `jnp.interp` — the shape a per-timestep fit to a stream simulator produces. A chord between two knots is shorter than the arc it approximates, so the interpolant falls measurably short of unit speed between knots — here on a circle of radius 2 km sampled at 400 evenly spaced knots:

```pycon
>>> class SampledCurve(eqx.Module):
...     knots: jax.Array
...     xs: jax.Array
...     ys: jax.Array
...     zs: jax.Array
...     def __call__(self, s):
...         v = s.ustrip("km")
...         return u.Q(
...             jnp.stack(
...                 [
...                     jnp.interp(v, self.knots, self.xs),
...                     jnp.interp(v, self.knots, self.ys),
...                     jnp.interp(v, self.knots, self.zs),
...                 ]
...             ),
...             "km",
...         )
...

>>> theta = jnp.linspace(0.0, 2 * jnp.pi, 401)
>>> sampled = SampledCurve(
...     knots=2.0 * theta,
...     xs=2.0 * jnp.cos(theta),
...     ys=2.0 * jnp.sin(theta),
...     zs=jnp.zeros_like(theta),
... )
>>> round(speed_at(sampled, 1.3), 6)
0.99999
```

Wrapping it in `ArcLength` restores unit speed, re-measuring against the interpolated chords rather than trusting them:

```pycon
>>> arc_sampled = cxfc.ArcLength(sampled, "km")
>>> round(speed_at(arc_sampled, 1.3), 6)
1.0
```

## Step 5: Differentiating Through a Fitted Field

`curve` is a pytree field on `ArcLength` like any other, so `jax.grad` reaches a module's own fields straight through the ODE solve. Here the gradient is with respect to `Circle`'s `radius` from Step 1:

```pycon
>>> def x_of_radius(radius_km):
...     curve = Circle(radius=u.Q(radius_km, "km"))
...     return cxfc.ArcLength(curve, "s")(u.Q(1.3, "km")).ustrip("km")[0]
...

>>> analytic = float(jax.grad(x_of_radius)(2.0))
>>> h = 1e-4
>>> numeric = float((x_of_radius(2.0 + h) - x_of_radius(2.0 - h)) / (2 * h))
>>> round(analytic, 6), round(numeric, 6)
(1.189455, 1.189455)
```

## Summary

| Step | Curve shape | How it reaches unit speed |
| --- | --- | --- |
| 1 | `__call__(tau)`, parametrised by time | `ArcLength(curve, "s")` |
| 2 | `__call__(s)`, parametrised by arc length | none needed |
| 3 | `__call__(s, t)`, unit-speed in `s` at every slice | `AtTime(curve, t)` |
| 4 | `__call__(s)`, backed by `jnp.interp` over sampled knots | `ArcLength(curve, "km")` |
| 5 | Any of the above, with a differentiable field | `jax.grad` reaches straight through |

See {ref}`Working With Curve Charts <arc-length-reparametrisation>` for the Eulerian/Lagrangian distinction on time-dependent curves, and a trap where a linear blend of two unit-speed curves is not itself unit-speed.
