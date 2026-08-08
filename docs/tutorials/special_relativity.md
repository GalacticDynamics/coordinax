# Special Relativity: Minkowski Spacetime

This tutorial works through special relativity in `coordinax`, using a real measurement as the thread: **why cosmic-ray muons reach the ground when, classically, they should not.** You will learn how to:

- Represent events on `MinkowskiManifold` with the `MinkowskiCT` chart
- Measure with `interval` instead of `separation`, and why that swap is forced
- Classify pairs of events as timelike, null, or spacelike
- Change inertial frame with `LorentzBoost`
- See which quantities are frame-dependent and which are invariant

**Prerequisites**: [Working With Charts](../guides/charts.md) and [Working With Manifolds](../guides/manifolds.md).

```pycon
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm
>>> import coordinax.representations as cxr
>>> import coordinax.transforms as cxfm
>>> import unxt as u
>>> import quaxed.numpy as jnp
```

## The Scenario

Muons are created about 15 km up, when cosmic rays strike the upper atmosphere. A muon at rest decays with a mean lifetime of about $2.2\;\mu\text{s}$, and the ones we care about move at $\beta = v/c \approx 0.998$.

Multiply those together the naive way and a muon covers

$$ v\,\tau_0 \approx 0.998 \times c \times 2.2\;\mu\text{s} \approx 660\;\text{m}, $$

so essentially none should reach sea level. Detectors at sea level find plenty of them. Special relativity is what closes the gap, and we can watch it close.

## Step 1: The Chart

Minkowski spacetime is `minkowski4d`, and its canonical chart is `minkowskict`, with components $(ct, x, y, z)$.

```pycon
>>> M = cxm.minkowski4d
>>> M
MinkowskiManifold(ndim=4)

>>> cxc.minkowskict.components
('ct', 'x', 'y', 'z')
```

Note that **all four components carry length units**, not just the spatial ones. The time coordinate is $ct$ — a time multiplied by the speed of light. This is not a quirk of the implementation; it is what makes the metric dimensionless and lets a boost be an ordinary dimensionless matrix. It also means "one second of time" is written as one light-second of $ct$.

The metric carries the signature that does all the work:

```pycon
>>> M.metric.signature
(-1, 1, 1, 1)
```

That single minus sign is the entire difference between this tutorial and Euclidean geometry.

## Step 2: Why `separation` Is Not the Tool

Let us set up the muon's birth and death as two events. We work first in the **muon's own rest frame**, where it does not move: it is born at the origin and decays $2.2\;\mu\text{s}$ later at the same place.

```pycon
>>> c = u.Q(299792458.0, "m/s")
>>> tau0 = u.Q(2.2, "us")
>>> ct0 = (c * tau0).uconvert("m")
>>> round(float(ct0.ustrip("m")), 1)
659.5

>>> def event(ct, x, y=0.0, z=0.0):
...     return {"ct": u.Q(ct, "m"), "x": u.Q(x, "m"), "y": u.Q(y, "m"), "z": u.Q(z, "m")}
...

>>> birth = event(0.0, 0.0)
>>> death = event(float(ct0.ustrip("m")), 0.0)
```

The reflex from Euclidean geometry is to ask for the `separation` between them. That does not work here, and `coordinax` says so rather than handing back a number:

```pycon
>>> try:
...     cxm.separation(cxc.minkowskict, birth, death)
... except NotImplementedError as e:
...     print(str(e)[:50])
...
separation() supports only positive-definite metri
```

The refusal is not squeamishness. `separation` is $\sqrt{\Delta x^\top G\, \Delta x}$, and with a $(-1,1,1,1)$ signature that quadratic form goes **negative** whenever the time difference dominates. The square root of a negative number is `nan`, which would be a wrong answer wearing the costume of a right one.

## Step 3: The Interval

The quantity that _is_ defined is the same form without the square root — the **invariant interval**:

$$ \Delta s^2 = -(c\Delta t)^2 + \Delta x^2 + \Delta y^2 + \Delta z^2 $$

```pycon
>>> ds2 = cxm.interval(cxc.minkowskict, birth, death)
>>> round(float(ds2.ustrip("m2")))
-434998
```

Negative, as promised. Its **sign** is physically meaningful, and `causal_character` reads it off.

Note the namespace: `interval` lives directly on `coordinax.manifolds` because the signed quadratic form is defined for _every_ metric, but the verbs that read its sign need a timelike direction, so they live in `cxm.lorentzian`. (Named for the signature rather than "spacetime" because `galileanct` is a spacetime too, and is not Lorentzian.)

```pycon
>>> cxm.lorentzian.causal_character(cxc.minkowskict, birth, death)
'timelike'
```

Timelike means a slower-than-light observer can be present at both events — here, obviously, the muon itself. The three cases:

| $\Delta s^2$ | name | meaning |
| --- | --- | --- |
| $< 0$ | timelike | one observer can attend both events |
| $= 0$ | null | only a light ray connects them |
| $> 0$ | spacelike | no observer attends both; neither causes the other |

```pycon
>>> cxm.lorentzian.causal_character(cxc.minkowskict, birth, event(1.0, 5.0))
'spacelike'
>>> cxm.lorentzian.causal_character(cxc.minkowskict, birth, event(3.0, 3.0))
'null'
```

For a **timelike** pair, the magnitude is the elapsed **proper time** — what a wristwatch carried between the two events would read. Since we built these events from the muon's lifetime, we should get it back:

```pycon
>>> tau = cxm.lorentzian.proper_time(cxc.minkowskict, birth, death)
>>> round(float(tau.uconvert("us").ustrip("us")), 3)
2.2
```

For a **spacelike** pair the magnitude is a proper distance instead, and asking for the wrong one is an error rather than a silent `nan`:

```pycon
>>> round(
...     float(
...         cxm.lorentzian.proper_distance(cxc.minkowskict, birth, event(3.0, 5.0)).ustrip(
...             "m"
...         )
...     ),
...     3,
... )
4.0

>>> try:
...     cxm.lorentzian.proper_time(cxc.minkowskict, birth, event(1.0, 5.0))
... except ValueError as e:
...     print(str(e)[:52])
...
proper_time() is defined only for timelike-separated
```

## Step 4: Changing Frames with a Boost

So far everything is in the muon's rest frame. The atmosphere is 15 km thick in **our** frame, so to compare we must change frames — which is what a `LorentzBoost` does.

The boost parameter is the dimensionless $\boldsymbol\beta = \mathbf{v}/c$. That is the chart-native choice: since the chart already measures time in length units, no speed-of-light constant is needed here at all.

```pycon
>>> boost = cxfm.LorentzBoost([0.998, 0.0, 0.0])
>>> round(float(boost.gamma), 2)
15.82
```

$\gamma \approx 15.8$ is the whole story in one number. There is also `rapidity`, the parameter that _adds_ under composition where velocities do not:

```pycon
>>> round(float(boost.rapidity), 3)
3.453
```

### An aside: why a boost is not "time-dependent"

`coordinax` marks transforms whose point action varies with the evolution parameter $\tau$. A Lorentz boost is **not** one of them, which surprises people who know that its Galilean cousin is:

```pycon
>>> boost.is_time_dependent
False
>>> galilean = cxfm.Boost(
...     {"x": jnp.asarray(1.0), "y": jnp.asarray(0.0), "z": jnp.asarray(0.0)},
...     chart=cxc.cart3d,
... )
>>> galilean.is_time_dependent
True
```

The difference is _where time lives_. For `Boost`, time is a parameter outside the manifold and the action $x \mapsto x + \Delta v\,\tau$ genuinely depends on it. For `LorentzBoost`, $ct$ is a **coordinate of the manifold** — time is already inside the vector being transformed — so $\Lambda$ is just a constant matrix. That is why `act` above takes `None` for $\tau$.

An _accelerating_ frame, where the rapidity itself grows with $\tau$, is built by wrapping a builder in `TimeDep`:

```pycon
>>> import equinox as eqx

>>> class UniformlyAccelerating(eqx.Module):
...     rate: jnp.ndarray
...     def __call__(self, tau):
...         return cxfm.LorentzBoost(self.rate * tau)
...

>>> accelerating = cxfm.TimeDep(UniformlyAccelerating(jnp.asarray([0.1, 0.0, 0.0])))
>>> accelerating.is_time_dependent
True
```

Now transform the muon's death event into the frame where the muon is moving. The `None` in the second slot is the time parameter $\tau$, and a boost does not use it — see the note below.

```pycon
>>> death_lab = cxfm.act(boost, None, death, cxc.minkowskict, cxr.point)
>>> round(float(death_lab["ct"].ustrip("m")))
10434
>>> round(float(death_lab["x"].uconvert("km").ustrip("km")), 2)
10.41
```

Two things changed. In this frame the muon **travelled 10.4 km**, not zero — of course, it is moving here. And the elapsed coordinate time grew:

```pycon
>>> elapsed = (death_lab["ct"] / c).uconvert("us")
>>> round(float(elapsed.ustrip("us")), 1)
34.8
```

$34.8\;\mu\text{s}$, against the $2.2\;\mu\text{s}$ we started with — a factor of $\gamma = 15.82$. This is **time dilation**, and it is what saves the muon: 10.4 km of travel gets it most of the way down through a 15 km atmosphere, where the naive 660 m would not have gotten it out of the stratosphere.

## Step 5: What Did _Not_ Change

The coordinates moved, the elapsed time moved, the distance travelled moved. It is easy to come away thinking everything is relative. The interval is not:

```pycon
>>> birth_lab = cxfm.act(boost, None, birth, cxc.minkowskict, cxr.point)
>>> ds2_lab = cxm.interval(cxc.minkowskict, birth_lab, death_lab)
>>> round(float(ds2_lab.ustrip("m2")))
-434998
```

The same $-434998\;\text{m}^2$ we computed in the rest frame. And therefore so is the proper time:

```pycon
>>> tau_lab = cxm.lorentzian.proper_time(cxc.minkowskict, birth_lab, death_lab)
>>> round(float(tau_lab.uconvert("us").ustrip("us")), 3)
2.2
```

**The muon still ages 2.2 μs.** It has to: that is a reading on a physical clock, and no choice of coordinates can change what a clock says. What changed is how much of _our_ coordinate time that corresponds to.

The causal character is likewise absolute — no boost can turn a timelike pair spacelike, which is why relativity does not let you reorder cause and effect:

```pycon
>>> cxm.lorentzian.causal_character(cxc.minkowskict, birth_lab, death_lab)
'timelike'
```

This invariance is the defining property of a Lorentz transformation, $\Lambda^\top \eta \Lambda = \eta$, and you can check it directly on the matrix:

```pycon
>>> eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
>>> lam = boost._raw_matrix
>>> bool(jnp.allclose(lam.T @ eta @ lam, eta, atol=1e-5))
True
```

## Step 6: Relativity of Simultaneity

One more frame-dependent thing, and the least intuitive. Take two events that are simultaneous in one frame but happen in different places:

```pycon
>>> here = event(0.0, 0.0)
>>> there = event(0.0, 1.0)
```

Both at $ct = 0$: simultaneous. After a boost they are not:

```pycon
>>> here_b = cxfm.act(boost, None, here, cxc.minkowskict, cxr.point)
>>> there_b = cxfm.act(boost, None, there, cxc.minkowskict, cxr.point)
>>> round(float(there_b["ct"].ustrip("m")), 6) != round(float(here_b["ct"].ustrip("m")), 6)
True
```

"At the same time" is not a property of a pair of events; it is a property of a pair of events _and a frame_. Note this is only possible because the pair is **spacelike** separated — for a timelike pair the ordering is fixed, so causality survives:

```pycon
>>> cxm.lorentzian.causal_character(cxc.minkowskict, here, there)
'spacelike'
```

## Step 7: Velocities Do Not Add

A last piece of bookkeeping worth seeing. Boost by $0.6c$, then by $0.6c$ again. The answer is not $1.2c$:

```pycon
>>> b1 = cxfm.LorentzBoost([0.6, 0.0, 0.0])
>>> combined = b1._raw_matrix @ b1._raw_matrix
>>> round(float(combined[0, 1] / combined[0, 0]), 4)
0.8824
```

$0.882c$ — the relativistic velocity-addition formula $(\beta_1 + \beta_2)/(1 + \beta_1\beta_2)$. You can never reach $c$ by composing subluminal boosts.

Rapidity is the parameterisation that _does_ add, which is why it exists:

```pycon
>>> r1 = cxfm.LorentzBoost.from_rapidity(0.3)
>>> r2 = cxfm.LorentzBoost.from_rapidity(0.5)
>>> r3 = cxfm.LorentzBoost.from_rapidity(0.8)
>>> bool(jnp.allclose(r2._raw_matrix @ r1._raw_matrix, r3._raw_matrix, atol=1e-5))
True
```

## Summary

| quantity                         | frame-dependent? |
| -------------------------------- | ---------------- |
| coordinates $(ct, x, y, z)$      | ✅ yes           |
| elapsed coordinate time          | ✅ yes           |
| distance travelled               | ✅ yes           |
| simultaneity of spacelike events | ✅ yes           |
| **interval** $\Delta s^2$        | ❌ **invariant** |
| **proper time**                  | ❌ **invariant** |
| **causal character**             | ❌ **invariant** |

The practical rule: reach for `interval`, `proper_time`, and `causal_character` when you want a statement about _physics_, and read coordinates only when you genuinely want a statement about a particular frame. `separation` and `norm` belong to the Riemannian world and will refuse Minkowski input rather than let you mix the two up.

## See Also

- [Working With Manifolds](../guides/manifolds.md) — metrics and signatures
- [Working With Transforms](../guides/transforms.md) — the operator machinery `LorentzBoost` is built on
- [Working With Frames](../guides/frames.md) — the transformation-group taxonomy, including the Lorentz and Poincaré groups
