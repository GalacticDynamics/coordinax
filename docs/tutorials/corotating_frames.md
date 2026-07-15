# Co-Rotating Frames and Fictitious Forces

This tutorial builds a **co-rotating reference frame** — the frame of an observer riding a spinning turntable — and shows how coordinax's kinematic prolongation reproduces the **Coriolis** and **centrifugal** forces of a first classical-mechanics course.

The companion [Time-Dependent Frames tutorial](./time_dependent_frames.md) transforms _positions_ into a rotating frame. Here we transform an entire **phase-space state** — position, velocity, _and_ acceleration together — with a single operator, and watch the fictitious forces fall out.

You will learn how to:

- Build a co-rotating frame with a time-dependent `Rotate`
- Carry a `Coordinate` bundle (point + velocity + acceleration) into that frame
- Decompose the transformed acceleration into named Coriolis and centrifugal terms
- Trace the curved path a "straight" trajectory acquires in the rotating frame
- Isolate the pure centrifugal term with a released co-rotating puck

```{admonition} The physics in one paragraph
:class: note

A turntable spins at constant angular velocity
$\boldsymbol{\Omega} = \omega\,\hat{\mathbf{z}}$. A frictionless puck feels no real force, so in the
lab (inertial) frame it slides in a **straight line**. In the turntable frame
it appears to accelerate; for a free particle that apparent acceleration is
entirely fictitious,

$$
\mathbf{a}_\text{rot}
  = \underbrace{-2\,\boldsymbol{\Omega}\times\mathbf{v}_\text{rot}}_{\text{Coriolis}}
  \;-\;\underbrace{\boldsymbol{\Omega}\times(\boldsymbol{\Omega}\times\mathbf{r}_\text{rot})}_{\text{centrifugal}},
$$

with $\mathbf{r}_\text{rot}, \mathbf{v}_\text{rot}$ the position and velocity
*in the rotating frame*. coordinax computes exactly this — as the second total
time-derivative of the transformed trajectory — for any transform you define.
```

## Step 1: Build the Co-Rotating Frame

The turntable spins counter-clockwise at $\omega = 2\ \text{rad}\,\text{s}^{-1}$ (chosen round for readable numbers). Expressing a lab point in the turntable's own axes rotates its coordinates by $-\omega t$, so the inertial → turntable operator is $R_z(-\omega t)$.

The key move: pass a **callable** $\tau \mapsto \text{matrix}$ to `Rotate` instead of a fixed matrix. coordinax evaluates it at the time parameter on every `act`, and — crucially — differentiates through it to transform velocities and accelerations.

```pycon
>>> import jax
>>> import quaxed.numpy as jnp
>>> import unxt as u
>>> import coordinax.charts as cxc
>>> import coordinax.representations as cxr
>>> import coordinax.frames as cxf
>>> import coordinax.transforms as cxfm
>>> import coordinax.vectors as cxv
>>> from jaxtyping import Array, Real

>>> OMEGA = 2.0  # rad / s

>>> def turntable_matrix(tau) -> Real[Array, "3 3"]:
...     """Inertial -> turntable rotation R_z(-omega * t)."""
...     angle = -OMEGA * tau.ustrip("s")
...     c, s = jnp.cos(angle), jnp.sin(angle)
...     return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
...
```

Wrap it in a `Rotate` operator and attach it to an inertial base frame to make the co-rotating frame:

```pycon
>>> inertial = cxf.alice  # any inertial frame
>>> turntable = cxf.TransformedReferenceFrame(inertial, cxfm.Rotate(turntable_matrix))
>>> op = cxf.frame_transition(inertial, turntable)
```

`op` is the operator that carries data from the inertial frame into the turntable frame.

## Step 2: The Puck's Phase-Space State

A puck launched from near the center along the lab $x$-axis, with **no real force** acting on it. At $t = 0$ its state is position $\mathbf{r} = (1, 0, 0)\ \text{m}$, velocity $\mathbf{v} = (3, 0, 0)\ \text{m}\,\text{s}^{-1}$, and acceleration $\mathbf{a} = \mathbf{0}$ (free particle).

We bundle all three into a single `Coordinate`. Each fibre carries its kinematic role (`vel`, `acc`):

```pycon
>>> def phase_space(r, v, a):
...     """A Coordinate bundle {point, velocity, acceleration} in Cartesian m/s."""
...     point = cxv.Point.from_(r, "m")
...     vel = cxv.Tangent(
...         {"x": u.Q(v[0], "m/s"), "y": u.Q(v[1], "m/s"), "z": u.Q(v[2], "m/s")},
...         cxc.cart3d,
...         cxr.coord_basis,
...         cxr.vel,
...     )
...     acc = cxv.Tangent(
...         {"x": u.Q(a[0], "m/s2"), "y": u.Q(a[1], "m/s2"), "z": u.Q(a[2], "m/s2")},
...         cxc.cart3d,
...         cxr.coord_basis,
...         cxr.acc,
...     )
...     return cxv.Coordinate(point, velocity=vel, acceleration=acc)
...

>>> puck = phase_space([1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0])
```

## Step 3: Transform Into the Turntable Frame

Act the transition on the whole bundle at $t = 0$. One operator carries the point _and_ both fibres consistently:

```pycon
>>> t0 = u.Q(0.0, "s")
>>> seen = cxfm.act(op, t0, puck, usys=u.unitsystems.si)

>>> print(seen.point.round(3))
<Point: chart=Cart3D (x, y, z) [m]
    [1. 0. 0.]>

>>> print(seen["velocity"].round(3))
<Tangent: chart=Cart3D (x, y, z) [m / s]
    [ 3. -2.  0.]>

>>> print(seen["acceleration"].round(3))
<Tangent: chart=Cart3D (x, y, z) [m / s2]
    [ -4. -12.   0.]>
```

The puck was launched purely along $x$ with **zero** acceleration in the lab, yet in the turntable frame it has a velocity component along $-y$ (the frame rotating out from under it) and a large non-zero acceleration. That acceleration is entirely fictitious.

## Step 4: The Fictitious Forces

Pull the transformed vectors out as plain arrays and reconstruct the two named forces from the textbook formula, using coordinax's _own_ $\mathbf{r}_\text{rot}, \mathbf{v}_\text{rot}$:

```pycon
>>> r_rot = jnp.stack([seen.point.data[k].ustrip("m") for k in "xyz"])
>>> v_rot = jnp.stack([seen["velocity"].data[k].ustrip("m/s") for k in "xyz"])
>>> a_rot = jnp.stack([seen["acceleration"].data[k].ustrip("m/s2") for k in "xyz"])

>>> Omega = jnp.array([0.0, 0.0, OMEGA])
>>> coriolis = -2.0 * jnp.cross(Omega, v_rot)
>>> centrifugal = -jnp.cross(Omega, jnp.cross(Omega, r_rot))

>>> coriolis
Array([ -8., -12.,   0.], dtype=float64)

>>> centrifugal
Array([ 4., -0., -0.], dtype=float64)
```

The Coriolis force deflects the puck sideways (here toward $-y$); the centrifugal force pushes it radially outward (here $+x$). Their sum is exactly the acceleration coordinax computed:

```pycon
>>> bool(jnp.allclose(coriolis + centrifugal, a_rot, atol=1e-6))
True
```

coordinax never mentions "Coriolis" or "centrifugal" — it simply differentiates the point action $\mathbf{x} \mapsto R(\tau)\,\mathbf{x}$ twice along the trajectory. The named forces are what that derivative _is_. A puck of mass $m$ feels a fictitious force $\mathbf{F} = m\,\mathbf{a}_\text{rot}$.

## Step 5: The Curved Path

A straight lab trajectory becomes a curve in the turntable frame. Because the frame's parameter $\tau$ and the trajectory's time are the _same_ physical time, we must feed the puck's **advancing** state $\mathbf{r}(t) = \mathbf{r}_0 + \mathbf{v}\,t$ at each instant. `jax.vmap` sweeps over time:

```pycon
>>> r0 = jnp.array([1.0, 0.0, 0.0])
>>> v0 = jnp.array([3.0, 0.0, 0.0])

>>> def turntable_xy(t):
...     state = phase_space(r0 + v0 * t, v0, [0.0, 0.0, 0.0])
...     out = cxfm.act(op, u.Q(t, "s"), state, usys=u.unitsystems.si)
...     return jnp.stack([out.point.data[k].ustrip("m") for k in "xy"])
...

>>> ts = jnp.array([0.0, 0.2, 0.4, 0.6])
>>> jax.vmap(turntable_xy)(ts).round(3)
Array([[ 1.   ,  0.   ],
       [ 1.474, -0.623],
       [ 1.533, -1.578],
       [ 1.015, -2.61 ]], dtype=float64)
```

In the lab the puck moves straight out along $x$; in the turntable frame it veers steadily toward $-y$ — the Coriolis deflection, curving to the right for a counter-clockwise turntable.

## Step 6: Isolating the Centrifugal Term

The Coriolis force depends on the **rotating-frame** velocity $\mathbf{v}_\text{rot}$, not the lab velocity. To see pure centrifugal, we need $\mathbf{v}_\text{rot} = \mathbf{0}$: a puck momentarily _co-rotating_ with the turntable, then released. Riding at radius $r = 2\ \text{m}$, its lab velocity is $\boldsymbol{\Omega}\times\mathbf{r} = (0, 4, 0)\ \text{m}\,\text{s}^{-1}$; free after release ($\mathbf{a} = \mathbf{0}$):

```pycon
>>> released = phase_space([2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0])
>>> seen_rel = cxfm.act(op, t0, released, usys=u.unitsystems.si)

>>> print(seen_rel["velocity"].round(3))
<Tangent: chart=Cart3D (x, y, z) [m / s]
    [0. 0. 0.]>

>>> print(seen_rel["acceleration"].round(3))
<Tangent: chart=Cart3D (x, y, z) [m / s2]
    [8. 0. 0.]>
```

At the release instant the puck is at rest in the turntable frame ($\mathbf{v}_\text{rot} = \mathbf{0}$, so no Coriolis) and accelerates radially **outward** at $\omega^2 r = 4 \times 2 = 8\ \text{m}\,\text{s}^{-2}$ — the pure centrifugal term $-\boldsymbol{\Omega}\times(\boldsymbol{\Omega}\times\mathbf{r})$. This is the familiar "flung outward when the merry-go-round lets go."

## Summary

| Goal | Code |
| --- | --- |
| Co-rotating frame | `TransformedReferenceFrame(inertial, Rotate(R_z(-ωt)))` |
| Phase-space state | `Coordinate(point, velocity=Tangent(..., vel), acceleration=Tangent(..., acc))` |
| Transform state | `cxfm.act(op, tau, state, usys=...)` |
| Rotating-frame acceleration | `out["acceleration"]` — the fictitious force per unit mass |
| Sweep over time | `jax.vmap` the transition over the advancing state |

The lesson: coordinax's transforms only need a **point action** ($\mathbf{x} \mapsto R(\tau)\,\mathbf{x}$). Velocities, accelerations, and hence the Coriolis and centrifugal forces come for free from the kinematic prolongation (`act`/`prolong`) — see the [transforms guide](../guides/transforms.md#time-dependence-couples-the-ladder-kinematic-prolongation) for the machinery.
