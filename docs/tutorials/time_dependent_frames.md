# Time-Dependent Frame Transformations

This tutorial walks through building a rotating reference frame whose relationship to an inertial frame changes over time. You will learn how to:

- Make a transform's parameters depend on time via `Parametric`
- Wrap a time-dependent operator in a `TransformedReferenceFrame`
- Apply `frame_transition` and `act` with an evolution parameter $\tau$
- JIT-compile and vectorize over time (and over the physical rate) with JAX
- Differentiate an observed position with respect to a physical parameter

**Prerequisites**: [Working With Frames](../guides/frames.md) and [Working With Transforms](../guides/transforms.md).

## The Scenario

Earth spins about its z-axis at angular velocity $\omega \approx 7.29 \times 10^{-5}\;\text{rad}\,\text{s}^{-1}$ — one full rotation every sidereal day ($23.9345\;\text{h}$, the time for the stars to return to the same position, slightly shorter than the 24 h solar day). An observatory on Earth's surface lives in the **body frame** — a frame that rotates with Earth (analogous to ECEF). A distant star is fixed in the inertial frame. We want to compute the star's coordinates as seen from the observatory at an arbitrary time $t$.

```pycon
>>> import coordinax.frames as cxf
>>> import coordinax.transforms as cxfm
>>> import coordinax.vectors as cxv
>>> import coordinax.charts as cxc
>>> import unxt as u
>>> import quaxed.numpy as jnp
>>> import jax
>>> import equinox as eqx
```

## Step 1: Review — Static Frame Transition

Before adding time dependence, recall how a static `TransformedReferenceFrame` works. We define a frame that is rotated $30°$ around $z$ relative to the inertial frame and check the transition.

```pycon
>>> theta_static = jnp.pi / 6  # 30 degrees
>>> R_static = jnp.array(
...     [
...         [jnp.cos(theta_static), -jnp.sin(theta_static), 0.0],
...         [jnp.sin(theta_static), jnp.cos(theta_static), 0.0],
...         [0.0, 0.0, 1.0],
...     ]
... )

>>> inertial = cxf.alice
>>> rotated_30deg = cxf.TransformedReferenceFrame(inertial, cxfm.Rotate(R_static))
```

`frame_transition` returns the operator that transforms coordinates from one frame into another:

```pycon
>>> op_static = cxf.frame_transition(inertial, rotated_30deg)
```

Apply it to a star at [1, 0, 0] kpc in the inertial frame. Because `TransformedReferenceFrame` uses active semantics, `op_static` is exactly the stored `xop` — here `Rotate(R_static)`:

```pycon
>>> star_inertial = cxv.Point.from_([1, 0, 0], "kpc")
>>>
>>> # tau=None for a time-independent transform
>>> star_rotated = cxfm.act(op_static, None, star_inertial)
```

Inverting the transition takes us back:

```pycon
>>> op_back = cxf.frame_transition(rotated_30deg, inertial)
>>> star_recovered = cxfm.act(op_back, None, star_rotated)
```

## Step 2: The Time-Dependent Rotation

Now we make the rotation angle grow linearly with time, using Earth's real sidereal rotation rate.

The key idea: instead of passing a numeric matrix to `Rotate`, wrap a **builder** in `Parametric`. Coordinax calls the builder at every `act` invocation, passing the time parameter, and builds a fresh `Rotate` at that instant. `cxfm.RotationAboutAxis(omega, axis=...)` is the built-in builder for exactly this — uniform rotation about a fixed axis:

```pycon
>>> SIDEREAL_DAY = u.Q(23.9345, "hr")
>>> omega = u.Q(360.0, "deg") / SIDEREAL_DAY
>>> axis = jnp.array([0.0, 0.0, 1.0])

>>> rotating_op = cxfm.Parametric(cxfm.RotationAboutAxis(omega, axis=axis))
```

Unlike a hand-written closure, `omega` here is an ordinary field of `rotating_op.builder` — a pytree leaf. That is what makes it differentiable and `vmap`-able later in this tutorial, without any special handling.

## Step 3: Build the TransformedReferenceFrame

```pycon
>>> body_frame = cxf.TransformedReferenceFrame(inertial, rotating_op)
```

`body_frame` now carries the time-dependent operator. The frame object itself is a JAX PyTree (an equinox Module), so it can be stored, passed to JIT, and vmapped over.

Get the transition operator from the inertial frame to the body frame:

```pycon
>>> xform = cxf.frame_transition(inertial, body_frame)
```

## Step 4: Apply the Transition at Specific Times

Define the star's position in the inertial frame:

```pycon
>>> star = cxv.Point.from_([1.0, 0.0, 0.0], "kpc")
```

Compute where the star appears in the body frame at $t = 0$:

```pycon
>>> tau_0 = u.Q(0.0, "s")
>>> star_at_t0 = cxfm.act(xform, tau_0, star)
>>> star_at_t0
Point({'x': Q(1., 'kpc'), 'y': Q(0., 'kpc'), 'z': Q(0., 'kpc')}, chart=Cart3D(M=Rn(3)))
```

A quarter of a sidereal day later ($\approx 5.98\;\text{h}$, a $90°$ rotation), the star lies along the body frame's $-y$ axis. We compute the quarter-turn time directly from `omega` rather than hard-coding it:

```pycon
>>> omega_rad_s = omega.ustrip("rad/s")
>>> tau_quarter = u.Q(jnp.pi / (2 * omega_rad_s), "s")  # 90° rotation
>>> star_at_quarter = cxfm.act(xform, tau_quarter, star)
>>> star_at_quarter
Point(
    {'x': Q(6.123234e-17, 'kpc'), 'y': Q(1., 'kpc'), 'z': Q(0., 'kpc')},
    chart=Cart3D(M=Rn(3))
)
```

Half a sidereal day later ($\approx 11.97\;\text{h}$, a $180°$ rotation), the star appears at $[-1, 0, 0]$:

```pycon
>>> tau_half = u.Q(jnp.pi / omega_rad_s, "s")  # 180° rotation
>>> star_at_half = cxfm.act(xform, tau_half, star)
>>> star_at_half
Point(
    {'x': Q(-1., 'kpc'), 'y': Q(1.2246468e-16, 'kpc'), 'z': Q(0., 'kpc')},
    chart=Cart3D(M=Rn(3))
)
```

## Step 6: Invert the Transition

`frame_transition(body_frame, inertial)` gives the inverse operator:

```pycon
>>> xform_inv = cxf.frame_transition(body_frame, inertial)
```

Apply it to recover the star's inertial coordinates from the body-frame coordinates:

```pycon
>>> star_back = cxfm.act(xform_inv, tau_quarter, star_at_quarter)
>>> star_back
Point({'x': Q(1., 'kpc'), 'y': Q(0., 'kpc'), 'z': Q(0., 'kpc')}, chart=Cart3D(M=Rn(3)))
```

## Step 7: JIT Compilation and Structural Caching

Because `rotating_op.builder` is an `equinox.Module` whose field `omega` is an ordinary pytree leaf (not a Python closure), `jax.jit` and `eqx.filter_jit` key their compilation cache on the operator's **structure** — the pytree treedef — not on the identity of the Python object. Two operators built from the same builder type retrace only once, no matter what rotation rate they carry:

```python
star_q = u.Q([1.0, 0.0, 0.0], "kpc")


def transition_for(omega):
    op = cxfm.Parametric(cxfm.RotationAboutAxis(omega, axis=axis))
    frame = cxf.TransformedReferenceFrame(inertial, op)
    return cxf.frame_transition(inertial, frame)


traces = []


@eqx.filter_jit
def star_in_frame(xform, tau):
    traces.append(1)
    return cxfm.act(xform, tau, star_q)


xform_a = transition_for(omega)
xform_b = transition_for(2 * omega)  # Earth spinning twice as fast

star_a = star_in_frame(xform_a, u.Q(3600.0, "s"))
star_b = star_in_frame(xform_b, u.Q(3600.0, "s"))

assert len(traces) == 1  # one trace serves both rotation rates
```

Compare this with the old `Rotate(callable)` design it replaces: a fresh Python closure is a fresh object identity, so every new rotation rate forced a fresh trace. A builder's numeric fields being pytree leaves is what buys structural caching.

**The array-leaf hashing cliff.** This structural caching relies on `xform` being passed _into_ the jitted function as a traced argument. It breaks if a builder holding array leaves is instead jitted directly as the callable — a common mistake when experimenting at the REPL:

```python
tau = u.Q(3600.0, "s")

try:
    jax.jit(rotating_op.builder)(tau)  # jitting the builder itself, not through act
except TypeError as e:
    print(f"TypeError: {e}")
```

`RotationAboutAxis.omega` and `.axis` are `jax.Array`/`Quantity` leaves, which are not hashable, so plain `jax.jit` cannot treat the builder as a static argument. `eqx.filter_jit` partitions array leaves from static leaves correctly and has no trouble:

```python
built = eqx.filter_jit(rotating_op.builder)(tau)
```

A builder whose fields are plain Python floats instead of arrays would work fine under plain `jax.jit` even in this direct-call style, because Python floats are hashable — but `eqx.filter_jit` is the safe default for any builder that might carry array leaves, since it behaves identically to `jax.jit` when every field happens to be hashable.

## Step 8: Vectorizing Over Time — and Over the Rotation Rate

`jax.vmap` maps a scalar-time function over a batch of times. Combined with `jit`, this gives an efficient trajectory:

```python
times = u.Q(jnp.linspace(0.0, 86400.0, 200), "s")  # one full day, 200 samples

trajectory = jax.jit(jax.vmap(lambda tau: star_in_frame(xform_a, tau)))(times)
# trajectory has shape (200, 3) -- x, y, z in kpc
```

Because `omega` is a pytree leaf of the builder rather than a Python closure variable, we can just as easily vectorize over a **batch of rotation rates** at fixed time — something the old callable-based design could not express at all, since a closure only ever captures one fixed value:

```python
omegas = (
    u.Q(jnp.linspace(0.5, 2.0, 5), "") * omega
)  # a fan of spin rates around Earth's
tau_fixed = u.Q(3600.0, "s")

by_rate = jax.jit(jax.vmap(lambda om: star_in_frame(transition_for(om), tau_fixed)))(
    omegas
)
# by_rate has shape (5, 3): one row per rotation rate
```

`jax.vmap` traces `transition_for` and `star_in_frame` once and produces a _batched_ operator internally — not five separate Python objects — so this is a single compiled kernel, not a Python loop.

## Step 9: Differentiating Through the Frame

The physical payoff of `Parametric` over the old `Rotate(callable)` design: because `omega` is a differentiable pytree leaf all the way from the builder through `Rotate`'s matrix construction, `jax.grad` differentiates straight through frame construction and `act`. No special-casing, no finite differences.

Here we ask: how sensitive is the star's observed $y$-coordinate (in the body frame, one hour in) to Earth's spin rate?

```python
def star_y_in_body_frame(omega_deg_per_hr):
    om = u.Q(omega_deg_per_hr, "deg/hr")
    op = cxfm.Parametric(cxfm.RotationAboutAxis(om, axis=axis))
    frame = cxf.TransformedReferenceFrame(inertial, op)
    xf = cxf.frame_transition(inertial, frame)
    out = cxfm.act(xf, u.Q(3600.0, "s"), star_q)
    return out.ustrip("kpc")[1]


omega0 = 360.0 / 23.9345  # deg/hr, Earth's sidereal rate
dy_domega = jax.grad(star_y_in_body_frame)(omega0)
assert jnp.isfinite(dy_domega)
```

`dy_domega` is the derivative of the observed position with respect to a real physical parameter (Earth's angular rate) — useful, for instance, in fitting an unknown spin rate to observed star positions via gradient descent. Under the old design, `omega` was captured inside a Python closure passed to `Rotate`; being a trace-time constant, it was invisible to `jax.grad` and no such derivative could be taken at all.

## Step 10: Composing a Moving Rotating Frame

Real problems often combine rotation **and** translation. Suppose we also account for Earth's orbital motion around the Sun. Earth's orbit is (approximately) circular, so — unlike the uniform straight-line motion that `UniformTranslation` covers — this calls for a hand-written builder, this tutorial's own "write your own builder" moment:

```python
class OrbitalDrift(eqx.Module):
    """Displacement of Earth's centre along a circular heliocentric orbit."""

    radius: u.AbstractQuantity  # orbital radius, e.g. 1 AU
    omega_orbit: u.AbstractQuantity  # orbital angular frequency

    def __call__(self, tau):
        theta = (self.omega_orbit * tau).ustrip("rad")
        delta = {
            "x": self.radius * jnp.sin(theta),
            "y": self.radius * (1 - jnp.cos(theta)),
            "z": self.radius * 0.0,
        }
        return cxfm.Translate(delta, chart=cxc.cart3d)


ONE_AU = u.Q(1.495978707e8, "km")
ORBITAL_PERIOD = u.Q(365.25 * 86400.0, "s")
omega_orbit = u.Q(2 * jnp.pi, "rad") / ORBITAL_PERIOD

orbital_shift = cxfm.Parametric(OrbitalDrift(ONE_AU, omega_orbit))
```

At $\tau = 0$ the orbit builder returns a zero displacement (`sin(0) = 0`, `1 - cos(0) = 0`); as $\tau$ grows, Earth's centre traces out the circle, with initial velocity $\text{radius} \times \omega_\text{orbit} \approx 29.78\;\text{km}\,\text{s}^{-1}$ tangent to the orbit — matching Earth's known mean orbital speed. Because `radius` and `omega_orbit` are ordinary pytree leaves of `OrbitalDrift`, this builder is just as differentiable and `vmap`-able as `RotationAboutAxis` was above.

Compose: translate first (move the origin along the orbit), then rotate (spin Earth's axes). Evaluation order for `|` is left-to-right, like a Unix shell pipe, so `orbital_shift | rotating_op` applies `orbital_shift` first:

```python
combined_op = orbital_shift | rotating_op

orbiting_body_frame = cxf.TransformedReferenceFrame(inertial, combined_op)
xform_combined = cxf.frame_transition(inertial, orbiting_body_frame)
```

Compute the star's position in Earth's orbiting, rotating body frame at $t = 1\;\text{s}$:

```python
tau_1s = u.Q(1.0, "s")
star_combined = cxfm.act(xform_combined, tau_1s, star_q)
```

## Summary

| Step | Code |
| --- | --- |
| Static frame rotation | `TransformedReferenceFrame(base, Rotate(R_matrix))` |
| Time-dep. rotation (built-in) | `Parametric(RotationAboutAxis(omega, axis=...))` |
| Time-dep. translation (custom) | `Parametric(my_eqx_module_builder)` |
| Build frame | `TransformedReferenceFrame(inertial, rotating_op)` |
| Get transition | `xform = frame_transition(from_frame, to_frame)` |
| Apply at time $t$ | `act(xform, tau, vector)` |
| Inspect at time t | `materialize_transform(op, tau)` |
| Invert | `frame_transition(to_frame, from_frame)` |
| JIT with structural caching | `eqx.filter_jit` on a function taking `(xform, tau)` as arguments |
| Avoid the array-leaf cliff | `eqx.filter_jit`, not plain `jax.jit`, on a builder/operator directly |
| Batch over times | `jax.vmap(fn)(times)` |
| Batch over the rate itself | `jax.vmap(lambda om: ...)(omegas)` |
| Differentiate w.r.t. the rate | `jax.grad(fn)(omega_value)` |
| Compose ops | `translate_op \| rot_op` (translate first) |
