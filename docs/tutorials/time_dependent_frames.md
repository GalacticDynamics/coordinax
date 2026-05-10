# Time-Dependent Frame Transformations

This tutorial walks through building a rotating reference frame whose relationship to an inertial frame changes over time. You will learn how to:

- Make a transform's parameters depend on time via a callable field
- Wrap a time-dependent operator in a `TransformedReferenceFrame`
- Apply `frame_transition` and `act` with an evolution parameter $\tau$
- JIT-compile and vectorize over time with JAX

**Prerequisites**: [Working With Frames](../guides/frames.md) and [Working With Transforms](../guides/transforms.md).

## The Scenario

Earth spins about its z-axis at angular velocity $\omega \approx 7.27 \times 10^{-5}\;\text{rad}\,\text{s}^{-1}$ (one full rotation every 24 hours). An observatory on Earth's surface lives in the **body frame** — a frame that rotates with Earth (analogous to ECEF). A distant star is fixed in the inertial frame. We want to compute the star's coordinates as seen from the observatory at an arbitrary time $t$.

```pycon
>>> import coordinax.frames as cxf
>>> import coordinax.transforms as cxfm
>>> import coordinax.vectors as cxv
>>> import coordinax.charts as cxc
>>> import unxt as u
>>> import quaxed.numpy as jnp
>>> import jax
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

Now we make the rotation angle grow linearly with time. Earth rotates at $\omega = 2\pi / 86\,400\;\text{rad}\,\text{s}^{-1}$ — one full revolution per sidereal day.

The key idea: instead of passing a numeric matrix to `Rotate`, pass a **callable** $\tau \to \text{matrix}$. Coordinax calls this callable at every `act` invocation, passing the time parameter.

```pycon
>>> OMEGA = 2 * jnp.pi / 86400

>>> def earth_rotation_matrix(tau):
...     """Rotation matrix R_z(omega * t) for Earth's body frame at time tau."""
...     theta = OMEGA * tau.ustrip("s")  # extract numeric seconds
...     ct, st = jnp.cos(theta), jnp.sin(theta)
...     return jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
...
```

Wrap it in a `Rotate` operator. Because the argument is callable, it is stored as-is — not converted to an array:

```pycon
>>> rotating_op = cxfm.Rotate(earth_rotation_matrix)
```

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
Point(
    {'x': Q(1., 'kpc'), 'y': Q(0., 'kpc'), 'z': Q(0., 'kpc')},
    chart=Cart3D(M=Rn(3)), manifold=Rn(3)
)
```

After 6 hours (quarter turn, $90°$ rotation), the star lies along the body frame's $-y$ axis:

```pycon
>>> tau_quarter = u.Q(jnp.pi / (2 * OMEGA), "s")  # 90° rotation — 6 hours
>>> star_at_quarter = cxfm.act(xform, tau_quarter, star)
>>> star_at_quarter
Point(
    {'x': Q(6.123234e-17, 'kpc'), 'y': Q(1., 'kpc'), 'z': Q(0., 'kpc')},
    chart=Cart3D(M=Rn(3)), manifold=Rn(3)
)
```

After 12 hours (half turn, $180°$ rotation), the star appears at $[-1, 0, 0]$:

```pycon
>>> tau_half = u.Q(jnp.pi / OMEGA, "s")  # 180° rotation — 12 hours
>>> star_at_half = cxfm.act(xform, tau_half, star)
>>> star_at_half
Point(
    {'x': Q(-1., 'kpc'), 'y': Q(1.2246468e-16, 'kpc'), 'z': Q(0., 'kpc')},
    chart=Cart3D(M=Rn(3)), manifold=Rn(3)
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
Point(
    {'x': Q(1., 'kpc'), 'y': Q(0., 'kpc'), 'z': Q(0., 'kpc')},
    chart=Cart3D(M=Rn(3)), manifold=Rn(3)
)
```

## Step 7: JIT Compilation

Because `rotating_op` stores `earth_rotation_matrix` as a **static** field (functions are not JAX arrays), the operator itself is a valid static PyTree leaf. The numeric $\tau$ and the coordinate data are the only dynamic parts, so JIT compilation works cleanly.

Manifold, chart, and representation types are registered as static JAX nodes, so `@jax.jit` works directly with `Point` inputs:

```python
star_q = u.Q([1.0, 0.0, 0.0], "kpc")


@jax.jit
def star_in_body_frame(tau):
    return cxfm.act(xform, tau, star_q)


star_jit = star_in_body_frame(u.Q(1.0, "s"))
```

Subsequent calls reuse the compiled code and pay only the XLA execution cost:

```python
star_jit_2 = star_in_body_frame(u.Q(3600.0, "s"))  # 1 hour later
star_jit_5 = star_in_body_frame(u.Q(21600.0, "s"))  # 6 hours later
```

## Step 8: Vectorizing Over Time

`jax.vmap` maps a scalar-time function over a batch of times. Combined with `jit`, this gives an efficient trajectory:

```python
times = u.Q(jnp.linspace(0.0, 86400.0, 200), "s")  # one full day, 200 samples

trajectory = jax.jit(jax.vmap(star_in_body_frame))(times)
# trajectory has shape (200,) of Quantity([x, y, z], 'kpc')
```

The body-frame x-component traces a cosine; the y-component traces a sine:

```python
xs = jnp.stack([trajectory[i].ustrip("kpc")[0] for i in range(3)])
# xs ≈ [cos(ω·t) for the first three time steps]
```

## Step 9: Composing a Moving Rotating Frame

Real problems often combine rotation **and** translation. Suppose we also account for Earth's orbital motion around the Sun: its centre moves along its orbit at roughly $29.78\;\text{km}\,\text{s}^{-1}$.

```python
ORBITAL_VELOCITY = 29.78  # km/s — Earth's mean orbital speed


def orbit_offset(tau):
    """Displacement of Earth's centre along its orbit at time tau."""
    return {
        "x": u.Q(ORBITAL_VELOCITY * tau.ustrip("s"), "km"),
        "y": u.Q(0.0, "km"),
        "z": u.Q(0.0, "km"),
    }


orbital_shift = cxfm.Translate(orbit_offset, chart=cxc.cart3d)
```

Compose: translate first (move the origin along the orbit), then rotate (spin Earth's axes). Evaluation order for `|` is right-to-left, so `rotating_op | orbital_shift` applies `orbital_shift` first:

```python
combined_op = rotating_op | orbital_shift

orbiting_body_frame = cxf.TransformedReferenceFrame(inertial, combined_op)
xform_combined = cxf.frame_transition(inertial, orbiting_body_frame)
```

Compute the star's position in Earth's orbiting, rotating body frame at $t = 1\;\text{s}$:

```python
tau_1s = u.Q(1.0, "s")
star_combined = cxfm.act(xform_combined, tau_1s, star_q)
```

## Summary

| Step                  | Code                                                |
| --------------------- | --------------------------------------------------- |
| Static frame rotation | `TransformedReferenceFrame(base, Rotate(R_matrix))` |
| Time-dep. rotation    | `Rotate(callable_t_to_matrix)`                      |
| Time-dep. translation | `Translate(callable_t_to_dict, chart=...)`          |
| Build frame           | `TransformedReferenceFrame(inertial, rotating_op)`  |
| Get transition        | `xform = frame_transition(from_frame, to_frame)`    |
| Apply at time $t$     | `act(xform, tau, vector)`                           |
| Inspect at time t     | `materialize_transform(op, tau)`                    |
| Invert                | `frame_transition(to_frame, from_frame)`            |
| JIT (Quantity)        | `@jax.jit` on `tau` + coordinate                    |
| Batch over times      | `jax.vmap(fn)(times)`                               |
| Compose ops           | `rot_op \| translate_op` (translate first)          |
