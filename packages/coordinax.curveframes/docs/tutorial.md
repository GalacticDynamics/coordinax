# Tutorial: Frenet–Serret Curve Frames

This tutorial walks through a complete example: define a space curve, attach a Frenet–Serret frame, transform points into and out of the curve frame, chain through multiple frames, and leverage JAX for JIT compilation and vectorisation.

**Prerequisites**: [Working With Curve Frames](guide.md), [Working With Frames](../../../docs/guides/frames.md).

```pycon
>>> import jax
>>> import jax.numpy as jnp
>>> import numpy as np

>>> import quaxed.numpy as qnp
>>> import unxt as u

>>> import coordinax.frames as cxf
>>> import coordinax.transforms as cxfm
>>> import coordinax.curveframes as cxfc
```

## Step 1: Define a Space Curve

We start with a circle of radius 1 km in the $xy$-plane, parameterised by $\tau$ in seconds:

```pycon
>>> def circle(tau):
...     t = tau.ustrip("s")
...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")
...
```

Evaluate it to confirm:

```pycon
>>> circle(u.Q(0.0, "s"))
Q([1., 0., 0.], 'km')

>>> result = circle(u.Q(jnp.pi / 2, "s"))
>>> np.testing.assert_allclose(result.value, [0.0, 1.0, 0.0], atol=1e-6)
```

At $\tau = 0$ the curve is at $(1, 0, 0)$ km; at $\tau = \pi/2$ it is at $(0, 1, 0)$ km.

## Step 2: Build the Transform

`FrenetSerretTransform.from_curve` computes the Frenet–Serret frame fields automatically via JAX autodiff:

```pycon
>>> fs_xform = cxfc.FrenetSerretTransform.from_curve(circle)
>>> fs_xform
FrenetSerretTransform(...)
```

Check the frame at $\tau = 0$. For a unit-speed circle, $\mathbf{T}$ points along $+y$, $\mathbf{N}$ points along $-x$, and $\mathbf{B}$ points along $+z$:

```pycon
>>> tau0 = u.Q(0.0, "s")
>>> fs_xform.location(tau0)
Q([1., 0., 0.], 'km')

>>> T = fs_xform.tangent(tau0)
>>> np.testing.assert_allclose(T.value, [0, 1, 0], atol=1e-6)

>>> N = fs_xform.normal(tau0)
>>> np.testing.assert_allclose(N.value, [-1, 0, 0], atol=1e-6)

>>> B = fs_xform.binormal(tau0)
>>> np.testing.assert_allclose(B.value, [0, 0, 1], atol=1e-6)
```

## Step 3: Build a FrenetSerretFrame relative to Alice

Attach the transform to Alice's frame:

```pycon
>>> fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle)
>>> fs_frame.base_frame
Alice()

>>> isinstance(fs_frame.xop, cxfc.FrenetSerretTransform)
True
```

## Step 4: Transform Points — Alice to Curve Frame

Get the frame-transition operator and apply it. A point sitting at the curve origin $\gamma(0) = (1, 0, 0)$ km should map to the origin of the curve frame:

```pycon
>>> op_to_fs = cxf.frame_transition(cxf.Alice(), fs_frame)

>>> p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
>>> p_fs = cxfm.act(op_to_fs, tau0, p)
>>> np.testing.assert_allclose(p_fs.value, [0.0, 0.0, 0.0], atol=1e-6)
```

A point offset by 1 km in the $+x$ direction from the curve has $\delta = (1, 0, 0)$. In the curve frame $R\,\delta = (T \cdot \delta,\; N \cdot \delta,\; B \cdot \delta) = (0, -1, 0)$:

```pycon
>>> p2 = u.Q(jnp.array([2.0, 0.0, 0.0]), "km")
>>> p2_fs = cxfm.act(op_to_fs, tau0, p2)
>>> np.testing.assert_allclose(p2_fs.value, [0.0, -1.0, 0.0], atol=1e-6)
```

## Step 5: Round-Trip — Curve Frame Back to Alice

The reverse operator recovers the original ambient coordinates:

```pycon
>>> op_from_fs = cxf.frame_transition(fs_frame, cxf.Alice())
>>> p_back = cxfm.act(op_from_fs, tau0, p2_fs)
>>> np.testing.assert_allclose(p_back.value, p2.value, atol=1e-6)
```

## Step 6: Evaluate at Different $\tau$ Values

The frame moves along the curve. At $\tau = \pi/2$, the curve is at $(0, 1, 0)$ with $\mathbf{T} = (-1, 0, 0)$. A point at the new curve origin maps to $(0, 0, 0)$:

```pycon
>>> tau_quarter = u.Q(jnp.pi / 2, "s")
>>> p_quarter = u.Q(jnp.array([0.0, 1.0, 0.0]), "km")
>>> p_q_fs = cxfm.act(op_to_fs, tau_quarter, p_quarter)
>>> np.testing.assert_allclose(p_q_fs.value, [0.0, 0.0, 0.0], atol=1e-5)
```

Different $\tau$ values give different results for the same ambient point:

```pycon
>>> r0 = cxfm.act(op_to_fs, u.Q(0.0, "s"), p2)
>>> r1 = cxfm.act(op_to_fs, u.Q(1.0, "s"), p2)
>>> assert not np.allclose(r0.value, r1.value, atol=1e-3)
```

## Step 7: Three-Frame Chain — Alice ↔ FrenetSerret ↔ Alex

Curve frames compose with any coordinax reference frame. Here we chain `Alice → FS(τ) → Alex` and verify the round-trip:

```pycon
>>> tau = u.Q(0.3, "s")
>>> p = u.Q(jnp.array([5.0, -2.0, 1.0]), "km")
```

Forward path: Alice → curve frame → Alex:

```pycon
>>> op_a_to_fs = cxf.frame_transition(cxf.Alice(), fs_frame)
>>> op_fs_to_alex = cxf.frame_transition(fs_frame, cxf.Alex())

>>> p_fs = cxfm.act(op_a_to_fs, tau, p)
>>> p_alex = cxfm.act(op_fs_to_alex, tau, p_fs)
```

Reverse path: Alex → curve frame → Alice:

```pycon
>>> op_alex_to_fs = cxf.frame_transition(cxf.Alex(), fs_frame)
>>> op_fs_to_a = cxf.frame_transition(fs_frame, cxf.Alice())

>>> p_fs2 = cxfm.act(op_alex_to_fs, tau, p_alex)
>>> p_recovered = cxfm.act(op_fs_to_a, tau, p_fs2)

>>> np.testing.assert_allclose(p_recovered.value, p.value, atol=1e-5)
```

The full chain `Alice → FS → Alex → FS → Alice` is the identity for any point and any $\tau$.

## Step 8: JIT Compilation

Because the transform fields are pure-function closures (static PyTree leaves), JIT compilation works directly:

```pycon
>>> op = cxf.frame_transition(cxf.Alice(), fs_frame)
>>> p_test = u.Q(jnp.array([2.0, 0.0, 0.0]), "km")

>>> @jax.jit
... def transform_at(tau, p):
...     return cxfm.act(op, tau, p)
...

>>> result_eager = cxfm.act(op, tau0, p_test)
>>> result_jit = transform_at(tau0, p_test)
>>> np.testing.assert_allclose(result_jit.value, result_eager.value, atol=1e-10)
```

## Step 9: Vectorize Over $\tau$ with vmap

Map a scalar-$\tau$ function over a batch of parameter values:

```pycon
>>> taus = u.Q(jnp.linspace(0.0, 2.0, 5), "s")
>>> trajectory = jax.vmap(lambda t: cxfm.act(op, t, p_test))(taus)
>>> trajectory.shape
(5, 3)
```

Each row is the curve-frame representation of `p_test` at a different $\tau$. Combine with `jit` for maximum throughput:

```pycon
>>> trajectory_fast = jax.jit(jax.vmap(lambda t: cxfm.act(op, t, p_test)))(taus)
>>> np.testing.assert_allclose(trajectory_fast.value, trajectory.value, atol=1e-10)
```

## Summary

| Step | What you did                                          |
| ---- | ----------------------------------------------------- |
| 1    | Defined a circle curve `tau -> Quantity[(3,)]`        |
| 2    | Built `FrenetSerretTransform` via `from_curve`        |
| 3    | Wrapped it in a `FrenetSerretFrame` relative to Alice |
| 4–5  | Transformed points to/from the curve frame            |
| 6    | Evaluated at different $\tau$ values                  |
| 7    | Chained Alice ↔ FrenetSerret ↔ Alex                   |
| 8–9  | JIT-compiled and vmapped over $\tau$                  |

The key idea: **$\tau$ is not stored on the frame**. It is passed at evaluation time, making the frame object lightweight and fully compatible with JAX transformations.

---

# Tutorial: Bishop (Rotation-Minimising) Curve Frames

This tutorial shows how to use the **Bishop frame** — the rotation-minimising alternative to Frenet–Serret. You will see how it handles straight lines gracefully and how to compare both frames on the same curve.

**Prerequisites**: [Working With Curve Frames](guide.md), the Frenet–Serret tutorial above.

```pycon
>>> import coordinax.curveframes as cxfc
>>> import coordinax.frames as cxf
>>> import unxt as u
>>> import jax
>>> import jax.numpy as jnp
>>> import quaxed.numpy as qnp
>>> import numpy as np
```

## Step 1: Bishop Frame on a Helix

Define a helix with a vertical drift:

```pycon
>>> def helix(tau: u.Q) -> u.Q:
...     t = tau.ustrip("s")
...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")
...
```

Build the Bishop transform:

```pycon
>>> bt = cxfc.BishopTransform.from_curve(helix)
>>> bt
BishopTransform(...)
```

Evaluate the triad at $\tau = 0$:

```pycon
>>> tau0 = u.Q(0.0, "s")
>>> loc = bt.location(tau0)
>>> loc
Q([1., 0., 0.], 'km')

>>> T = bt.tangent(tau0)
>>> U1 = bt.normal1(tau0)
>>> U2 = bt.normal2(tau0)
```

Verify orthonormality:

```pycon
>>> np.testing.assert_allclose(jnp.dot(T.value, U1.value), 0.0, atol=1e-5)
>>> np.testing.assert_allclose(jnp.dot(T.value, U2.value), 0.0, atol=1e-5)
>>> np.testing.assert_allclose(jnp.dot(U1.value, U2.value), 0.0, atol=1e-5)

>>> np.testing.assert_allclose(jnp.linalg.norm(T.value), 1.0, atol=1e-5)
>>> np.testing.assert_allclose(jnp.linalg.norm(U1.value), 1.0, atol=1e-5)
>>> np.testing.assert_allclose(jnp.linalg.norm(U2.value), 1.0, atol=1e-5)
```

Right-handedness: $\mathbf{U}_2 = \mathbf{T} \times \mathbf{U}_1$:

```pycon
>>> np.testing.assert_allclose(U2.value, jnp.cross(T.value, U1.value), atol=1e-5)
```

## Step 2: Handling a Straight Line

The Frenet–Serret frame is **singular** on a straight line (zero curvature). The Bishop frame handles it cleanly:

```pycon
>>> def line(tau):
...     t = tau.ustrip("s")
...     return u.Q(jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")
...

>>> bt_line = cxfc.BishopTransform.from_curve(line)
```

The normals are well-defined unit vectors at any $\tau$:

```pycon
>>> U1 = bt_line.normal1(u.Q(5.0, "s"))
>>> np.testing.assert_allclose(jnp.linalg.norm(U1.value), 1.0, atol=1e-5)

>>> U2 = bt_line.normal2(u.Q(5.0, "s"))
>>> np.testing.assert_allclose(jnp.linalg.norm(U2.value), 1.0, atol=1e-5)

>>> T = bt_line.tangent(u.Q(5.0, "s"))
>>> np.testing.assert_allclose(jnp.dot(T.value, U1.value), 0.0, atol=1e-5)
```

## Step 3: Build a Bishop Frame and Transform Points

Attach the Bishop transform to Alice's frame:

```pycon
>>> b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), helix)
>>> b_frame.base_frame
Alice()

>>> isinstance(b_frame.xop, cxfc.BishopTransform)
True
```

Transform a point at the curve origin to the Bishop frame:

```pycon
>>> op_to_b = cxf.frame_transition(cxf.Alice(), b_frame)
>>> p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")  # gamma(0) = (1, 0, 0)
>>> p_b = cxfm.act(op_to_b, tau0, p)
>>> np.testing.assert_allclose(p_b.value, [0.0, 0.0, 0.0], atol=1e-5)
```

## Step 4: Round-Trip Verification

```pycon
>>> op_from_b = cxf.frame_transition(b_frame, cxf.Alice())
>>> p_back = cxfm.act(op_from_b, tau0, p_b)
>>> np.testing.assert_allclose(p_back.value, p.value, atol=1e-5)
```

## Step 5: Custom Initial Normal and $\tau_0$

Specify an explicit initial normal:

```pycon
>>> bt_custom = cxfc.BishopTransform.from_curve(
...     helix, initial_normal=jnp.array([0.0, 0.0, 1.0])
... )
>>> U1_custom = bt_custom.normal1(tau0)
>>> np.testing.assert_allclose(jnp.linalg.norm(U1_custom.value), 1.0, atol=1e-5)
```

Shift the reference parameter:

```pycon
>>> bt_shifted = cxfc.BishopTransform.from_curve(helix, tau_0=u.Q(1.0, "s"))
>>> bt_shifted.tau_0
Q(1., 's')
```

## Step 6: Inverse and Double-Inverse

```pycon
>>> bt_inv = bt.inverse
>>> bt_inv
BishopTransform(...)
```

Double-inversion recovers the original:

```pycon
>>> loc_orig = bt.location(tau0)
>>> loc_double = bt.inverse.inverse.location(tau0)
>>> np.testing.assert_allclose(loc_double.value, loc_orig.value, atol=1e-5)
```

## Step 7: Three-Frame Chain — Alice ↔ Bishop ↔ Alex

```pycon
>>> tau = u.Q(0.3, "s")
>>> p = u.Q(jnp.array([5.0, -2.0, 1.0]), "km")
```

Forward: Alice → Bishop → Alex:

```pycon
>>> op_a_to_b = cxf.frame_transition(cxf.Alice(), b_frame)
>>> op_b_to_alex = cxf.frame_transition(b_frame, cxf.Alex())

>>> p_b = cxfm.act(op_a_to_b, tau, p)
>>> p_alex = cxfm.act(op_b_to_alex, tau, p_b)
```

Reverse: Alex → Bishop → Alice:

```pycon
>>> op_alex_to_b = cxf.frame_transition(cxf.Alex(), b_frame)
>>> op_b_to_a = cxf.frame_transition(b_frame, cxf.Alice())

>>> p_b2 = cxfm.act(op_alex_to_b, tau, p_alex)
>>> p_recovered = cxfm.act(op_b_to_a, tau, p_b2)

>>> np.testing.assert_allclose(p_recovered.value, p.value, atol=1e-5)
```

## Step 8: JIT and vmap

```pycon
>>> op = cxf.frame_transition(cxf.Alice(), b_frame)
>>> p_test = u.Q(jnp.array([2.0, 0.0, 0.0]), "km")

>>> @jax.jit
... def transform_at(tau, p):
...     return cxfm.act(op, tau, p)
...

>>> result_eager = cxfm.act(op, tau0, p_test)
>>> result_jit = transform_at(tau0, p_test)
>>> np.testing.assert_allclose(result_jit.value, result_eager.value, atol=1e-10)
```

vmap over $\tau$:

```pycon
>>> taus = u.Q(jnp.linspace(0.0, 2.0, 5), "s")
>>> trajectory = jax.vmap(lambda t: cxfm.act(op, t, p_test))(taus)
>>> trajectory.shape
(5, 3)
```

## Summary

| Step | What you did                                 |
| ---- | -------------------------------------------- |
| 1    | Built a `BishopTransform` on a helix         |
| 2    | Verified it works on a straight line         |
| 3    | Built a `BishopFrame` and transformed points |
| 4    | Verified the round-trip                      |
| 5    | Used custom initial normal and $\tau_0$      |
| 6    | Tested inverse and double-inverse            |
| 7    | Chained Alice ↔ Bishop ↔ Alex                |
| 8    | JIT-compiled and vmapped                     |

The Bishop frame is ideal when your curve has zero-curvature regions or when you need a twist-free frame. For curves with non-vanishing curvature, both Bishop and Frenet–Serret produce valid orthonormal frames — they differ by a $\tau$-dependent rotation in the normal plane.
