# Working With Curve Frames

This guide introduces **curve-attached reference frames** provided by `coordinax.curveframes`. You will learn what a curve frame is, how to build one from a space curve, and how to transform coordinate data between curve frames and ordinary reference frames.

For the mathematical specification see the [curveframes spec](spec.md). For API reference on the base frame system see [Working With Frames](../../../docs/guides/frames.md) and [Working With Transforms](../../../docs/guides/transforms.md).

```python
import jax
import jax.numpy as jnp

import quaxed.numpy as qnp
import unxt as u

import coordinax.curveframes as cxfc
import coordinax.frames as cxf
import coordinax.transforms as cxfm
```

## What Is a Curve Frame?

A **curve frame** is a reference frame that rides along a smooth space curve $\gamma(\tau)$. At each value of the evolution parameter $\tau$, the frame:

- is centred at $\gamma(\tau)$, and
- has oriented axes derived from the curve's local geometry.

The most common choice is the **Frenet–Serret frame**, whose axes are the tangent $\mathbf{T}$, normal $\mathbf{N}$, and binormal $\mathbf{B}$ vectors.

Curve frames are useful whenever coordinates are most naturally expressed relative to a moving curve — particle beams along a beamline, satellites along an orbit, or galactic streams along a stellar track.

## The Frenet–Serret Transform

Before building a frame, you need a **transform** — the operator that maps ambient coordinates into curve-local coordinates. `FrenetSerretTransform` stores the curve's geometry as four $\tau$-dependent callables:

| Field      | Meaning                                  |
| ---------- | ---------------------------------------- |
| `location` | curve position $\gamma(\tau)$            |
| `tangent`  | unit tangent $\mathbf{T}(\tau)$          |
| `normal`   | unit principal normal $\mathbf{N}(\tau)$ |
| `binormal` | unit binormal $\mathbf{B}(\tau)$         |

All four fields are **lazy**: they are callables, not pre-evaluated arrays. Computation happens only when you evaluate the transform at a concrete $\tau$.

### Building a Transform from a Curve

Define a curve as a function `tau -> Quantity[(3,)]` and pass it to `from_curve`:

```python
def helix(tau: u.Q) -> u.Q:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


fs_transform = cxfc.FrenetSerretTransform.from_curve(helix)
```

`from_curve` uses `unxt.experimental.jacfwd` to compute unit-correct first and second derivatives of the curve, then builds the tangent, normal, and binormal closures via Gram–Schmidt orthogonalisation.

Evaluate the fields at a specific $\tau$:

```python
tau = u.Q(0.0, "s")
fs_transform.location(tau)  # Q([1., 0., 0.], 'km')
fs_transform.tangent(tau)  # unit vector along curve velocity
```

### The `tau_unit` Parameter

By default `from_curve` assumes $\tau$ has units of seconds. If your parameter has different units (e.g. radians, years), pass `tau_unit`:

```python
fs_rad = cxfc.FrenetSerretTransform.from_curve(helix, tau_unit="rad")
```

This affects only the automatic differentiation step — the returned callables still accept any `Quantity` with compatible dimensions.

### Inversion

Every `FrenetSerretTransform` has an `.inverse` that reverses the mapping:

```python
fs_inv = fs_transform.inverse
```

The inverse is also a `FrenetSerretTransform` with its own callable fields. Double-inversion recovers the original: `fs_transform.inverse.inverse` reconstructs the forward transform cleanly (no closure accumulation).

## Building a Frenet–Serret Frame

A `FrenetSerretFrame` pairs a `FrenetSerretTransform` with a **base frame** — the ambient reference frame in which the curve is defined.

### Direct Construction

```python
fs_frame = cxfc.FrenetSerretFrame(
    base_frame=cxf.Alice(),
    xop=fs_transform,
    xop_inv=fs_transform.inverse,
)
```

### Convenience Constructor

`from_curve` combines both steps — it builds the transform and wraps it:

```python
fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), helix)
```

This is equivalent to the direct construction above.

### Fields

`FrenetSerretFrame` inherits two fields from `AbstractTransformedReferenceFrame`:

- `base_frame` — the ambient reference frame (e.g. `Alice()`).
- `xop` — the `FrenetSerretTransform` connecting them.

The evolution parameter $\tau$ is **not** stored on the frame. It is supplied at evaluation time when applying the transform via `act`.

## Frame Transitions

The standard `frame_transition` function works with curve frames just like ordinary frames. It returns a composable operator that you apply with `act`:

```python
# Operator: Alice -> curve frame
op_to_curve = cxf.frame_transition(cxf.Alice(), fs_frame)

# Operator: curve frame -> Alice
op_from_curve = cxf.frame_transition(fs_frame, cxf.Alice())
```

### Applying the Transition

Use `cxfm.act(op, tau, x)` to transform a point. The `tau` parameter is passed through to the `FrenetSerretTransform` callables:

```python
p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
tau = u.Q(0.0, "s")

# Transform p into the curve frame at tau=0
p_curve = cxfm.act(op_to_curve, tau, p)

# Transform back
p_back = cxfm.act(op_from_curve, tau, p_curve)
```

### Chaining Through Multiple Frames

Curve frames compose with any other reference frame. If you have `Alice` and `Alex` as two ordinary frames:

```python
# Alice -> curve frame
op1 = cxf.frame_transition(cxf.Alice(), fs_frame)

# curve frame -> Alex
op2 = cxf.frame_transition(fs_frame, cxf.Alex())
```

The full chain `Alice -> FS(tau) -> Alex` can be applied step-by-step:

```python
p_fs = cxfm.act(op1, tau, p)  # Alice -> curve frame
p_alex = cxfm.act(op2, tau, p_fs)  # curve frame -> Alex
```

And the reverse `Alex -> FS(tau) -> Alice` recovers the original point:

```python
op3 = cxf.frame_transition(cxf.Alex(), fs_frame)
op4 = cxf.frame_transition(fs_frame, cxf.Alice())

p_fs2 = cxfm.act(op3, tau, p_alex)
p_recovered = cxfm.act(op4, tau, p_fs2)
```

## JAX Integration

Curve frames are JAX-native. All transform fields are pure-function closures and the frame object is a valid JAX PyTree (via Equinox).

### JIT Compilation

Because `FrenetSerretTransform` fields are stored as **static** PyTree leaves (functions), the only dynamic parts are $\tau$ and the coordinate data. JIT works directly:

```python
@jax.jit
def transform_point(tau, p):
    return cxfm.act(op_to_curve, tau, p)
```

### Vectorizing Over $\tau$

Use `jax.vmap` to evaluate the transform at many parameter values simultaneously:

```python
taus = u.Q(jnp.linspace(0.0, 6.28, 100), "s")
p = u.Q(jnp.array([2.0, 0.0, 0.0]), "km")

trajectory = jax.vmap(lambda t: cxfm.act(op_to_curve, t, p))(taus)
```

Combine with `jit` for maximum performance:

```python
trajectory = jax.jit(jax.vmap(lambda t: cxfm.act(op_to_curve, t, p)))(taus)
```

## The Bishop Transform

The **Bishop transform** (also called rotation-minimising or parallel-transport frame) provides an alternative to the Frenet–Serret frame. Its key advantage is that it is **well-defined even when the curvature vanishes** ($\kappa = 0$), where the Frenet–Serret normal is singular.

`BishopTransform` stores the same set of $\tau$-dependent callables:

| Field | Meaning |
| --- | --- |
| `location` | curve position $\gamma(\tau)$ |
| `tangent` | unit tangent $\mathbf{T}(\tau)$ |
| `normal1` | first normal $\mathbf{U}_1(\tau)$ (parallel-transported) |
| `normal2` | second normal $\mathbf{U}_2(\tau) = \mathbf{T} \times \mathbf{U}_1$ |

### How the Bishop Frame Differs from Frenet–Serret

| Property | Frenet–Serret | Bishop |
| --- | --- | --- |
| Defined at $\kappa = 0$? | No (singular) | **Yes** |
| Depends on $\gamma''$? | Yes | No (only $\gamma'$) |
| Normal-plane twist | Tracks torsion | **Zero** (rotation-minimising) |
| Initial condition needed? | No | Yes ($\mathbf{U}_1$ at $\tau_0$) |

The Bishop frame normal vectors are obtained by solving a **parallel-transport ODE**:

$$
\frac{d\mathbf{U}_i}{d\tau} = -(\mathbf{U}_i \cdot \mathbf{T}')\,\mathbf{T}
$$

This keeps $\mathbf{U}_1, \mathbf{U}_2$ perpendicular to $\mathbf{T}$ while minimising rotation in the normal plane.

### Building a Bishop Transform from a Curve

```python
def helix(tau: u.Q) -> u.Q:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


bt = cxfc.BishopTransform.from_curve(helix)
```

`from_curve` automatically:

1. Computes $\mathbf{T}$ via JAX autodiff
2. Chooses an initial normal $\mathbf{U}_{1,0}$ via Gram–Schmidt (unless you supply one)
3. Solves the parallel-transport ODE with `jax.experimental.ode.odeint`

Evaluate at a specific $\tau$:

```python
tau = u.Q(0.0, "s")
bt.location(tau)  # Q([1., 0., 0.], 'km')
bt.tangent(tau)  # unit tangent
bt.normal1(tau)  # parallel-transported U1
bt.normal2(tau)  # T x U1
```

### Controlling the Initial Normal

By default, `from_curve` picks the standard basis vector least aligned with $\mathbf{T}(\tau_0)$ via Gram–Schmidt. You can provide an explicit initial normal:

```python
bt_custom = cxfc.BishopTransform.from_curve(
    helix, initial_normal=jnp.array([0.0, 0.0, 1.0])
)
```

The reference parameter $\tau_0$ can also be set:

```python
bt_shifted = cxfc.BishopTransform.from_curve(helix, tau_0=u.Q(1.0, "s"))
```

### Straight Lines

The Bishop frame handles straight lines gracefully — exactly the situation where the Frenet–Serret frame fails:

```python
def line(tau):
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")


bt_line = cxfc.BishopTransform.from_curve(line)
bt_line.normal1(u.Q(5.0, "s"))  # well-defined unit vector
```

### Inversion

Like `FrenetSerretTransform`, every `BishopTransform` has an `.inverse`:

```python
bt_inv = bt.inverse
```

Double-inversion recovers the original: `bt.inverse.inverse` reconstructs the forward transform.

## Building a Bishop Frame

A `BishopFrame` pairs a `BishopTransform` with a base frame, exactly like `FrenetSerretFrame`.

### Convenience Constructor

```python
b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), helix)
```

### Frame Transitions

Frame transitions work identically to the Frenet–Serret case:

```python
op_to_bishop = cxf.frame_transition(cxf.Alice(), b_frame)
op_from_bishop = cxf.frame_transition(b_frame, cxf.Alice())

p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
tau = u.Q(0.0, "s")

p_bishop = cxfm.act(op_to_bishop, tau, p)
p_back = cxfm.act(op_from_bishop, tau, p_bishop)
```

### When to Use Bishop vs Frenet–Serret

- Use **Bishop** when your curve may have zero-curvature segments (e.g. straight-line portions, inflection points) or when you need a twist-free frame.
- Use **Frenet–Serret** when you want the classical differential-geometry frame that tracks curvature and torsion directly.

## Design Notes

### Lazy Evaluation

All transform fields are callables, not pre-computed arrays. This means:

- **Memory-efficient**: no large arrays stored on the object.
- **Composable**: fields can be freely passed to `jit`, `vmap`, `grad`.
- **Exact**: no discretisation error from pre-sampling; the curve is evaluated analytically at each $\tau$.

### Active Semantics

Curve frames follow coordinax's **active transformation** convention. `act(op, tau, x)` moves the represented point data — it does not merely relabel coordinates. The forward transform takes ambient coordinates and expresses them in the curve frame; the inverse takes curve-frame coordinates and returns them to the ambient frame.

### Scalar-First Design

Functions operate on scalar $\tau$ and scalar-component vectors. Batching is achieved via `jax.vmap`, not by passing shaped arrays. This keeps the implementation simple and composes cleanly with all JAX transformations.
