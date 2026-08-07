# Working With Curve Frames

This guide introduces **curve-attached reference frames** provided by `coordinaxs.curveframes`. You will learn what a curve frame is, how to build one from a space curve, and how to transform coordinate data between curve frames and ordinary reference frames.

For the mathematical specification see the {doc}`curveframes spec <spec>`. For API reference on the base frame system see [Working With Frames](../../../docs/guides/frames.md) and [Working With Transforms](../../../docs/guides/transforms.md).

```python
import equinox as eqx
import jax
import jax.numpy as jnp

import quaxed.numpy as qnp
import unxt as u

import coordinaxs.curveframes as cxfc
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

Before building a frame, you need a **transform** — the operator that maps ambient coordinates into curve-local coordinates. `FrenetSerretBuilder` is an `equinox.Module` builder: `tau -> Translate(-gamma) | Rotate(R)`. It stores the curve itself, not pre-computed geometry:

| Field | Meaning |
| --- | --- |
| `curve` | the curve $\gamma \mapsto \boldsymbol{\gamma}(\gamma)$ — a pytree leaf; make it an `equinox.Module` for differentiable curve parameters |
| `tau_unit` | physical unit of the curve parameter (static) |
| `gamma` | optional fixed curve parameter (a leaf); `None` means $\tau$ is the curve parameter |

The tangent, normal, and binormal are not stored — they are computed by `rotation_matrix(tau)` (and the `tangent`/`normal`/`binormal` convenience methods) from `curve` via automatic differentiation, every time they're evaluated.

### Building a Transform from a Curve

Define a curve as a function `tau -> Quantity[(3,)]` and pass it to the builder:

```python
def helix(tau: u.Q) -> u.Q:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


fs_transform = cxfc.FrenetSerretBuilder(helix)
```

`FrenetSerretBuilder` uses `unxt.experimental.jacfwd` to compute unit-correct first and second derivatives of the curve, then derives the tangent, normal, and binormal via Gram–Schmidt orthogonalisation. Nothing is precomputed: `tangent`, `normal`, and `binormal` are methods on the builder, each evaluated at the $\tau$ you pass it.

Evaluate the fields at a specific $\tau$:

```python
tau = u.Q(0.0, "s")
fs_transform.location(tau)  # Q([1., 0., 0.], 'km')
fs_transform.tangent(tau)  # unit vector along curve velocity
```

### The `tau_unit` Parameter

By default the builder assumes $\tau$ has units of seconds. If your parameter has different units (e.g. radians, years), pass `tau_unit`:

```python
fs_rad = cxfc.FrenetSerretBuilder(helix, tau_unit="rad")
```

This affects only the automatic differentiation step — the builder's methods still accept any `Quantity` with compatible dimensions.

### Inversion

The builder itself has no `.inverse` — wrap it in `TimeDep` first, whose `.inverse` reverses the mapping pointwise in $\tau$:

```python
fs_op = cxfm.TimeDep(fs_transform)
fs_inv = fs_op.inverse
```

The inverse is another `TimeDep` family whose builder inverts pointwise in $\tau$. Double-inversion recovers the original: `fs_op.inverse.inverse.builder is fs_op.builder`.

## Building a Frenet–Serret Frame

A `FrenetSerretFrame` pairs a `FrenetSerretBuilder` with a **base frame** — the ambient reference frame in which the curve is defined.

### Direct Construction

```python
fs_frame = cxfc.FrenetSerretFrame(
    base_frame=cxf.Alice(),
    xop=fs_op,
    xop_inv=fs_op.inverse,
)
```

### Convenience Constructor

`from_curve` combines both steps — it builds the transform and wraps it:

```python
fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), helix)
```

This is equivalent to the direct construction above.

### Fields

`FrenetSerretFrame` inherits three fields from `AbstractTransformedReferenceFrame`:

- `base_frame` — the ambient reference frame (e.g. `Alice()`).
- `xop` — the `TimeDep` family (wrapping a `FrenetSerretBuilder`) connecting base frame to curve frame.
- `xop_inv` — its pre-computed inverse, `xop.inverse`.

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

Use `cxfm.act(op, tau, x)` to transform a point. The `tau` parameter is passed through to the `FrenetSerretBuilder` callables:

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

Curve frames are JAX-native. The builder is an `equinox.Module`, so its fields — the curve (and, when the curve is itself an `equinox.Module`, its parameters), `gamma`, `tau_0`, `initial_normal` — are genuine PyTree leaves: differentiable and vmappable.

### JIT Compilation

JIT works directly when the operator is _closed over_ rather than passed as an argument:

```python
@jax.jit
def transform_point(tau, p):
    return cxfm.act(op_to_curve, tau, p)
```

This works because `op_to_curve` never becomes a traced argument of `transform_point` — it's baked in at trace time. Passing a builder holding **array leaves** — `BishopBuilder`'s `tau_0`, a `gamma`, or any curve that is itself an `equinox.Module` with array fields — as an _argument_ to a plain `jax.jit` is a different story: `jax.jit` treats a bound-method argument as a static, hashable value, and JAX cannot hash an array leaf:

```python
class Helix(eqx.Module):
    radius: jax.Array  # a differentiable/vmappable leaf, not static

    def __call__(self, tau):
        t = tau.ustrip("s")
        return u.Q(
            jnp.stack([self.radius * jnp.cos(t), self.radius * jnp.sin(t), 0.3 * t]),
            "km",
        )


builder = cxfc.FrenetSerretBuilder(Helix(jnp.asarray(1.5)))

try:
    jax.jit(builder.rotation_matrix)(tau)
except TypeError as exc:
    assert "unhashable" in str(exc)

# eqx.filter_jit partitions the array leaves from the static fields first,
# so it works where plain jax.jit cannot:
R = eqx.filter_jit(builder.rotation_matrix)(tau)
```

A curve whose parameters are plain Python floats (`eqx.field(static=True)`, or just a bare closure) _is_ hashable, so plain `jax.jit` happens to work for it — but then those parameters are trace-time constants, not differentiable leaves. This is the cliff to watch for: switching a curve parameter from static to a leaf (to make it differentiable) silently breaks plain `jax.jit` on any function that takes the builder as an argument; reach for `eqx.filter_jit` by default.

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

`BishopBuilder` extends `AbstractCurveFrameBuilder` with two more fields, in addition to `curve`, `tau_unit`, `gamma`:

| Field | Meaning |
| --- | --- |
| `tau_0` | reference parameter where the initial frame is defined (a leaf); `None` resolves to `Q(0.0, tau_unit)` |
| `initial_normal` | initial $\mathbf{U}_{1,0}$ (a leaf), or `None` for Gram–Schmidt auto-selection |

…plus `diffeqsolver`, a single [`diffraxtra.DiffEqSolver`](https://github.com/GalacticDynamics/diffraxtra) holding the whole `diffrax` configuration — solver, step-size controller, adjoint, step budget — covered in [Configuring the solve](#configuring-the-solve). It is a _static_ field, so it adds no pytree leaves and a `jax.tree.map` over the curve's parameters cannot reach it.

As with `FrenetSerretBuilder`, the tangent and normals are not stored — `normal1(tau)`/`normal2(tau)` compute $\mathbf{U}_1(\tau)$ (by solving the parallel-transport ODE from `tau_0`) and $\mathbf{U}_2(\tau) = \mathbf{T}\times\mathbf{U}_1$ on every call.

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


bt = cxfc.BishopBuilder(helix)
```

`BishopBuilder` automatically:

1. Computes $\mathbf{T}$ via JAX autodiff
2. Chooses an initial normal $\mathbf{U}_{1,0}$ via Gram–Schmidt (unless you supply one)
3. Solves the parallel-transport ODE with [`diffrax`](https://docs.kidger.site/diffrax/)

Evaluate at a specific $\tau$:

```python
tau = u.Q(0.0, "s")
bt.location(tau)  # Q([1., 0., 0.], 'km')
bt.tangent(tau)  # unit tangent
bt.normal1(tau)  # parallel-transported U1
bt.normal2(tau)  # T x U1
```

(configuring-the-solve)=

### Configuring the Solve

The whole solve lives in one field, `diffeqsolver`, a `diffraxtra.DiffEqSolver`. Its defaults — `Tsit5`, `PIDController(rtol=1e-10, atol=1e-10)`, `DirectAdjoint`, `max_steps=16384` — hold orthonormality to $\sim 9 \times 10^{-12}$ out to $|\tau| = 60$ on the helix above.

**Change one knob by deriving from the default, not by building a `DiffEqSolver` from scratch.** `dataclasses.replace` keeps every field you do not name:

```python
import dataclasses

import diffrax as dfx

bt = cxfc.BishopBuilder(helix)
fast = dataclasses.replace(
    bt,
    diffeqsolver=dataclasses.replace(
        bt.diffeqsolver, stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6)
    ),
)
type(fast.diffeqsolver.adjoint).__name__  # 'DirectAdjoint'
```

`equinox.tree_at` works on the `DiffEqSolver` too, but not through the builder — a `static=True` field is not a pytree leaf.

Why it matters: `DiffEqSolver`'s _own_ field defaults are read off `diffrax.diffeqsolve`'s signature, so `DiffEqSolver(dfx.Tsit5(), stepsize_controller=…)` silently picks up `RecursiveCheckpointAdjoint` — the one adjoint that disables tangent and jet propagation. Deriving from `bt.diffeqsolver` cannot do that.

**Choosing an adjoint** is the one knob that changes what the frame can _do_, not just how fast it does it. The default is `DirectAdjoint` rather than `diffrax`'s own default because `act` on tangent data and `act_jet` differentiate the solve in _forward_ mode:

| Adjoint                      | forward (AD) | reverse (AD) | speed   |
| ---------------------------- | ------------ | ------------ | ------- |
| `DirectAdjoint` (default)    | yes          | yes          | slowest |
| `RecursiveCheckpointAdjoint` | **no**       | yes          | fastest |
| `ForwardMode`                | yes          | **no**       | fast    |
| `BacksolveAdjoint`           | **no**       | **no**       | fast    |

`BacksolveAdjoint` is unusable here in _either_ mode: the reparametrised ODE right-hand side closes over $\Delta\tau$ and $\tau_0$, which its backwards solve cannot carry, so both directions raise JAX's `CustomVJPException` ("…with respect to a closed-over value"). That is a property of the reparametrisation, not of the curve — it fails the same way for a curve written as a bare function with no array leaves at all, so there is no way to write the curve that recovers it.

For `RecursiveCheckpointAdjoint`, "forward: no" means tangent and jet propagation raise `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function` — accurate, but it never names the adjoint you chose. Reach for it only when you want `grad` and nothing else.

### Controlling the Initial Normal

By default, the builder picks the standard basis vector least aligned with $\mathbf{T}(\tau_0)$ via Gram–Schmidt. You can provide an explicit initial normal:

```python
bt_custom = cxfc.BishopBuilder(helix, initial_normal=jnp.array([0.0, 0.0, 1.0]))
```

The reference parameter $\tau_0$ can also be set:

```python
bt_shifted = cxfc.BishopBuilder(helix, tau_0=u.Q(1.0, "s"))
```

### Straight Lines

The Bishop frame handles straight lines gracefully — exactly the situation where the Frenet–Serret frame fails:

```python
def line(tau):
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")


bt_line = cxfc.BishopBuilder(line)
bt_line.normal1(u.Q(5.0, "s"))  # well-defined unit vector
```

### Propagating Velocities and Jets

A curve frame is $\tau$-dependent, so transforming a velocity is a _kinematic prolongation_, not a frozen-$\tau$ pushforward: the result picks up the $\dot{R}$ and $\dot{\gamma}$ terms. This works identically for both frame types.

```python
import coordinax.charts as cxc
import coordinax.representations as cxr

at = {"x": u.Q(2.0, "km"), "y": u.Q(-1.0, "km"), "z": u.Q(3.0, "km")}
vel = {"x": u.Q(0.1, "km/s"), "y": u.Q(0.2, "km/s"), "z": u.Q(-0.3, "km/s")}
bishop_op = cxfm.TimeDep(bt)

cxfm.act(
    bishop_op, u.Q(0.7, "s"), vel, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel, at=at
)
```

`act_jet` does the same for a whole jet at once, returning every slot:

```python
cxfm.act_jet(bishop_op, u.Q(0.7, "s"), {0: at, 1: vel}, cxc.cart3d)
```

### Inversion

Like `FrenetSerretBuilder`, wrap `BishopBuilder` in `TimeDep` to get `.inverse`:

```python
bt_op = cxfm.TimeDep(bt)
bt_inv = bt_op.inverse
```

Double-inversion recovers the original: `bt_op.inverse.inverse.builder is bt_op.builder`.

## Building a Bishop Frame

A `BishopFrame` pairs a `BishopBuilder` with a base frame, exactly like `FrenetSerretFrame`.

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

### Builder Evaluation

A curve-frame builder is an `equinox.Module`: `curve`, `gamma` (and, for `BishopBuilder`, `tau_0`, `initial_normal`) are pytree **leaves**, not lazy callables stashed on the instance. Nothing is pre-computed at construction time — `rotation_matrix(tau)` and `__call__(tau)` are ordinary methods that derive the tangent/normal/binormal (or Bishop's parallel-transported normals) from `curve` afresh, on every call, via `unxt.experimental.jacfwd`. This means:

- **Structural, not procedural, JAX integration**: because the parameters are real pytree data, `jit`, `vmap`, and `grad` operate on a builder — or a whole frame — the same way they operate on any other pytree; there's no separate "make it JAX-compatible" step.
- **Differentiable curve parameters**: if `curve` is itself an `equinox.Module` with leaf fields, gradients flow through curve construction and into the frame (see "Differentiating the Curve" below) — not possible when a curve was a bare Python closure.
- **Exact**: no discretisation error from pre-sampling; the curve and its derivatives are evaluated analytically at each $\tau$.

#### Differentiating the Curve

Because a curve can be an `equinox.Module`, its own parameters are ordinary pytree leaves — differentiable through frame construction and evaluation:

```python
class Helix(eqx.Module):
    """A helix whose radius (km) is a differentiable pytree leaf."""

    radius: jax.Array

    def __call__(self, tau):
        t = tau.ustrip("s")
        return u.Q(
            jnp.stack([self.radius * jnp.cos(t), self.radius * jnp.sin(t), 0.3 * t]),
            "km",
        )


def readout(radius):
    builder = cxfc.FrenetSerretBuilder(Helix(radius))
    op = cxfm.TimeDep(builder)
    p = u.Q(jnp.array([2.0, 1.0, -0.5]), "km")
    out = cxfm.act(op, u.Q(0.4, "s"), p)
    return out.ustrip("km")[0]


grad_radius = jax.grad(readout)(1.5)
assert abs(float(grad_radius)) > 1e-3
```

`jax.grad` differentiates through the Gram–Schmidt tangent/normal construction and the rigid-body transform, all the way back to the helix radius. This is the capability the old closure-based `curve` fields could not offer: whatever a plain Python closure captured was a trace-time constant, invisible to `jax.grad`.

#### A Frame Field via `vmap` over `gamma`

With `gamma` set, a builder produces a fixed, $\tau$-independent frame anchored at $\boldsymbol{\gamma}(\gamma)$ — a frame _field_ rather than a moving frame. Because `gamma` is a pytree leaf, `jax.vmap` over a batch of `gamma` values builds a batch of frames in a single call — the frame field along the whole curve, not a Python loop over frames:

```python
def circle(tau):
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


p = u.Q(jnp.array([2.0, 1.0, -0.5]), "km")
gammas = u.Q(jnp.linspace(0.0, 1.5, 5), "s")


def at_gamma(g):
    op = cxfm.TimeDep(cxfc.FrenetSerretBuilder(circle, "s", g))
    return cxfm.act(op, u.Q(0.0, "s"), p)


field = jax.vmap(at_gamma)(gammas)
assert field.ustrip("km").shape == (5, 3)
```

Each row of `field` is the same point `p` expressed in the frame anchored at the corresponding `gamma` value.

### Active Semantics

Curve frames follow coordinax's **active transformation** convention. `act(op, tau, x)` moves the represented point data — it does not merely relabel coordinates. The forward transform takes ambient coordinates and expresses them in the curve frame; the inverse takes curve-frame coordinates and returns them to the ambient frame.

### Scalar-First Design

`rotation_matrix`, `__call__`, and the convenience accessors operate on scalar $\tau$ and scalar-component vectors — a builder's fields hold a single curve, not a batch of curves. Batching over $\tau$, over `gamma`, or over a curve's own parameters is achieved by `jax.vmap`-ing the builder or the `TimeDep` operator, not by passing shaped arrays into the builder's fields. This keeps the builder implementation simple and composes cleanly with all JAX transformations.
