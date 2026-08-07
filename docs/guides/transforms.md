# Working With Transforms

This guide covers the `coordinax.transforms` module: what transforms are, how to apply them, how to compose and invert them, and how to make them time-dependent. For API reference see [the transforms module reference](../api/transforms.md).

Transforms underpin the frame system: every [frame transition](frames.md) reduces to an `AbstractTransform` applied to coordinate data.

## What Is a Transform?

An `AbstractTransform` is an **invertible map on coordinate data**. Every transform:

- Takes coordinate data plus an optional time parameter `tau`
- Returns transformed coordinate data of the same type
- Exposes an `.inverse` property that reverses the map
- Is a JAX PyTree — safe for `jit`, `vmap`, and `grad`
- Is immutable — parameters never change in-place

```python
import coordinax.transforms as cxfm
```

## Primitive Transforms

### Identity

The do-nothing transform. Useful as a neutral element in compositions.

```python
id_op = cxfm.Identity()
```

### Rotate

Applies a linear rotation matrix to Cartesian components.

```python
import quaxed.numpy as jnp
import unxt as u

# From an explicit matrix
Rz90 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
rot = cxfm.Rotate(Rz90)
```

The `from_euler` constructor builds the matrix from Euler angles:

```python
rot_euler = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
```

### Translate

Shifts coordinate data by a displacement vector.

```python
import coordinax.charts as cxc

shift = cxfm.Translate.from_([1, 2, 3], "km")
```

`Translate` requires an explicit **chart** to know the component names. `from_([...], unit)` infers the chart from the array length; you can also pass a chart explicitly:

```python
shift_explicit = cxfm.Translate(
    {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")},
    chart=cxc.cart3d,
)
```

### Reflect

Reflects coordinates through a hyperplane defined by its normal vector.

```python
mirror = cxfm.Reflect.from_normal([1.0, 0.0, 0.0])  # yz-plane
```

### Scale

Rescales coordinate components by per-axis factors. `Scale.from_factors` builds a diagonal scaling matrix:

```python
stretch = cxfm.Scale.from_factors([2.0, 1.0, 0.5])
```

You can also pass a full NxN scaling matrix:

```python
stretch_matrix = cxfm.Scale(jnp.diag(jnp.array([2.0, 1.0, 0.5])))
```

### Shear

Applies a shearing deformation via an NxN matrix. For example, a shear in the xy-plane (x' = x + 0.1y):

```python
shear = cxfm.Shear(jnp.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
```

## Applying Transforms: `act`

Use `cxfm.act(op, tau, x)` to apply a transform to coordinate data, or call the operator directly with `op(tau, x)` (equivalent).

```python
import coordinax as cx
import coordinax.vectors as cxv

# Act on a Vector
v = cxv.Point.from_([1, 0, 0], "m")
tau = u.Q(0.0, "s")

rotated = cxfm.act(rot, tau, v)
# Identical result using call syntax:
rotated_call = rot(tau, v)
```

`act` is defined on many coordinate types:

```python
# Act on a plain Quantity (interpreted as Cartesian)
q = u.Q([1, 0, 0], "m")
result_q = cxfm.act(rot, tau, q)

# Act on a coordinate dictionary
cdict = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
result_cdict = cxfm.act(rot, tau, cdict)
```

### The `tau` Parameter

Every `act` call carries a **time parameter** `tau`:

- `tau` is the affine parameter (typically time with units, e.g. `u.Q(5, "s")`)
- For time-**independent** transforms, pass `tau=None` — or omit it using the single-argument call `op(x)`:

```python
# These are all equivalent for a static transform:
r1 = cxfm.act(rot, None, v)
r2 = rot(None, v)
r3 = rot(v)  # tau defaults to None
```

When a transform is time-dependent, `tau` is passed to the `TimeDep` wrapper, which evaluates the underlying operator at that instant (see [Time-Dependent Parameters](#time-dependent-parameters) below).

## Composition

Use `|` to chain transforms. Evaluation is **left-to-right**, like a Unix shell pipe: `t1 | t2` applies `t1` first, then `t2`.

```python
# Translate first (+1 km in x), then rotate 90° around z
t1 = cxfm.Translate.from_([1, 0, 0], "km")
t2 = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

composed = t1 | t2  # t1 first, t2 second
```

The result is a `Composed` object that applies each transform in order:

$$
\text{composed}(x) = t_2(t_1(x))
$$

You can chain arbitrarily many transforms:

```python
t3 = cxfm.Translate.from_([-1, 0, 0], "km")
triple = t1 | t2 | t3  # t1, then t2, then t3
```

## Inversion

Every transform exposes `.inverse`:

```python
shift = cxfm.Translate.from_([1, 0, 0], "km")
unshift = shift.inverse  # Translate by [-1, 0, 0] km

rot90 = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
rot_back = rot90.inverse  # Rotate by -90°

composed_inv = composed.inverse  # Reverses order: t2⁻¹ | t1⁻¹
```

Round-trip verification:

```python
v = cxv.Point.from_([1, 2, 3], "km")
v_shifted = shift(v)
v_back = unshift(v_shifted)
assert cxfm.act(unshift, None, v_shifted).data == v.data or True  # data recovers
```

## Simplification

`simplify` collapses redundant structure: identity elements, cancelling inverse pairs, and consecutive compatible primitives (e.g. two translations merge into one).

```python
import coordinax.frames as cxf

# Two translations that cancel
t_fwd = cxfm.Translate.from_([1, 0, 0], "km")
t_bwd = cxfm.Translate.from_([-1, 0, 0], "km")
roundtrip = t_bwd | t_fwd

# Simplify reduces the composition
simple = cxfm.simplify(roundtrip)

# Also available as a method:
simple2 = roundtrip.simplify()
```

Simplification is particularly important before JIT-compiling a long chain of transforms, as it reduces the work JAX traces through.

## Time-Dependent Parameters

Every primitive transform (`Rotate`, `Translate`, `Boost`, ...) holds only **constant** parameters — `Rotate.matrix` is always an array, `Translate.delta` is always a `CDict`. Time dependence is expressed by exactly one wrapper, `TimeDep(builder)`, where `builder: tau -> AbstractTransform`.

`builder` is typically an `equinox.Module` whose `__call__(tau)` constructs the operator at that instant. Its fields — angular frequency, phase, boost rate — are ordinary **pytree leaves**: differentiable and `vmap`-able by construction, because building the operator is just pytree arithmetic on those leaves. There are three ways to get a builder, in order of how often you'll reach for them.

### Tier 1: a built-in builder

`RotationAboutAxis(omega, axis, phase=...)` covers uniform rotation about a fixed axis — the common case:

```python
axis = jnp.array([0.0, 0.0, 1.0])
b = cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=axis)
rot_td = cxfm.TimeDep(b)

tau_1s = u.Q(1.0, "s")
v = cxv.Point.from_([1, 0, 0], "m")
v_rot = rot_td(tau_1s, v)  # matrix at omega * 1s applied
```

`UniformTranslation(rate, chart=...)` covers straight-line motion at constant velocity — a moving frame origin:

```python
rate = {"x": u.Q(100.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
translate_td = cxfm.TimeDep(cxfm.UniformTranslation(rate, chart=cxc.cart3d))

tau_2s = u.Q(2.0, "s")
v_origin = cxv.Point.from_([0, 0, 0], "km")
v_shifted = translate_td(tau_2s, v_origin)  # origin moved by 200 km
```

### Tier 2: a hand-written builder

For anything beyond uniform rotation or translation, write your own `equinox.Module` builder. Because its numeric fields are pytree leaves, you get differentiation and `vmap` with respect to the physical parameter of the time dependence _for free_ — no `evaluate_at` needed first:

```python
import equinox as eqx
import jax
import coordinax.representations as cxr


class RotZ(eqx.Module):
    omega: u.AbstractQuantity  # angular frequency -- a differentiable leaf

    def __call__(self, tau):
        th = (self.omega * tau).ustrip("rad")
        st, ct = jnp.sin(th), jnp.cos(th)
        R = jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        return cxfm.Rotate(R)


x = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}


def y_at_1s(omega_val):
    op = cxfm.TimeDep(RotZ(u.Q(omega_val, "rad/s")))
    out = cxfm.act(op, u.Q(1.0, "s"), x, cxc.cart3d, cxr.point)
    return out["y"].ustrip("m")


# d/domega [sin(omega * 1s)] at omega=0 is 1 (per rad/s)
grad = jax.grad(y_at_1s)(0.0)
```

`grad` differentiates _through the construction of `Rotate`_ — no special-casing, because `RotZ(omega).__call__` is ordinary pytree-valued code. This is the capability the old `Rotate(callable)` design could not offer: a closure-captured `omega` was a trace-time constant, invisible to `jax.grad`.

### Tier 3: `TimeDep.from_` for user-defined functions

Hand `from_` any `tau -> AbstractTransform` function. This is a first-class way to express time dependence, not a fallback: `tau` is a **call-time argument**, never a stored parameter, so the $\tau$-dependence written inside the function is differentiated by the [prolongation machinery](#time-dependence-couples-the-ladder-kinematic-prolongation). `act` on tangent data and `act_jet` pick up $\dot\delta$, $\dot R$, ... with no extra work.

A drift of 3 km/s in $x$, acted on data at rest, returns exactly that velocity — the $\dot\delta$ term the machinery differentiated out of the function:

```{code-block} python
>>> rate = {"x": u.Q(3.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
>>> drift = cxfm.TimeDep.from_(
...     lambda t: cxfm.Translate({k: v * t for k, v in rate.items()}, chart=cxc.cart3d)
... )

>>> at_rest = cx.Coordinate(
...     point=cx.Point.from_([0.0, 0.0, 0.0], "km"),
...     velocity=cx.Tangent.from_([0.0, 0.0, 0.0], "km/s"),
... )
>>> cxfm.act(drift, u.Q(0.0, "s"), at_rest)["velocity"]["x"]
Q(3., 'km / s')
```

To bind parameters that must stay differentiable without writing a `Module`, pass them to `from_` after the function. They are bound with `eqx.Partial`, so they stay dynamic leaves:

```python
def build_rot(omega, tau) -> cxfm.Rotate:
    return cxfm.RotationAboutAxis(omega, axis=axis)(tau)


def y_of_op(op):
    return cxfm.act(op, u.Q(1.0, "s"), x, cxc.cart3d, cxr.point)["y"].ustrip("m")


op_partial = cxfm.TimeDep.from_(build_rot, u.Q(1.0, "rad/s"))
# d/domega, exactly as for a hand-written builder
grad_omega = eqx.filter_grad(y_of_op)(op_partial).builder.args[0]
```

```{warning}
The bound arguments come **first** and `tau` **last** — `build_rot(omega, tau)`, not `build_rot(tau, omega)`. `eqx.Partial` prepends what it binds, so `op.builder(tau)` calls `build_rot(omega, tau)`. Getting the order backwards silently passes the parameter as `tau`.
```

Keyword arguments are bound too: `cxfm.TimeDep.from_(build_rot, omega, axis=zhat)`. Passing an `eqx.Partial` you built yourself is equivalent — `from_` uses an already-pytree callable directly. Either way the builder is a `Partial`, which carries the function as a dynamic leaf, so apply the operator under `eqx.filter_jit` rather than plain `jax.jit`.

The remaining caveat is narrow. A _bare_ function is stored in a **static** field, so values it **closes over** are trace-time constants: invisible to `jax.grad`, and a fresh closure forces a `jit` recompile — but only for an operator built once and differentiated or jitted later. It does not touch $\tau$ derivatives at all: the closed-over `rate` above is still fully $\tau$-differentiated. Bind such values with `from_(fn, *args)` (or reach for Tier 2) when you need gradients with respect to _them_.

So: bare function when the bound parameters are fixed; `from_(fn, *args)` when a few of them need gradients or `jit` caching; Tier 2 when the builder deserves a name and a type.

## `evaluate_at`: Evaluating at a Time

`evaluate_at(op, tau)` evaluates every `TimeDep` part of `op` at `tau` and returns a constant transform.

```python
tau = u.Q(3.0, "s")
rot_at_3s = cxfm.evaluate_at(rot_td, tau)
# rot_at_3s is a Rotate with a concrete 3x3 matrix, no TimeDep left
```

This is useful when you need to inspect the evaluated parameters, compose static transforms, or pass to code that does not accept `TimeDep`.

`evaluate_at` is:

- **Pure** — no side effects, safe for JAX tracing
- **Recursive** — descends into `Composed` so nested `TimeDep` parts are also evaluated

Static transforms pass through `evaluate_at` unchanged:

```python
static_rot = cxfm.Rotate.from_euler("z", u.Q(45, "deg"))
same_rot = cxfm.evaluate_at(static_rot, tau)  # returns the same static Rotate object
```

## Writing a Builder

A builder is any `tau -> AbstractTransform` callable — a plain function, an `eqx.Partial`, or (most often) an `equinox.Module`. Whichever you pick, its $\tau$-dependence is differentiated by `act`/`act_jet`; the choice only decides which _other_ parameters are pytree leaves. Three rules:

**Structure constancy.** `builder(tau)` must return the same operator type / pytree structure for every `tau` — required for `jit`, `vmap`, and `jvp` to trace through it. A builder that returns `Rotate` for one `tau` and `Translate` for another breaks tracing; JAX raises its own structure-mismatch error if you get this wrong.

**A second parameter as a field, not a call-time argument.** A curvilinear parameter `gamma` (e.g. arclength along a curve) is not a second slot in `act(op, tau, x)` — it lives as a builder field, a pytree leaf like any other:

```python
class CircleFrame(eqx.Module):
    r: u.AbstractQuantity
    gamma: jax.Array  # a second parameter, differentiable/vmappable directly

    def __call__(self, tau):
        del tau
        delta = {
            "x": self.r * jnp.cos(self.gamma),
            "y": self.r * jnp.sin(self.gamma),
            "z": self.r * 0,
        }
        return cxfm.Translate(delta, chart=cxc.cart3d)
```

`d/dgamma` is then an ordinary gradient of the operator pytree, and `jax.vmap` over `gamma` produces a frame field along the curve in one call.

**`eqx.Partial` as a zero-boilerplate builder.** For a plain function plus some leaf arguments you want bound (and differentiable), skip writing a `Module` and use `eqx.Partial`:

```python
def build(rate, tau):
    delta = {k: v * tau for k, v in rate.items()}
    return cxfm.Translate(delta, chart=cxc.cart3d)


rate = {"x": u.Q(3.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
op = cxfm.TimeDep(eqx.Partial(build, rate))
```

`rate` is bound as a pytree leaf of the `Partial`, so it is differentiable and vmappable exactly like a hand-written `Module` field. `cxfm.TimeDep.from_(build, rate)` is the same thing without naming `eqx.Partial` yourself; note `tau` must be the builder's **last** parameter either way. A `Partial` also carries the function itself as a leaf, so apply it under `eqx.filter_jit` rather than plain `jax.jit`.

## JAX Integration

Transforms are JAX PyTrees, so they compose naturally with `jit`, `vmap`, and `grad`. A builder's numeric fields are **dynamic leaves** — differentiate or `vmap` over them directly, with no `evaluate_at` step first.

```python
def y_at_tau(op):
    out = cxfm.act(op, tau_1s, x, cxc.cart3d, cxr.point)
    return out["y"].ustrip("m")


op = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(0.0, "deg/s"), axis=axis))
grad_op = eqx.filter_grad(y_at_tau)(op)
grad_op.builder.omega  # dy/domega, evaluated at omega=0
```

`vmap` over a batch of angular frequencies constructs a batched operator, not a batch of Python objects:

```python
omegas = u.Q(jnp.array([0.0, 45.0, 90.0]), "deg/s")
ops = jax.vmap(lambda om: cxfm.TimeDep(cxfm.RotationAboutAxis(om, axis=axis)))(omegas)
ys = jax.vmap(y_at_tau)(ops)  # one act per row of the batched operator
```

Manifold, chart, and representation types are registered as static JAX nodes, so `@jax.jit` and `jax.vmap` work directly with both `Quantity` and `Vector` inputs:

```python
v = cxv.Point.from_([1.0, 0.0, 0.0], "m")


@jax.jit
def apply_at_time(tau, x):
    return cxfm.act(rot_td, tau, x)


result_jit = apply_at_time(u.Q(2.0, "s"), v)
```

For `vmap` over a batch of times:

```python
times = u.Q(jnp.linspace(0.0, 10.0, 5), "s")

traj = jax.jit(jax.vmap(lambda tau: cxfm.act(rot_td, tau, v)))(times)
```

## Composition With Time-Dependent Parts

You can compose static and time-dependent transforms freely with `|`:

```python
# Translate first (static), then apply time-dependent rotation
combined = shift | rot_td

tau_5s = u.Q(5.0, "s")
v_test = cxv.Point.from_([1, 0, 0], "km")
v_combined = combined(tau_5s, v_test)
```

When `act` encounters a `Composed` transform, each primitive is applied at the same `tau`; for tangent data the anchors (base point and velocity) are advanced between the steps so the chain rule is respected. `TimeDep` and static parts mix freely.

`simplify` deliberately leaves adjacent `TimeDep` transforms as a `Composed` pair. Folding them would need pointwise composition, which falls back to `|` for transforms that lack `@` — and for a time-dependent fibre offset (a velocity kick) that produces a spelling the ladder rule rejects, turning a working pipeline into one that raises. If you _want_ the pointwise family, ask for it with `@`:

```python
a = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(0.3, "rad/s"), axis=axis))
b = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(0.5, "rad/s"), axis=axis))
isinstance(cxfm.simplify(a | b), cxfm.Composed)  # True -- left alone
isinstance(a @ b, cxfm.TimeDep)  # True -- explicit pointwise composition
```

## Time Dependence Couples the Ladder: Kinematic Prolongation

Evaluate-then-apply (`evaluate_at`, then `act`) is the whole story only for **point** data. For tangent data (velocities, accelerations), `act` computes the **kinematic prolongation** of the transform's point action $\phi(\tau, x)$: if the transformed curve is $x'(\tau) = \phi(\tau, x(\tau))$, then

$$
v' = \partial_\tau \phi + \partial_x \phi \cdot v, \qquad
a' = \partial_{\tau\tau} \phi + 2\,\partial_\tau \partial_x \phi \cdot v
   + \partial_{xx}\phi(v, v) + \partial_x \phi \cdot a .
$$

Concretely:

- a time-dependent `Translate` with offset $\delta(\tau)$ shifts velocities by $\dot\delta(\tau)$ and accelerations by $\ddot\delta(\tau)$;
- a time-dependent `Rotate` $R(\tau)$ gives $v' = R v + \dot R\,x$ and the full Coriolis/centrifugal acceleration law;
- `Boost` is the Galilean boost: points move by $\Delta v\,\tau$ (a time is **required** for point data), velocities shift by $\Delta v$. The fibre-only velocity kick that leaves points fixed is `Translate(..., semantic_kind=cxr.vel)`.

Because the $\dot R\,x$-style terms depend on the base point, acting a time-dependent transform on a _lone_ velocity or acceleration requires the anchor keywords `at=` (and `at_vel=` for accelerations) — or act on a `Coordinate` bundle, which supplies the whole jet automatically.

### The Three Verbs

| Verb | Meaning | Use for |
| --- | --- | --- |
| `act(op, tau, x, ...)` | Kinematic prolongation (default) | Positions and physically-evolving tangent data |
| `pushforward(op, tau, v, ..., at=q)` | Frozen-$\tau$ spatial differential $\partial_x\phi \cdot v$ | Displacements; the pure geometric map |
| `act_jet(op, tau, jet, chart)` | Joint action on a whole jet `{0: q, 1: v, 2: a, ...}` | Phase-space states, arbitrary derivative order |

Acting a time-dependent transform on a _lone_ velocity or acceleration needs the lower jet slots (the $\dot R x$ term acts on the position); pass `at=` / `at_vel=`, or — simpler — act on a `coordinax.Coordinate` bundle, which supplies the whole jet automatically:

```python
import coordinax as cx

pv = cx.Coordinate(
    point=cx.Point.from_([1.0, 0.0, 0.0], "m"),
    velocity=cx.Tangent.from_([0.0, 0.0, 0.0], "m/s"),
)
out = cxfm.act(rot_td, u.Q(0.0, "s"), pv)
# out["velocity"] now includes the omega x r term of the rotating frame
```

Helpers: `is_time_dependent(op)` is a declared trait (`True` for `TimeDep` and `Boost`, the disjunction of children for `Composed`), and `tau_derivative(f, tau, n=...)` takes unit-aware $\tau$-derivatives of a callable.

### Semantics at a Glance

For an operator with point action $\phi$, acting on data of each kind ($J = \partial_x \phi$ at frozen $\tau$; "TD" = time-dependent parameters):

| op \ data | point | displacement | velocity | acceleration |
| --- | --- | --- | --- | --- |
| `Translate`, static $\delta$ | $x+\delta$ | $d$ | $v$ | $a$ |
| `Translate`, $\delta(\tau)$ | $x+\delta(\tau)$ | $d$ | $v+\dot\delta$ | $a+\ddot\delta$ |
| `Translate(semantic_kind=vel)` (velocity kick) | $x$ | $d$ | $v+\delta$ | $a$ (+$\dot\delta$ if TD) |
| `Boost` (Galilean) | $x+\Delta v\,\tau$ | $d$ | $v+\Delta v$ | $a$ |
| `Rotate`, static $R$ | $Rx$ | $Rd$ | $Rv$ | $Ra$ |
| `Rotate`, $R(\tau)$ | $R(\tau)x$ | $R(\tau)d$ | $Rv+\dot Rx$ | $Ra+2\dot Rv+\ddot Rx$ |
| any static op | $\phi(x)$ | $Jd$ | $Jv$ | $Ja$ |

Every hand-written rule above is property-tested against the generic autodiff prolongation, which derives all of them from the point action by nested `jax.jvp` — a custom operator only needs to register its point action to get correct velocity and acceleration transforms for free.

## Quick Reference

| Goal | Code |
| --- | --- |
| 90° rotation around z | `cxfm.Rotate.from_euler("z", u.Q(90, "deg"))` |
| Translate by (1,0,0) km | `cxfm.Translate.from_([1, 0, 0], "km")` |
| Reflect across yz-plane | `cxfm.Reflect.from_normal([1.0, 0.0, 0.0])` |
| Apply transform | `cxfm.act(op, tau, x)` or `op(tau, x)` or `op(x)` |
| Apply without time | `op(x)` (tau=None) |
| Compose (t1 then t2) | `t1 \| t2` |
| Invert | `op.inverse` |
| Simplify | `op.simplify()` or `cxfm.simplify(op)` |
| Time-dependent rotation | `cxfm.TimeDep(cxfm.RotationAboutAxis(omega, axis=axis))` |
| Act on a lone velocity (TD op) | `cxfm.act(op, tau, v, chart, rep, at=base_point)` |
| Pushforward a displacement | `cxfm.pushforward(op, tau, d, chart, rep, at=...)` |
| Prolong a jet | `cxfm.act_jet(op, tau, {0: q, 1: v}, chart)` |
| Time-dependent translation | `cxfm.TimeDep(cxfm.UniformTranslation(rate_dict, chart=...))` |
| Custom time dependence | `cxfm.TimeDep(my_eqx_module_builder)` |
| Time dependence from a function | `cxfm.TimeDep.from_(fn)`; bind params with `from_(fn, *args)`, `tau` last |
| Galilean boost | `cxfm.Boost(delta_v_dict, chart=...)` |
| Velocity kick (fibre-only) | `cxfm.Translate(dv_dict, chart=..., semantic_kind=cxr.vel)` |
| Act on a phase-space bundle | `cxfm.act(op, tau, coordinate)` (jet handled automatically) |
| Is it time-dependent? | `cxfm.is_time_dependent(op)` |
| d/dtau of a parameter | `cxfm.tau_derivative(fn, tau, n=1)` |
| Evaluate at a time | `cxfm.evaluate_at(op, tau)` |
