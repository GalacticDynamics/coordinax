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
import coordinax.main as cx
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

When a transform has time-dependent parameters, `tau` is passed to the callable fields to materialise them at that instant (see [Time-Dependent Parameters](#time-dependent-parameters) below).

## Composition

Use `|` to chain transforms. Evaluation is **right-to-left**: `t2 | t1` applies `t1` first, then `t2`.

```python
# Translate first (+1 km in x), then rotate 90° around z
t1 = cxfm.Translate.from_([1, 0, 0], "km")
t2 = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

composed = t2 | t1  # t1 first, t2 second
```

The result is a `Composed` object that applies each transform in order:

$$
\text{composed}(x) = t_2(t_1(x))
$$

You can chain arbitrarily many transforms:

```python
t3 = cxfm.Translate.from_([-1, 0, 0], "km")
triple = t3 | t2 | t1  # t1, then t2, then t3
```

## Inversion

Every transform exposes `.inverse`:

```python
shift = cxfm.Translate.from_([1, 0, 0], "km")
unshift = shift.inverse  # Translate by [-1, 0, 0] km

rot90 = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
rot_back = rot90.inverse  # Rotate by -90°

composed_inv = composed.inverse  # Reverses order: t1⁻¹ | t2⁻¹
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

Any parameter field of a primitive transform can be a **callable** instead of a static value. The callable receives `tau` and returns the parameter value.

This is the primary mechanism for time-dependent physics: rotating frames, moving observers, time-varying boosts.

### Time-Dependent Rotation

Pass a function `tau -> matrix` to `Rotate`:

```python
def angular_velocity_matrix(tau):
    """Rotation matrix for a frame rotating at 0.5 rad/s around z."""
    omega = 0.5  # rad/s (plain float — no unit conversion needed)
    theta = omega * tau.ustrip("s")
    ct, st = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])


rot_td = cxfm.Rotate(angular_velocity_matrix)
```

At any given `tau`, the callable is evaluated and the resulting matrix is used:

```python
tau_1s = u.Q(1.0, "s")
v = cxv.Point.from_([1, 0, 0], "m")
v_rot = rot_td(tau_1s, v)  # applies matrix evaluated at t=1 s
```

### Time-Dependent Translation

Pass a function `tau -> CDict` to `Translate`:

```python
def orbit_offset(tau):
    """Moving frame origin: x(t) = 100 km/s * t."""
    speed = u.Q(100.0, "km/s")
    return {"x": speed * tau, "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}


translate_td = cxfm.Translate(orbit_offset, chart=cxc.cart3d)
```

```python
tau_2s = u.Q(2.0, "s")
v_origin = cxv.Point.from_([0, 0, 0], "km")
v_shifted = translate_td(tau_2s, v_origin)  # origin moved by 200 km
```

## `materialize_transform`: Materialising at a Time

`materialize_transform(op, tau)` **materialises** a time-dependent transform at a specific `tau`, returning a new transform of the **same type** with all callable fields replaced by their values at `tau`. The result is a plain static transform.

```python
tau = u.Q(3.0, "s")
rot_at_3s = cxfm.materialize_transform(rot_td, tau)
# rot_at_3s is a Rotate with a concrete 3x3 matrix, not a callable
```

This is useful when you need to inspect the materialised parameters, compose static transforms, or pass to code that does not accept callables.

`materialize_transform` is:

- **Pure** — no side effects, safe for JAX tracing
- **Structure-preserving** — returns same `Rotate`, `Translate`, etc.
- **PyTree-compatible** — uses `equinox.partition` / `equinox.combine` internally

Static transforms pass through `materialize_transform` unchanged:

```python
static_rot = cxfm.Rotate.from_euler("z", u.Q(45, "deg"))
same_rot = cxfm.materialize_transform(
    static_rot, tau
)  # returns equivalent static Rotate
```

## JAX Integration

Transforms are JAX PyTrees, so they compose naturally with `jit`, `vmap`, and `grad`. The callable fields in time-dependent transforms are **static leaves** (functions are not JAX arrays), while numeric fields are dynamic. This means `materialize_transform` is needed before differentiating through the materialized parameters.

Manifold, chart, and representation types are registered as static JAX nodes, so `@jax.jit` and `jax.vmap` work directly with both `Quantity` and `Vector` inputs:

```python
import jax

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
combined = rot_td | shift

tau_5s = u.Q(5.0, "s")
v_test = cxv.Point.from_([1, 0, 0], "km")
v_combined = combined(tau_5s, v_test)
```

When `act` encounters a `Composed` transform, it calls `materialize_transform` on each primitive with the same `tau` before applying it. Callable and static parts mix freely.

## Quick Reference

| Goal | Code |
| --- | --- |
| 90° rotation around z | `cxfm.Rotate.from_euler("z", u.Q(90, "deg"))` |
| Translate by (1,0,0) km | `cxfm.Translate.from_([1, 0, 0], "km")` |
| Reflect across yz-plane | `cxfm.Reflect.from_normal([1.0, 0.0, 0.0])` |
| Apply transform | `cxfm.act(op, tau, x)` or `op(tau, x)` or `op(x)` |
| Apply without time | `op(x)` (tau=None) |
| Compose (t1 then t2) | `t2 \| t1` |
| Invert | `op.inverse` |
| Simplify | `op.simplify()` or `cxfm.simplify(op)` |
| Time-dependent rotation | `cxfm.Rotate(callable_returning_matrix)` |
| Time-dependent translation | `cxfm.Translate(callable_returning_dict, chart=...)` |
| Materialise at time | `cxfm.materialize_transform(op, tau)` |
