# Working With Tangent Vectors

This tutorial covers `Tangent` — coordinax's type for **tangent-space quantities** (velocities, displacements, accelerations). Tangent vectors represent elements of the tangent space $T_p M$ and transform differently from points under chart conversion.

You will learn how to:

- Understand why tangent vectors need a different type from points
- Construct `Tangent` objects from arrays, quantities, and dictionaries
- Convert tangent vectors between charts using the Jacobian pushforward
- Attach reference frames to tangent vectors
- Use `Tangent` with JAX `jit` and `vmap`

**Prerequisites**: [Working With Vectors](../guides/vectors.md). `Tangent` is one of several coordinate levels — for the full picture see the [Point](./point_objects.md) and [Coordinate](./coordinate_objects.md) tutorials.

## Setup

```{code-block} python
>>> import coordinax.main as cx
>>> import coordinax.charts as cxc
>>> import coordinax.frames as cxf
>>> import coordinax.representations as cxr
>>> import unxt as u
>>> import jax.numpy as jnp
>>> import jax
```

## Why Tangent Vectors?

Points and tangent vectors both store component dictionaries attached to a chart, but they follow **different transformation laws** under chart conversion:

- A `Point` `(r, θ, φ)` transforms by the chart transition map.
- A `Tangent` `(ṙ, θ̇, φ̇)` transforms by the **Jacobian** of that map, evaluated at the base point.

Using the wrong law gives silently incorrect physics. `Tangent` encodes the correct transformation, so `cconvert` always does the right thing.

Beyond the chart, each `Tangent` carries a **basis** and a **semantic kind**:

- **basis** — `coord_basis` (coordinate/tangent basis, components scale with the metric) or `phys_basis` (orthonormal physical frame, dimension-consistent components).
- **semantic** — `vel` (velocity), `dpl` (displacement), or `acc` (acceleration).

Pre-built singletons combine these: `cxr.coord_vel`, `cxr.phys_vel`, `cxr.coord_acc`, etc.

## Constructing Tangent Vectors

### From A Component Dictionary (Primary)

Pass a component dict, a chart, and a representation singleton:

```{code-block} python
>>> vel = cx.Tangent.from_(
...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
...     cxc.cart3d, cxr.coord_vel)
>>> vel.chart
Cart3D(M=Rn(3))
>>> vel.rep == cxr.coord_vel
True
>>> vel["x"]
Q(1., 'm / s')
>>> vel.frame
NoFrame()
```

### From An Array And Unit

When the chart can be inferred from the array shape and unit, pass the array and unit directly:

```{code-block} python
>>> vel = cx.Tangent.from_([1.0, 2.0, 3.0], "m/s")
>>> vel.chart
Cart3D(M=Rn(3))
```

## Converting Charts — Jacobian Pushforward

Converting a tangent vector to a new chart requires the **base point** at which the Jacobian is evaluated. Pass it via `at=`:

```{code-block} python
>>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
>>> vel_cart = cx.Tangent.from_(
...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
...     cxc.cart3d, cxr.coord_vel)

>>> vel_sph = vel_cart.cconvert(cxc.sph3d, at=point)
>>> print(vel_sph)
<Tangent: chart=Spherical3D (r[m / s], theta[rad / s], phi[rad / s])
    [ 1. -0.  0.]>
```

The basis (`coord_basis`) and semantic (`vel`) are preserved across the conversion; only the chart and component values change.

## Attaching A Reference Frame

Construct a `Tangent` without a frame, then attach one:

```{code-block} python
>>> vel = cx.Tangent.from_(
...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
...     cxc.cart3d, cxr.coord_vel)
>>> vel_alice = cx.Tangent.from_(vel, cxf.alice)
>>> vel_alice.frame
Alice()

>>> vel_alex = vel_alice.to_frame(cxf.alex)
>>> vel_alex.frame
Alex()
```

`cx.Tangent.from_(vel, frame)` replaces the frame without transforming components. `to_frame` applies the full frame transformation.

## JAX Integration

`Tangent` is a JAX PyTree and works with all JAX transformations.

### JIT Compilation

```{code-block} python
>>> to_spherical = jax.jit(lambda v, pt: cx.cconvert(v, cxc.sph3d, at=pt))

>>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
>>> vel = cx.Tangent.from_(
...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
...     cxc.cart3d, cxr.coord_vel)

>>> vel_sph = to_spherical(vel, point)
>>> vel_sph.chart
Spherical3D(M=Rn(3))
```

### Vectorisation With vmap

```{code-block} python
>>> to_sph_one = lambda v, pt: cx.cconvert(v, cxc.sph3d, at=pt)

>>> pts = cx.Point.from_(jnp.ones((3, 3)), "m")
>>> vels = cx.Tangent.from_(jnp.zeros((3, 3)), "m/s")

>>> vels_sph = jax.vmap(to_sph_one)(vels, pts)
>>> vels_sph.chart
Spherical3D(M=Rn(3))
>>> vels_sph.shape
(3,)
```

## When To Use Tangent

| You have | Use |
| --- | --- |
| Position only | `Point` — see [Point tutorial](./point_objects.md) |
| Velocity / displacement / acceleration only | `Tangent` (this tutorial) |
| Position + tangent field(s) | `Coordinate` — see [Coordinate tutorial](./coordinate_objects.md) |

To bundle a `Tangent` with a base `Point`, use `Coordinate`. It handles chart conversion for the whole bundle — base point via the transition map, each fibre via the Jacobian — in a single call.
