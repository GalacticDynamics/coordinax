# Working With Vectors

This guide provides a conceptual introduction to coordinax vectors and practical patterns for working with them. For API reference, see [the vector module reference](../api/vectors.md).

## Motivation: Why A Separate Vector Class?

In pure NumPy or JAX, coordinate data is just arrays. But geometry demands more:

1. **Coordinate systems vary**: the same point is `(x, y, z)` in Cartesian but `(r, θ, φ)` in spherical. Which is it?
2. **Units matter**: is `1.0` in meters, parsecs, or degrees? Silent unit confusion causes disasters.
3. **Transformation laws differ**: point coordinates change by the chart transition; velocity fields transform by the Jacobian. These rules cannot be implicit.
4. **Type safety**: mixing spherical and Cartesian accidentally should be impossible, not silently wrong.

Coordinax solves this by attaching **chart** (coordinate system), **data** (component values), and **representation** (transformation law) to every vector, so every number carries its full mathematical context.

## The Four Concepts

| Concept | Type | What it is |
| --- | --- | --- |
| Chart | `AbstractChart` | coordinate system — defines component names and their dimensions |
| Point | `Point` | position in a chart, with an optional reference frame |
| Tangent | `Tangent` | tangent-space quantity (velocity, displacement, acceleration) at a base point |
| Coordinate | `Coordinate` | bundle of a `Point` with named `Tangent` fibre fields |

A **reference frame** (e.g., `Alice()`, `Alex()`) records the observer perspective. A `Point` without a frame defaults to `NoFrame()`.

## From Charts to Points

If you have not yet read [Working With Charts](./charts.md), do so first. Charts define coordinate systems; `Point` expresses data in those systems.

```python
import coordinax.main as cx
import coordinax.charts as cxc
import unxt as u

# Infer chart from array length (3 → cart3d)
p = cx.Point.from_([1, 2, 3], "m")

# Explicit: named components + chart
p_sph = cx.Point.from_(
    {"r": u.Q(1, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(1.0, "rad")},
    cxc.sph3d,
)

# Chart is always accessible
print(p.chart)  # Cart3D(M=Rn(3))
print(p_sph.chart)  # Spherical3D(M=Rn(3))
```

Shape inference for `from_([...], unit)`:

| Array length | Inferred chart |
| ------------ | -------------- |
| 3            | `cart3d`       |
| 2            | `cart2d`       |
| 1            | `cart1d`       |

For a full walkthrough of all construction patterns, see the [Point & Coordinate tutorial](../tutorials/coordinate_objects.md).

## Converting Charts

Use `cconvert()` to change the coordinate system. The geometric point is preserved; only the chart and component values change:

```python
v_cart = cx.Point.from_([1, 2, 3], "m")
v_sph = cx.cconvert(v_cart, cxc.sph3d)  # or v_cart.cconvert(cxc.sph3d)

print(v_sph.chart)  # Spherical3D(M=Rn(3))

# Round-trip
v_back = cx.cconvert(v_sph, cxc.cart3d)
print(v_back.chart)  # Cart3D(M=Rn(3))
```

## Converting Reference Frames

Use `to_frame()` to change the observer. The chart is preserved; the component values are transformed into the new frame:

```python
import coordinax.frames as cxf

p_alice = cx.Point.from_([1, 2, 3], "m", cxf.alice)
p_alex = p_alice.to_frame(cxf.alex)

print(p_alice.frame)  # Alice()
print(p_alex.frame)  # Alex()
print(p_alex.chart)  # Cart3D(M=Rn(3))  — unchanged
```

Identity frame changes are no-ops (same object returned):

```python
assert p_alice.to_frame(cxf.alice) is p_alice
```

For time-dependent frames, pass an optional evolution parameter:

```python
p_t = p_alice.to_frame(cxf.alex, t=u.Q(1.0, "s"))
```

## Tangent Fields and Coordinate Bundles

`Point` represents a location. To carry **tangent quantities** (velocities, displacements, accelerations) anchored at that location, use `Tangent` and bundle everything into a `Coordinate`.

```python
import coordinax.representations as cxr

point = cx.Point.from_([1.0, 0.0, 0.0], "m")
vel = cx.Tangent.from_(
    {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    cxc.cart3d,
    cxr.coord_vel,
)

pv = cx.Coordinate(point=point, velocity=vel)

# Convert the whole bundle — point by transition map, velocity by Jacobian
pv_sph = pv.cconvert(cxc.sph3d)
print(pv_sph.point.chart)  # Spherical3D(M=Rn(3))
print(pv_sph["velocity"].chart)  # Spherical3D(M=Rn(3))
```

The **basis** controls how tangent components transform:

- `coord_basis` — coordinate/tangent basis; components scale with the metric.
- `phys_basis` — orthonormal physical frame; dimension-consistent components.

For a full treatment of `Tangent`, basis kinds, and `Coordinate` bundles, see the [Tangent Vectors guide](./tangents.md) and the [Point & Coordinate tutorial](../tutorials/coordinate_objects.md).

## Operations Decision Table

| Goal | API | What changes | What stays invariant |
| --- | --- | --- | --- |
| Change coordinate system | `p.cconvert(chart)` | chart, component values | geometric point, frame |
| Change coordinate system + all tangents | `pv.cconvert(chart)` | chart of point + all fibres | geometric point, frame |
| Change reference frame | `p.to_frame(frame)` | frame, component values | chart, geometric point |
| Change frame of whole bundle | `pv.to_frame(frame)` | frame of point + all fibres | chart, geometric point |
| Convert units | `u.uconvert(units_dict, v)` | component values | chart, frame, geometric point |

## Combined Frame + Chart Pipelines

Operations chain naturally — each step is independent:

```python
import coordinax.transforms as cxfm

rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
rotated_frame = cxf.TransformedReferenceFrame(cxf.alice, rot90z)

p_alice = cx.Point.from_([1, 2, 3], "m", cxf.alice)

# Frame first, then chart
result = p_alice.to_frame(rotated_frame).cconvert(cxc.sph3d)
print(result.frame)  # TransformedReferenceFrame(...)
print(result.chart)  # Spherical3D(M=Rn(3))
```

## JAX Integration

`Point`, `Tangent`, and `Coordinate` are all JAX PyTrees (via Equinox). They work with `jit`, `vmap`, and `grad` without special handling.

### JIT Compilation

```python
import jax

to_spherical = jax.jit(lambda v: cx.cconvert(v, cxc.sph3d))

p = cx.Point.from_([1.0, 0.0, 0.0], "m")
p_sph = to_spherical(p)
print(p_sph.chart)  # Spherical3D(M=Rn(3))
```

### Vectorisation With vmap

Design functions over scalar (single-point) objects, then batch with `vmap`:

```python
import jax.numpy as jnp

# Scalar function
to_sph = lambda v: cx.cconvert(v, cxc.sph3d)

# Batch via vmap
many = cx.Point.from_(jnp.ones((5, 3)), "m")
many_sph = jax.vmap(to_sph)(many)
print(many_sph.chart)  # Spherical3D(M=Rn(3))
```

Chart, frame, and representation metadata are preserved through all JAX transformations — they are PyTree static fields, not array leaves.

## Immutability

Vectors are immutable. Use `equinox.tree_at` to create a modified copy:

```python
import equinox as eqx

v = cx.Point.from_({"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")})
v2 = eqx.tree_at(lambda t: t.data["x"], v, u.Q(10, "m"))

print(v.data["x"])  # Q(1, 'm')
print(v2.data["x"])  # Q(10, 'm')
```

Immutability ensures no hidden mutations during JAX tracing and that pure functions compose reliably.
