# Working With Points And Coordinates

This tutorial covers two linked types:

- **`Point`** — a position carrying its chart and reference frame, so every number has full provenance: _what_, _where_, and _from whose perspective_.
- **`Coordinate`** — a bundle of a `Point` with named `Tangent` fibre fields (velocities, displacements, accelerations). Chart conversion and frame changes propagate to all fields at once.

You will learn how to:

- Construct `Point` objects with reference frames
- Convert `Point` between charts and frames
- Bundle a `Point` with `Tangent` fields into a `Coordinate`
- Convert the whole bundle between charts in one call
- Change reference frame for the whole bundle
- Use both types with JAX `jit` and `vmap`

**Prerequisites**: [Working With Vectors](../guides/vectors.md), [Working With Tangent Vectors](./tangent_objects.md), and [Working With Frames](../guides/frames.md).

## Setup

```{code-block} python
>>> import coordinax.main as cx
>>> import coordinax.charts as cxc
>>> import coordinax.frames as cxf
>>> import coordinax.representations as cxr
>>> import coordinax.transforms as cxfm
>>> import unxt as u
>>> import jax.numpy as jnp
>>> import jax
```

## Part 1: Point — A Position With Frame

A `Point` stores a position: component data, a chart (coordinate system), and optionally a **reference frame** — the observer from whose perspective the coordinates are expressed. A `Point` without a frame defaults to `NoFrame()`.

### Constructing A Point

```{code-block} python
>>> p = cx.Point.from_([1.0, 2.0, 3.0], "km")
>>> print(p)
<Point: chart=Cart3D (x, y, z) [km]
    [1. 2. 3.]>
```

Pass a frame as the last argument to attach it at construction:

```{code-block} python
>>> p_alice = cx.Point.from_([1.0, 2.0, 3.0], "km", cxf.alice)
>>> p_alice.frame
Alice()
```

Or attach a frame to an existing `Point`:

```{code-block} python
>>> p_alice2 = cx.Point.from_(p, cxf.alice)
>>> p_alice2.frame
Alice()
```

For the most explicit construction — specifying data, chart, and frame:

```{code-block} python
>>> p_explicit = cx.Point.from_(
...     {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")},
...     cxc.cart3d, cxf.alice)
>>> p_explicit.chart
Cart3D(M=Rn(3))
>>> p_explicit.frame
Alice()
```

### Changing Chart

`cconvert` preserves the frame; only the chart and component values change:

```{code-block} python
>>> p_sph = p_alice.cconvert(cxc.sph3d)
>>> p_sph.chart
Spherical3D(M=Rn(3))
>>> p_sph.frame
Alice()
```

Round-tripping:

```{code-block} python
>>> p_back = p_sph.cconvert(cxc.cart3d)
>>> p_back.chart
Cart3D(M=Rn(3))
```

### Changing Frame

`to_frame` transforms the components into the new observer's frame. The chart is preserved:

```{code-block} python
>>> p_alex = p_alice.to_frame(cxf.alex)
>>> p_alex.frame
Alex()
>>> p_alex.chart
Cart3D(M=Rn(3))
```

Identity frame changes are no-ops:

```{code-block} python
>>> p_alice.to_frame(cxf.alice) is p_alice
True
```

To apply a rotation and then convert to spherical in one pipeline:

```{code-block} python
>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
>>> rotated_frame = cxf.TransformedReferenceFrame(cxf.alice, rot90z)

>>> p_rotated_sph = p_alice.to_frame(rotated_frame).cconvert(cxc.sph3d)
>>> p_rotated_sph.chart
Spherical3D(M=Rn(3))
>>> isinstance(p_rotated_sph.frame, cxf.TransformedReferenceFrame)
True
```

See [point_objects.md](./point_objects.md) for a full walkthrough of `Point` including component dictionaries, applying transforms with `act`, unit conversion, and immutability.

---

## Part 2: Coordinate — Point With Tangent Fields

A `Coordinate` is a **vector bundle**: a base `Point` together with named `Tangent` fibre fields anchored at it.

```
Coordinate = Point (base) + {name: Tangent} (fibres)
```

Chart conversion propagates consistently: the base `Point` converts by the chart transition map, and each `Tangent` converts by the **Jacobian pushforward** at the base. On construction, every fibre field whose frame differs from the base point's frame is **automatically converted** to match it, so `pv["velocity"].frame == pv.point.frame` always holds.

### Constructing A Coordinate

Pass a base `Point` and any number of named `Tangent` keyword arguments:

```{code-block} python
>>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
>>> vel = cx.Tangent.from_(
...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
...     cxc.cart3d, cxr.coord_vel,
... )

>>> pv = cx.Coordinate(point=point, velocity=vel)
>>> isinstance(pv, cx.Coordinate)
True
```

### Accessing The Bundle

Properties delegate to the base point; fibre fields are accessed by name:

```{code-block} python
>>> pv.chart        # delegates to pv.point.chart
Cart3D(M=Rn(3))
>>> pv.frame        # delegates to pv.point.frame
NoFrame()
>>> list(pv.keys())
['velocity']
>>> isinstance(pv["velocity"], cx.Tangent)
True
>>> pv["velocity"].semantic
Velocity()
```

### Multiple Fibre Fields

A `Coordinate` can hold any number of named tangent fields:

```{code-block} python
>>> acc = cx.Tangent.from_(
...     {"x": u.Q(0.0, "m/s^2"), "y": u.Q(0.0, "m/s^2"), "z": u.Q(-9.8, "m/s^2")},
...     cxc.cart3d, cxr.coord_acc,
... )

>>> pv2 = cx.Coordinate(point=point, velocity=vel, acceleration=acc)
>>> sorted(pv2.keys())
['acceleration', 'velocity']
>>> pv2["acceleration"].semantic
Acceleration()
```

### Converting Charts (The Whole Bundle)

`cconvert` converts **all fields at once**:

1. Base `Point` converts by the chart transition map.
2. Each `Tangent` converts by the **Jacobian pushforward** at the base point.

```{code-block} python
>>> pv_sph = pv.cconvert(cxc.sph3d)
>>> pv_sph.point.chart
Spherical3D(M=Rn(3))
>>> pv_sph["velocity"].chart
Spherical3D(M=Rn(3))
```

Round-tripping:

```{code-block} python
>>> pv_back = pv_sph.cconvert(cxc.cart3d)
>>> pv_back.point.chart
Cart3D(M=Rn(3))
```

Override the target chart for individual fields with `field_charts`:

```{code-block} python
>>> pv_mixed = pv.cconvert(cxc.sph3d, field_charts={"velocity": cxc.cyl3d})
>>> pv_mixed.point.chart
Spherical3D(M=Rn(3))
>>> pv_mixed["velocity"].chart
Cylindrical3D(M=Rn(3))
```

### Changing Reference Frame

`to_frame` moves **all fields** into the new frame. Construct the framed pieces first, then bundle:

```{code-block} python
>>> point_alice = cx.Point.from_([1.0, 0.0, 0.0], "m", cxf.alice)
>>> vel_alice = cx.Tangent.from_(vel, cxf.alice)
>>> pv_alice = cx.Coordinate(point=point_alice, velocity=vel_alice)

>>> pv_alex = pv_alice.to_frame(cxf.alex)
>>> pv_alex.frame
Alex()
>>> pv_alex["velocity"].frame
Alex()
```

### Automatic Frame Alignment

If a fibre field's frame differs from the base point's frame on construction, it is **automatically converted** to the base's frame. No manual conversion needed:

```{code-block} python
>>> vel_alex = cx.Tangent.from_(vel, cxf.alex)

>>> # point is alice, vel is alex — vel is silently converted to alice
>>> pv_auto = cx.Coordinate(point=point_alice, velocity=vel_alex)
>>> pv_auto["velocity"].frame
Alice()
```

### Combined Pipeline: Frame + Chart

Frame changes and chart conversions can be chained:

```{code-block} python
>>> result = pv_alice.to_frame(rotated_frame).cconvert(cxc.sph3d)
>>> result.point.chart
Spherical3D(M=Rn(3))
>>> isinstance(result.frame, cxf.TransformedReferenceFrame)
True
```

### Indexing A Batched Coordinate

Integer/slice indexing applies to the base point and all fibre fields together:

```{code-block} python
>>> pts = cx.Point.from_(jnp.ones((4, 3)), "m")
>>> vels = cx.Tangent.from_(jnp.zeros((4, 3)), "m/s")
>>> pv_batch = cx.Coordinate(point=pts, velocity=vels)

>>> pv_0 = pv_batch[0]
>>> pv_0.point.shape
()
```

## JAX Integration

Both `Point` and `Coordinate` are JAX PyTrees and work with all JAX transformations.

### JIT Compilation

```{code-block} python
>>> to_spherical = jax.jit(lambda coord: coord.cconvert(cxc.sph3d))

>>> result = to_spherical(pv)
>>> result.point.chart
Spherical3D(M=Rn(3))
```

### Vectorisation With vmap

```{code-block} python
>>> pts_batch = cx.Point.from_(
...     jnp.stack([jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])]), "m"
... )
>>> vels_batch = cx.Tangent.from_(jnp.zeros((2, 3)), "m/s")
>>> pv_batch2 = cx.Coordinate(point=pts_batch, velocity=vels_batch)

>>> pv_sph_batch = jax.vmap(to_spherical)(pv_batch2)
>>> pv_sph_batch.point.chart
Spherical3D(M=Rn(3))
```

## When To Use

| You have | Use |
| --- | --- |
| Position only | `Point` (Part 1 of this tutorial) |
| Velocity / displacement / acceleration only | `Tangent` — see [Tangent tutorial](./tangent_objects.md) |
| Position + tangent field(s) | `Coordinate` (Part 2 of this tutorial) |

Use `Coordinate` when you need chart conversion to apply the correct transformation law to every field automatically, or when frame changes must propagate to all fields at once.
