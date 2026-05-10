# Working With Points And Reference Frames

This tutorial covers `Point` with a reference frame — coordinax's **richest** coordinate container. A `Point` stores data (components), a chart (coordinate system), a representation (transformation law), a manifold, and optionally a **reference frame**, giving every number full provenance: what it measures, in which system, and from whose perspective.

You will learn how to:

- Construct points with frames from multiple input forms
- Change chart (coordinate system) with `cconvert`
- Change reference frame with `to_frame`
- Apply transforms directly with `act`
- Inspect and convert units
- Use frame-attached points with JAX `jit` and `vmap`

**Prerequisites**: [Working With Vectors](../guides/vectors.md) and [Working With Frames](../guides/frames.md).

```{admonition} Object Levels
:class: tip

Coordinax supports four levels of coordinate representation, each adding
more metadata. This tutorial covers the **top level** — a `Point` with a reference frame.

| Level | Type | See tutorial |
| --- | --- | --- |
| **Point (with frame)** | `Point` | *this page* |
| Point (no frame) | `Point` | [Vector tutorial](./vector_objects.md) |
| CDict | `dict[str, Quantity]` | [CDict tutorial](./cdict_objects.md) |
| Quantity | `unxt.Quantity` | [Quantity tutorial](./quantity_objects.md) |
| Array | `jax.Array` | [Array tutorial](./array_objects.md) |

A `Point` always has a `frame` attribute. When constructed without one, it defaults to `NoFrame()`.
```

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

## What Is A Point With Frame?

A `Point` bundles five things together:

```
Point = data + chart + rep + manifold + frame
```

- **data**: component values (e.g. `{"x": Q(1, "km"), ...}`)
- **chart**: the coordinate system (e.g. Cartesian, spherical)
- **rep**: the transformation law (point, tangent, etc.)
- **manifold**: the geometric space
- **frame**: the reference observer — defaults to `NoFrame()` when omitted

This ensures that **every number carries its full context**: the values, the coordinate system, the transformation law, the manifold, and the observer.

## Constructing Points With Frames

`Point.from_()` accepts an optional frame as the last argument.

### From A Point And Frame

Pass an existing `Point` and a frame — the frame is attached to it:

```{code-block} python
>>> vec = cx.Point.from_(
...     {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
... )

>>> coord = cx.Point.from_(vec, cxf.alice)
>>> coord.chart
Cart3D(M=Rn(3))
>>> coord.frame
Alice()
```

### From A Component Dictionary And Chart

Pass a component dictionary, chart, and frame:

```{code-block} python
>>> coord = cx.Point.from_(
...     {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")},
...     cxc.cart3d,
...     cxf.alice,
... )
>>> coord.chart
Cart3D(M=Rn(3))
>>> coord.frame
Alice()
```

### From An Array, Unit, And Frame

The most compact form — the chart is inferred from the array shape (length 3 → `cart3d`):

```{code-block} python
>>> coord = cx.Point.from_([1, 2, 3], "km", cxf.alice)
>>> coord.chart
Cart3D(M=Rn(3))
>>> coord.frame
Alice()
```

### Full Specification (Dict + Chart + Rep + Frame)

When you need to control every aspect:

```{code-block} python
>>> coord = cx.Point.from_(
...     {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")},
...     cxc.cart3d, cxr.point, cxf.alice,
... )
>>> coord.rep
Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())
```

### Default Frame

If no frame is given, `NoFrame` is used:

```{code-block} python
>>> coord_noframe = cx.Point.from_(
...     cx.Point.from_({"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")})
... )
>>> coord_noframe.frame
NoFrame()
```

### Passthrough

Passing an existing point returns it unchanged:

```{code-block} python
>>> same = cx.Point.from_(coord)
>>> same is coord
True
```

## Inspecting Point Data

Access the components, chart, frame, representation, and manifold directly:

```{code-block} python
>>> coord = cx.Point.from_(
...     {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")},
...     cxc.cart3d, cxf.alice,
... )

>>> coord.chart
Cart3D(M=Rn(3))

>>> coord.frame
Alice()

>>> coord.rep
Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

>>> coord.manifold
Rn(3)

>>> isinstance(coord, cx.Point)
True

>>> sorted(coord.data.keys())
['x', 'y', 'z']
```

## Changing The Chart (Coordinate System)

Use `cconvert()` to change the chart while preserving the frame and the geometric point:

```{code-block} python
>>> coord_cart = cx.Point.from_(
...     {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")},
...     cxc.cart3d, cxf.alice,
... )

>>> coord_sph = coord_cart.cconvert(cxc.sph3d)
>>> coord_sph.chart
Spherical3D(M=Rn(3))

>>> coord_sph.frame  # unchanged
Alice()

>>> sorted(coord_sph.data.keys())
['phi', 'r', 'theta']
```

Round-tripping preserves the geometric point:

```{code-block} python
>>> coord_back = coord_sph.cconvert(cxc.cart3d)
>>> coord_back.chart
Cart3D(M=Rn(3))
```

## Changing The Reference Frame

Use `to_frame()` to transform the point into a different observer's frame. First, build a frame that is rotated 90° about the z-axis relative to Alice:

```{code-block} python
>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
>>> rotated_frame = cxf.TransformedReferenceFrame(cxf.alice, rot90z)

>>> coord_alice = cx.Point.from_(
...     {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")},
...     cxc.cart3d, cxf.alice,
... )

>>> coord_rotated = coord_alice.to_frame(rotated_frame)
>>> coord_rotated.frame
TransformedReferenceFrame(base_frame=Alice(), xop=Rotate(R=f...[3,3]))
```

Identity frame changes are no-ops:

```{code-block} python
>>> same = coord_alice.to_frame(cxf.alice)
>>> same is coord_alice
True
```

## Combined Pipeline: Frame + Chart

Real workflows often require both frame and chart changes.

```{code-block} python
>>> coord = cx.Point.from_(
...     {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")},
...     cxc.cart3d, cxf.alice,
... )

>>> rotated_frame = cxf.TransformedReferenceFrame(
...     cxf.alice, cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
... )

>>> # Pipeline: change frame, then chart
>>> result = coord.to_frame(rotated_frame).cconvert(cxc.sph3d)
>>> result.chart
Spherical3D(M=Rn(3))

>>> result.frame
TransformedReferenceFrame(base_frame=Alice(), xop=Rotate(R=f...[3,3]))
```

When reading a pipeline, each step preserves the geometric point but changes one metadata attribute:

1. `to_frame(...)` changes the **frame** (and component values).
2. `cconvert(...)` changes the **chart** (and component values).
3. Both preserve the **represented geometric point**.

## Applying Transforms Directly

Use `cxfm.act()` to apply a transform directly to a point, without building a frame:

```{code-block} python
>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

>>> coord = cx.Point.from_(
...     {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")},
...     cxc.cart3d, cxf.alice,
... )

>>> rotated = cxfm.act(rot90z, None, coord)
>>> isinstance(rotated, cx.Point)
True
```

The second argument (`None`) is the time parameter `tau` — pass `None` for time-independent transforms. For time-dependent transforms, see the [Time-Dependent Frames](./time_dependent_frames.md) tutorial.

## Unit Conversion

Convert component units with `u.uconvert()`:

```{code-block} python
>>> coord_m = cx.Point.from_(
...     {"x": u.Q(1000, "m"), "y": u.Q(2000, "m"), "z": u.Q(3000, "m")},
...     cxc.cart3d, cxf.alice,
... )

>>> coord_km = u.uconvert({"x": "km", "y": "km", "z": "km"}, coord_m)
>>> coord_km.data["x"]
Q(1., 'km')
```

## JAX Integration

Coordinates are JAX PyTrees by construction. They work with `jit` and `vmap`:

### JIT Compilation

```{code-block} python
>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
>>> rotated_frame = cxf.TransformedReferenceFrame(cxf.alice, rot90z)

>>> @jax.jit
... def to_rotated_spherical(c):
...     return c.to_frame(rotated_frame).cconvert(cxc.sph3d)

>>> coord = cx.Point.from_(
...     {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")},
...     cxc.cart3d, cxf.alice,
... )

>>> result = to_rotated_spherical(coord)
>>> result.chart
Spherical3D(M=Rn(3))
```

## When To Track Reference Frames

Attach a frame to a `Point` when you need **full provenance**:

- You are tracking data from multiple observers and need to know which frame each measurement is in.
- You want `to_frame()` semantics to transform between observers.
- Your workflow chains frame transitions and chart conversions.
- You are building an astronomy pipeline where frames like ICRS and Galactocentric are first-class concepts.

If you do not need frame tracking, omit the frame — a `Point` without one defaults to `NoFrame()` and is otherwise identical. See the [Vector tutorial](./vector_objects.md) for the frame-free workflow.
