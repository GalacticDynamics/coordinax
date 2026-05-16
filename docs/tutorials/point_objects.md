# Working With Vector Objects

This tutorial covers `Vector` — coordinax's **self-contained** geometric coordinate container. A `Vector` stores data together with its chart (coordinate system), representation (transformation law), and manifold, so every number carries its mathematical meaning.

You will learn how to:

- Construct vectors from dictionaries, arrays, and quantities
- Change coordinate system with `cconvert`
- Apply transforms with `act`
- Convert units per-component
- Access data fields and inspect metadata
- Use vectors with JAX `jit` and `vmap`

**Prerequisites**: [Working With Charts](../guides/charts.md).

```{admonition} Object Levels
:class: tip

Coordinax supports a few levels of coordinate representation, each adding
more metadata. This tutorial covers `Point`.

| Level | Type | See tutorial |
| --- | --- | --- |
| Coordinate bundle | `Coordinate` | [Tangent tutorial](./tangent_objects.md) |
| **Point** \& Tangent | `Point`, `Tangent` | *this page*, [Tangent tutorial](./tangent_objects.md) |
| CDict | `dict[str, Quantity]` | [CDict tutorial](./cdict_objects.md) |
| Quantity | `unxt.Quantity` | [Quantity tutorial](./quantity_objects.md) |
| Array | `jax.Array` | [Array tutorial](./array_objects.md) |
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

## What Is A Vector?

A `Vector` bundles four things together:

```
Vector = data + chart + representation + manifold
```

- **data**: a dictionary mapping component names to values (quantities or arrays)
- **chart**: the coordinate system (e.g. Cartesian, spherical)
- **representation**: how the data transforms (point, tangent, etc.)
- **manifold**: the geometric space the data lives in

Because all metadata is attached, `Vector` operations like `cconvert` and `act` need **no extra arguments** — the vector knows everything about itself.

## Constructing Vectors

`Vector.from_()` supports flexible input patterns.

### From An Array And Unit (Simplest)

The chart is inferred from the array length (3 → `cart3d`):

```{code-block} python
>>> v = cx.Point.from_([1, 2, 3], "m")
>>> v.chart
Cart3D(M=Rn(3))

>>> sorted(v.data.keys())
['x', 'y', 'z']
```

Shape inference:

- Length 3 → `cart3d`
- Length 2 → `cart2d`
- Length 1 → `cart1d`

### From A Quantity

```{code-block} python
>>> q = u.Q([1, 2, 3], "m")
>>> v = cx.Point.from_(q)
>>> v.chart
Cart3D(M=Rn(3))
```

### From A Component Dictionary

The most explicit pattern — every component is named:

```{code-block} python
>>> v = cx.Point.from_(
...     {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
... )
>>> v.chart
Cart3D(M=Rn(3))
```

### With Explicit Chart

Override the inferred chart:

```{code-block} python
>>> v = cx.Point.from_(
...     {"r": u.Q(1, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(1.0, "rad")},
...     cxc.sph3d,
... )
>>> v.chart
Spherical3D(M=Rn(3))
```

### With Explicit Chart And Representation

```{code-block} python
>>> v = cx.Point.from_(
...     {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")},
...     cxc.cart3d,
...     cxr.point,
... )
>>> print(v.rep)
Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())
```

### Passthrough

```{code-block} python
>>> v1 = cx.Point.from_([1, 2, 3], "m")
>>> v2 = cx.Point.from_(v1)
>>> v2 is v1
True
```

## Inspecting Vector Metadata

```{code-block} python
>>> v = cx.Point.from_([1, 2, 3], "m")

>>> v.chart
Cart3D(M=Rn(3))

>>> v.rep
point

>>> v.M
Rn(3)

>>> sorted(v.data.keys())
['x', 'y', 'z']

>>> v.data["x"]
Q(1, 'm')
```

## Reading Metric Diagonals

Because a vector carries both its chart and its manifold, you can ask the manifold for the metric diagonal entries at the represented location:

```{code-block} python
>>> v = cx.Point.from_(
...     {"r": u.Q(2, "km"), "theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")},
...     cxc.sph3d,
... )

>>> gdiag = v.M.scale_factors(v.chart, at=v.data)
>>> gdiag.shape
(3,)
>>> gdiag.unit.to_string()
'(, km2 / rad2, km2 / rad2)'
```

`scale_factors` returns the diagonal metric entries $g_{ii}$ as a 1-D `QuantityMatrix`, so each direction can keep its own unit.

## Changing The Chart (Coordinate Conversion)

Use `cx.cconvert()` to transform between coordinate systems. The geometric point is preserved; the chart and component values change:

```{code-block} python
>>> v_cart = cx.Point.from_(
...     {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
... )

>>> v_sph = cx.cconvert(v_cart, cxc.sph3d)
>>> v_sph.chart
Spherical3D(M=Rn(3))

>>> sorted(v_sph.data.keys())
['phi', 'r', 'theta']
```

Round-tripping:

```{code-block} python
>>> v_back = cx.cconvert(v_sph, cxc.cart3d)
>>> v_back.chart
Cart3D(M=Rn(3))
```

### Cartesian → Cylindrical → Spherical

```{code-block} python
>>> v = cx.Point.from_({"x": u.Q(1, "km"), "y": u.Q(1, "km"), "z": u.Q(1, "km")})

>>> v_cyl = cx.cconvert(v, cxc.cyl3d)
>>> v_cyl.chart
Cylindrical3D(M=Rn(3))

>>> v_sph = cx.cconvert(v_cyl, cxc.sph3d)
>>> v_sph.chart
Spherical3D(M=Rn(3))
```

## Applying Transforms

Use `cxfm.act()` to apply a transform. Because `Vector` is self-contained, no extra arguments are needed:

```{code-block} python
>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
>>> v = cx.Point.from_({"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")})

>>> rotated = cxfm.act(rot90z, None, v)
>>> isinstance(rotated, cx.Point)
True
```

Translation:

```{code-block} python
>>> shift = cxfm.Translate.from_([1, 2, 3], "km")
>>> v_origin = cx.Point.from_({"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")})

>>> shifted = cxfm.act(shift, None, v_origin)
>>> shifted.data["x"]
Q(1, 'km')
>>> shifted.data["y"]
Q(2, 'km')
>>> shifted.data["z"]
Q(3, 'km')
```

The second argument is `tau` (time parameter) — pass `None` for static transforms. For time-dependent transforms, see the [Time-Dependent Frames tutorial](./time_dependent_frames.md).

## Unit Conversion

Convert units per-component with `u.uconvert()`:

```{code-block} python
>>> v = cx.Point.from_([1000, 2000, 3000], "m")

>>> v_km = u.uconvert({"x": "km", "y": "km", "z": "km"}, v)
>>> v_km.data["x"]
Q(1., 'km')
>>> v_km.data["y"]
Q(2., 'km')
```

## Immutability

Vectors are immutable. To create a modified copy, use `equinox.tree_at()`:

```{code-block} python
>>> import equinox as eqx

>>> v = cx.Point.from_({"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")})
>>> v2 = eqx.tree_at(lambda t: t.data["x"], v, u.Q(10, "m"))

>>> v.data["x"]
Q(1, 'm')
>>> v2.data["x"]
Q(10, 'm')
```

## JAX Integration

Vectors are JAX PyTrees (via Equinox), so all JAX transformations work out of the box.

### JIT Compilation

```{code-block} python
>>> @jax.jit
... def rotate_to_spherical(v):
...     r = cxfm.act(cxfm.Rotate.from_euler("z", u.Q(90, "deg")), None, v)
...     return cx.cconvert(r, cxc.sph3d)

>>> v = cx.Point.from_({"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")})
>>> result = rotate_to_spherical(v)
>>> result.chart
Spherical3D(M=Rn(3))
```

## Upgrading To A Coordinate

A `Point` represents a location. If you also need to carry tangent quantities like velocity or acceleration **at** that point, bundle the `Point` with `Tangent` fields into a `Coordinate`:

```{code-block} python
>>> import coordinax.representations as cxr
>>> import unxt as u

>>> vel = cx.Tangent.from_(
...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
...     cxc.cart3d, cxr.coord_vel,
... )
>>> pv = cx.Coordinate(point=cx.Point.from_([1.0, 0.0, 0.0], "m"), velocity=vel)
>>> pv_sph = pv.cconvert(cxc.sph3d)
>>> pv_sph["velocity"].chart
Spherical3D(M=Rn(3))
```

See the [Coordinate tutorial](./coordinate_objects.md) for the full walkthrough.

`Tangent` on its own is **not** an upgrade of `Point` — it is a separate type for tangent-space quantities that exist independently of a base location. See the [Tangent tutorial](./tangent_objects.md) if you need to work with tangent vectors directly.

Attach a reference frame to a `Point` to track the observer:

```{code-block} python
>>> v = cx.Point.from_({"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")})
>>> coord = cx.Point.from_(v, cxf.alice)
>>> coord.frame
Alice()
>>> coord.chart
Cart3D(M=Rn(3))
```

## Promoting a Point to a Displacement

Sometimes you have a `Point` whose component data you want to reinterpret as a tangent-space **displacement** rather than an absolute location — for example, when feeding a position offset into an operation that expects a `Tangent`.

`change_basis` handles this in one call. The component data is **unchanged**; only the geometric type changes from `PointGeometry` to `TangentGeometry` with `Displacement` semantics:

```{code-block} python
>>> pt = cx.Point.from_([1.0, 2.0, 3.0], "m")
>>> disp = cxr.change_basis(pt, cxr.coord_basis)
>>> disp
Tangent( {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')},
         chart=Cart3D(M=Rn(3)), basis=coord_basis, semantic=dpl )
```

You can request the physical (orthonormal) basis instead:

```{code-block} python
>>> disp_phys = cxr.change_basis(pt, cxr.phys_basis)
>>> print(disp_phys.rep)
Representation(
    geom_kind=TangentGeometry(), basis=PhysicalBasis(), semantic_kind=Displacement()
)
```

See the [Tangent guide](../guides/tangents.md) for more on basis and semantic kinds.

## When To Use Point

Choose `Point` when:

- You need chart and representation metadata attached to your data.
- You want `cconvert` and `act` to work without extra arguments.
- You do not need reference-frame tracking (no `to_frame`).
- You are doing geometry: chart conversions, transform pipelines, or building higher-level abstractions.

If you need tangent fields (velocity, displacement, acceleration) bundled with the point, use `Coordinate`. See the [Coordinate tutorial](./coordinate_objects.md).

If you need standalone tangent quantities (without a base point), use `Tangent`. See the [Tangent tutorial](./tangent_objects.md).

If you want to reinterpret a `Point`'s data as a displacement, use `cxr.change_basis(pt, cxr.coord_basis)` to promote it to a `Tangent` with `Displacement` semantics.

If you need frame tracking, attach a frame to a `Point`. See the [Coordinate tutorial](./coordinate_objects.md).

If you want a lighter representation, use a component dictionary (`CDict`). See the [CDict tutorial](./cdict_objects.md).
