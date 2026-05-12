# Working With Quantity Objects As Coordinates

This tutorial covers using **unxt Quantities** — arrays with units — as coordinate data in coordinax. A `Quantity` (e.g. `u.Q([1, 2, 3], "km")`) carries units but not component names, chart, or frame metadata. Coordinax can infer charts from the array shape when possible, making quantities a convenient middle ground between bare arrays and full vectors.

You will learn how to:

- Pass quantities to `act` for transforms
- Decompose quantities into CDicts with `cdict()`
- Upgrade quantities to `AbstractVector` objects
- Convert units with `u.uconvert()`
- Use quantities with JAX

**Prerequisites**: [Working With Quantities (Angles & Distances)](../guides/quantities.md).

```{admonition} Object Levels
:class: tip

Coordinax supports five levels of coordinate representation, each adding
more metadata. This tutorial covers `Quantity`.

| Level | Type | See tutorial |
| --- | --- | --- |
| Coordinate | `Coordinate` | [Coordinate tutorial](./coordinate_objects.md) |
| Point | `Vector` | [Point tutorial](./point_objects.md) |
| CDict | `dict[str, Quantity]` | [CDict tutorial](./cdict_objects.md) |
| **Quantity** | `unxt.Quantity` | *this page* |
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

## What A Quantity Brings

A `Quantity` attaches **units** to an array. This prevents silent unit confusion — is `1.0` in metres, kilometres, or degrees?

```{code-block} python
>>> q = u.Q([1.0, 2.0, 3.0], "km")
>>> q.unit
Unit("km")
```

Quantities do **not** carry component names (no "x", "y", "z" labels), chart, representation, or frame — that metadata must be provided externally or inferred.

## Applying Transforms To Quantities

Use `cxfm.act()` with a quantity. Coordinax infers the chart from the array shape (length 3 → `cart3d`) and assumes point representation:

```{code-block} python
>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

>>> q = u.Q([1.0, 0.0, 0.0], "km")
>>> result = cxfm.act(rot90z, None, q)
>>> result.unit
Unit("km")
```

### With Explicit Chart

Override the inferred chart:

```{code-block} python
>>> result = cxfm.act(rot90z, None, q, cxc.cart3d)
>>> result.unit
Unit("km")
```

### With Explicit Chart And Representation

Full control:

```{code-block} python
>>> result = cxfm.act(rot90z, None, q, cxc.cart3d, cxr.point)
>>> result.unit
Unit("km")
```

### Translation

```{code-block} python
>>> shift = cxfm.Translate.from_([1, 2, 3], "km")

>>> q_origin = u.Q([0.0, 0.0, 0.0], "km")
>>> result = cxfm.act(shift, None, q_origin)
>>> result.unit
Unit("km")
```

### Identity

The identity transform returns the exact same object:

```{code-block} python
>>> q = u.Q([1.0, 2.0, 3.0], "km")
>>> result = cxfm.act(cxfm.Identity(), None, q)
>>> result is q
True
```

## Decomposing To A CDict

Use `cxc.cdict()` to split a quantity into named components:

```{code-block} python
>>> q = u.Q([1, 2, 3], "km")

>>> d = cxc.cdict(q)
>>> sorted(d.keys())
['x', 'y', 'z']

>>> d["x"]
Q(1, 'km')
```

With an explicit chart:

```{code-block} python
>>> d = cxc.cdict(u.Q([1, 2, 3], "km"), cxc.cart3d)
>>> sorted(d.keys())
['x', 'y', 'z']
```

Once decomposed into a CDict, you can change charts:

```{code-block} python
>>> d_sph = cxc.pt_map(d, cxc.cart3d, cxc.sph3d)
>>> sorted(d_sph.keys())
['phi', 'r', 'theta']
```

## Upgrading To A Vector

Promote a quantity to a `Vector`:

```{code-block} python
>>> q = u.Q([1, 2, 3], "m")

>>> v = cx.Point.from_(q)
>>> v.chart
Cart3D(M=Rn(3))

>>> isinstance(v, cx.Point)
True
```

With an explicit chart:

```{code-block} python
>>> v = cx.Point.from_(q, cxc.cart3d)
>>> v.chart
Cart3D(M=Rn(3))
```

## Upgrading To A Coordinate

Go all the way from a quantity to a `Coordinate`:

```{code-block} python
>>> q = u.Q([1, 2, 3], "km")

>>> v = cx.Point.from_(q)
>>> coord = cx.Point.from_(v, cxf.alice)
>>> coord.frame
Alice()
>>> coord.chart
Cart3D(M=Rn(3))
```

## Unit Conversion

Standard `unxt` API:

```{code-block} python
>>> q_m = u.Q([1000, 2000, 3000], "m")
>>> q_km = u.uconvert("km", q_m)
>>> q_km.unit
Unit("km")
```

## JAX Integration

Quantities are Quax `ArrayValue` objects and work with JAX transformations:

```{code-block} python
>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

>>> @jax.jit
... def rotate_qty(q):
...     return cxfm.act(rot90z, None, q)

>>> q = u.Q([1.0, 0.0, 0.0], "km")
>>> result = rotate_qty(q)
>>> result.unit
Unit("km")
```

## When To Use Quantity

Choose `Quantity` when:

- You want **unit safety** without the overhead of named components or chart/representation metadata.
- You are doing quick one-off computations (e.g. rotating a single point).
- You are passing data to `act` and are happy with chart inference from the array shape.
- You are working with existing unxt-based code and want coordinax transforms to "just work".

**Trade-off**: Chart inference depends on array shape — it only works for trailing axis sizes 1, 2, or 3, and always assumes Cartesian. For non-Cartesian data or explicit chart control, decompose to a CDict or upgrade to a `Point`. See the [CDict tutorial](./cdict_objects.md) or the [Point tutorial](./point_objects.md).

If you need even less overhead and are willing to manage units yourself, use a bare array. See the [Array tutorial](./array_objects.md).
