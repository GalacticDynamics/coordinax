# Working With Plain Arrays As Coordinates

This tutorial covers using **bare JAX arrays** as coordinate data in coordinax. A plain array carries no units, no component names, no chart, and no frame — all that metadata must be supplied explicitly at every call site. This is the **lowest level** of the coordinax object tower, offering maximum performance and direct interop with raw JAX code.

You will learn how to:

- Apply transforms to arrays with `act` (requires chart, rep, and usys)
- Understand the `usys` (unit system) requirement
- Decompose arrays into CDicts for chart conversion
- Upgrade arrays to higher-level objects
- Use arrays with JAX

```{admonition} Object Levels
:class: tip

Coordinax supports five levels of coordinate representation, each adding
more metadata. This tutorial covers the **bottom level** — plain arrays.

| Level | Type | See tutorial |
| --- | --- | --- |
| Coordinate | `Coordinate` | [Coordinate tutorial](./coordinate_objects.md) |
| Vector | `Vector` | [Vector tutorial](./vector_objects.md) |
| CDict | `dict[str, Quantity]` | [CDict tutorial](./cdict_objects.md) |
| Quantity | `unxt.Quantity` | [Quantity tutorial](./quantity_objects.md) |
| **Array** | `jax.Array` | *this page* |
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

## When To Use Plain Arrays

Plain arrays are the right choice when:

- You are in a **performance-critical inner loop** and cannot afford object-construction overhead.
- You are **interfacing with raw JAX code** (e.g. existing numerical solvers, neural network outputs).
- You **already know** the coordinate system and units and can supply them explicitly.
- You are **prototyping or teaching** and want minimal boilerplate.

The trade-off: you must pass chart, representation, and a unit system to every coordinax function call. If you find yourself repeating the same metadata, upgrade to a `Quantity` or `Vector`.

## Applying Transforms To Arrays

Use `cxfm.act()` with explicit chart, representation, and **unit system**:

```{code-block} python
>>> usys = u.unitsystem("m", "s", "kg", "rad")

>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

>>> arr = jnp.array([1.0, 0.0, 0.0])
>>> result = cxfm.act(rot90z, None, arr, cxc.cart3d, cxr.point, usys=usys)
>>> isinstance(result, jnp.ndarray)
True
```

The arguments:

1. `rot90z` — the transform
2. `None` — time parameter (None for static transforms)
3. `arr` — the data
4. `cxc.cart3d` — the chart (coordinate system)
5. `cxr.point` — the representation (point geometry)
6. `usys=usys` — the unit system (maps physical dimensions to concrete units)

### Why `usys` Is Required

Transforms like `Translate` store their offsets **with units** (e.g. `Translate({"x": Q(1, "km"), ...})`). A bare array has no units, so coordinax cannot add metres to a unitless number. The `usys` tells coordinax how to interpret the array: "these numbers are in metres."

```{code-block} python
>>> usys = u.unitsystem("km", "s", "kg", "rad")
>>> shift = cxfm.Translate.from_([1, 2, 3], "km")

>>> arr = jnp.array([0.0, 0.0, 0.0])
>>> result = cxfm.act(shift, None, arr, cxc.cart3d, cxr.point, usys=usys)
>>> isinstance(result, jnp.ndarray)
True
```

## Decomposing Arrays Into CDicts

For chart conversion, first convert the array into a CDict using `cxc.cdict()` with a chart and unit:

```{code-block} python
>>> d = cxc.cdict(jnp.array([1.0, 2.0, 3.0]), "km", cxc.cart3d)
>>> sorted(d.keys())
['x', 'y', 'z']
```

Then convert charts via `pt_map`:

```{code-block} python
>>> d_sph = cxc.pt_map(d, cxc.cart3d, cxc.sph3d)
>>> sorted(d_sph.keys())
['phi', 'r', 'theta']
```

## The Upgrade Path

Arrays sit at the bottom of the coordinax tower. You can upgrade step-by-step:

### Array → Quantity

Attach units:

```{code-block} python
>>> arr = jnp.array([1.0, 2.0, 3.0])
>>> q = u.Q(arr, "km")
>>> q.unit
Unit("km")
```

### Quantity → Vector

Attach chart and representation:

```{code-block} python
>>> v = cx.Point.from_(q)
>>> v.chart
Cart3D(M=Rn(3))
```

### Vector → Coordinate

Attach a reference frame:

```{code-block} python
>>> coord = cx.Point.from_(v, cxf.alice)
>>> coord.frame
Alice()
```

### Shortcut: Array → Vector

Skip the quantity step:

```{code-block} python
>>> v = cx.Point.from_([1, 2, 3], "km")
>>> v.chart
Cart3D(M=Rn(3))
```

## JAX Integration

Plain arrays are native JAX — `jit`, `vmap`, and `grad` work without any special handling:

```{code-block} python
>>> usys = u.unitsystem("m", "s", "kg", "rad")
>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

>>> @jax.jit
... def rotate_array(x):
...     return cxfm.act(rot90z, None, x, cxc.cart3d, cxr.point, usys=usys)

>>> arr = jnp.array([1.0, 0.0, 0.0])
>>> result = rotate_array(arr)
>>> isinstance(result, jnp.ndarray)
True
```

## Comparison With Higher Levels

| Feature | Array | Quantity | CDict | Vector | Coordinate |
| --- | --- | --- | --- | --- | --- |
| Units | ✗ | ✓ | ✓ | ✓ | ✓ |
| Component names | ✗ | ✗ | ✓ | ✓ | ✓ |
| Chart | ✗ | ✗ | ✗ | ✓ | ✓ |
| Representation | ✗ | ✗ | ✗ | ✓ | ✓ |
| Frame | ✗ | ✗ | ✗ | ✗ | ✓ |
| `act` needs extra args | chart, rep, usys | (chart, rep) | chart, rep | none | none |
| `cconvert` / `to_frame` | ✗ | ✗ | ✗ | `cconvert` | both |

The further up the tower you go, the more metadata is attached and the fewer arguments you need to pass. Choose the level that matches your needs.
