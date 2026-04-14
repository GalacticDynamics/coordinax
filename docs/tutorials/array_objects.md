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

The trade-off: you must pass chart, representation, and a unit system to every coordinax function call. If you find yourself repeating the same metadata, upgrade to a `Quantity`.

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
