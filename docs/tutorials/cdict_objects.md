# Working With Component Dictionaries (CDict)

This tutorial covers **CDict** — coordinax's lightweight interchange format. A CDict is simply a `dict[str, Quantity]` mapping component names to their values. It is the common representation that all higher objects (`Point`, `Coordinate`) can be decomposed into, and that all lower objects (arrays, quantities) can be assembled from.

You will learn how to:

- Build CDicts by hand and from other types via `cdict()`
- Change coordinate system with `pt_map`
- Apply transforms with `act` (requires explicit chart and rep)
- Upgrade to `Point` or `Coordinate`
- Use CDicts with JAX

**Prerequisites**: [Working With Charts](../guides/charts.md).

```{admonition} Object Levels
:class: tip

Coordinax supports five levels of coordinate representation, each adding
more metadata. This tutorial covers `CDict`.

| Level | Type | See tutorial |
| --- | --- | --- |
| Coordinate | `Coordinate` | [Coordinate tutorial](./coordinate_objects.md) |
| Vector | `AbstractVector` | [Vector tutorial](./vector_objects.md) |
| **CDict** | `dict[str, Quantity]` | *this page* |
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
>>> import jax
>>> import jax.numpy as jnp
```

## What Is A CDict?

A CDict — short for **component dictionary** — is a plain Python dictionary with string keys (component names) and quantity or array values:

```{code-block} python
>>> d = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
>>> sorted(d.keys())
['x', 'y', 'z']

>>> d["x"]
Q(1, 'km')
```

CDicts carry **component names** but no chart, representation, or frame metadata. When you pass a CDict to functions like `act` or `pt_map`, you must supply the chart and representation explicitly.

## Building CDicts

### By Hand

Simply create a dictionary:

```{code-block} python
>>> d = {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
>>> sorted(d.keys())
['x', 'y', 'z']
```

### From A Quantity Via `cdict()`

`cxc.cdict()` splits a quantity's last axis into named components:

```{code-block} python
>>> d = cxc.cdict(u.Q([1, 2, 3], "km"))
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

### From An Array + Unit Via `cdict()`

```{code-block} python
>>> d = cxc.cdict(jnp.array([1.0, 2.0, 3.0]), "km", cxc.cart3d)
>>> sorted(d.keys())
['x', 'y', 'z']
```

### Identity Pass-Through

Passing an existing CDict returns it unchanged:

```{code-block} python
>>> d = {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
>>> cxc.cdict(d) is d
True
```

## Changing The Chart (Coordinate Conversion)

Use `cxc.pt_map()` with explicit source and target charts:

```{code-block} python
>>> d_cart = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}

>>> d_sph = cxc.pt_map(d_cart, cxc.cart3d, cxc.sph3d)
>>> sorted(d_sph.keys())
['phi', 'r', 'theta']
```

Round-tripping:

```{code-block} python
>>> d_back = cxc.pt_map(d_sph, cxc.sph3d, cxc.cart3d)
>>> sorted(d_back.keys())
['x', 'y', 'z']
```

## Using CDicts As Tangent Components

A CDict can also carry **tangent-vector components** when you provide that meaning explicitly to a manifold-level API.

For example, `cxm.angle_between()` interprets two CDicts as tangent vectors in the coordinate basis of a chart and uses the manifold metric to compute the angle between them at a base point:

```{code-block} python
>>> import coordinax.manifolds as cxm

>>> M = cxm.EuclideanManifold(2)
>>> at = {"x": u.Q(0, "m"), "y": u.Q(0, "m")}
>>> uvec = {"x": u.Q(1, "m"), "y": u.Q(0, "m")}
>>> vvec = {"x": u.Q(0, "m"), "y": u.Q(1, "m")}

>>> ang = cxm.angle_between(M, cxc.cart2d, uvec, vvec, at=at)
>>> jnp.allclose(u.ustrip("rad", ang), jnp.pi / 2)
Array(True, dtype=bool)
```

This does **not** mean that arbitrary point CDicts are automatically tangent vectors. The chart and manifold tell coordinax how to interpret the component data.

## Applying Transforms

Use `cxfm.act()` with explicit chart and representation. A CDict alone does not know which chart it belongs to, so you must provide that context:

```{code-block} python
>>> rot90z = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

>>> d = {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
>>> result = cxfm.act(rot90z, None, d, cxc.cart3d, cxr.point)
>>> sorted(result.keys())
['x', 'y', 'z']
```

Translation:

```{code-block} python
>>> shift = cxfm.Translate.from_([1, 2, 3], "km")

>>> d_origin = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
>>> result = cxfm.act(shift, None, d_origin, cxc.cart3d, cxr.point)
>>> result["x"]
Q(1, 'km')
>>> result["y"]
Q(2, 'km')
>>> result["z"]
Q(3, 'km')
```

Identity transform returns the exact same object:

```{code-block} python
>>> d = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
>>> result = cxfm.act(cxfm.Identity(), None, d)
>>> result is d
True
```

## Changing Tangent Basis

Use `cxr.change_basis()` when you want to change the tangent basis of a CDict without changing its chart, geometry, or semantics.

```{code-block} python
>>> d = {"x": 1.0, "y": 0.0}
>>> at = {"x": 2.0, "y": 3.0}

>>> cxr.change_basis(d, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at)
{'x': 1.0, 'y': 0.0}

>>> cxr.change_basis(d, cxc.cart2d, cxr.coord_disp, cxr.phys_disp, at=at)
{'x': 1.0, 'y': 0.0}
```

Use `cconvert()` instead when the chart itself must change.

## Upgrading To A Point

Promote a CDict to a `Point` by providing chart context:

```{code-block} python
>>> d = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}

>>> v = cx.Point.from_(d, cxc.cart3d)
>>> v.chart
Cart3D(M=Rn(3))

>>> isinstance(v, cx.Point)
True
```

With chart inference (keys are recognized as Cartesian):

```{code-block} python
>>> v = cx.Point.from_(d)
>>> v.chart
Cart3D(M=Rn(3))
```

## Upgrading To A Coordinate

Go all the way to a `Coordinate` by providing chart and frame:

```{code-block} python
>>> d = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}

>>> coord = cx.Point.from_(d, cxc.cart3d, cxf.alice)
>>> coord.chart
Cart3D(M=Rn(3))
>>> coord.frame
Alice()
```

## Data Access

CDicts support all standard dictionary operations:

```{code-block} python
>>> d = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}

>>> d["x"]
Q(1, 'km')

>>> sorted(d.keys())
['x', 'y', 'z']

>>> len(d)
3

>>> "x" in d
True
```

## JAX Integration

Python dicts are native JAX PyTrees, so CDicts work with `jit` and `vmap` directly:

```{code-block} python
>>> @jax.jit
... def rotate_cdict(d):
...     rot = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
...     return cxfm.act(rot, None, d, cxc.cart3d, cxr.point)

>>> import jax

>>> d = {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
>>> result = rotate_cdict(d)
>>> sorted(result.keys())
['x', 'y', 'z']
```

## When To Use CDict

Choose CDict when:

- You want a **lightweight intermediate format** without the overhead of constructing a full `Point`.
- You are building interop layers with non-coordinax code.
- You need standard dict operations (iteration, merging, filtering).
- You are inside a performance-sensitive inner loop where allocating `Point` objects is undesirable.

**Trade-off**: CDicts require you to pass chart and representation explicitly to every function call. If you find yourself repeating `cxc.cart3d, cxr.point` everywhere, upgrade to `Point`. See the [Vector tutorial](./vector_objects.md).

If you want even less overhead and are willing to manage units externally, use a bare `Quantity`. See the [Quantity tutorial](./quantity_objects.md).
