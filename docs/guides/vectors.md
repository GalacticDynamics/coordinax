# Vectors

This guide covers vector objects in `coordinax`. A vector is _data_ plus a
representation (coordinate chart) and a role (Pos, Vel, Acc, ...).

## Creating Vector Objects

```python
import coordinax as cx
import unxt as u

q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    rep=cx.r.cart3d,
    role=cx.r.Pos(),
)

v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    rep=cx.r.cart3d,
    role=cx.r.Vel(),
)
```

If you already have array-valued data, you can construct vectors directly from
quantity-valued components:

```python
import coordinax as cx
import unxt as u

q3 = cx.Vector(
    data={
        "x": u.Q([1.0, 2.0], "kpc"),
        "y": u.Q([3.0, 4.0], "kpc"),
        "z": u.Q([5.0, 6.0], "kpc"),
    },
    rep=cx.r.cart3d,
    role=cx.r.Pos(),
)
v3 = cx.Vector(
    data={
        "x": u.Q([4.0, 5.0], "kpc/Myr"),
        "y": u.Q([6.0, 7.0], "kpc/Myr"),
        "z": u.Q([8.0, 9.0], "kpc/Myr"),
    },
    rep=cx.r.cart3d,
    role=cx.r.Vel(),
)
```

## Arithmetic and Mathematical Operations

Vector objects are compatible with JAX primitives; see the operators guide for
examples.

```python
import coordinax as cx
import unxt as u

q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    rep=cx.r.cart3d,
    role=cx.r.Pos(),
)
v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    rep=cx.r.cart3d,
    role=cx.r.Vel(),
)

# q and v are ready for use in JAX operations.
```

## Dimensionality: 1D, 2D, 3D

Representations are available for multiple dimensions:

- 1D: `cx.r.cart1d`, `cx.r.radial1d`
- 2D: `cx.r.cart2d`, `cx.r.polar2d`
- 3D: `cx.r.cart3d`, `cx.r.cyl3d`, `cx.r.sph3d`

## Conversion Between Representations

Vectors can be converted between coordinate systems:

```python
import coordinax as cx
import unxt as u

q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    rep=cx.r.cart3d,
    role=cx.r.Pos(),
)
v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    rep=cx.r.cart3d,
    role=cx.r.Vel(),
)

q_sph = q.vconvert(cx.r.sph3d)
v_sph = v.vconvert(cx.r.sph3d, q)
```

## Batch and Broadcast Operations

Component values can be arrays, enabling batch and broadcast behavior:

```python
import coordinax as cx
import unxt as u

arr = cx.Vector(
    data={
        "x": u.Q([1.0, 4.0], "kpc"),
        "y": u.Q([2.0, 5.0], "kpc"),
        "z": u.Q([3.0, 6.0], "kpc"),
    },
    rep=cx.r.cart3d,
    role=cx.r.Pos(),
)
# arr can be used in batched JAX computations.
```

## Space Objects: Grouping Related Vectors

A {class}`~coordinax.vecs.KinematicSpace` object collects related vectors (e.g.,
position, velocity, acceleration):

```python
import coordinax as cx
import unxt as u

q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    rep=cx.r.cart3d,
    role=cx.r.Pos(),
)
v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    rep=cx.r.cart3d,
    role=cx.r.Vel(),
)

space = cx.KinematicSpace(length=q, speed=v)
```

---

:::{seealso}

[API Documentation for Vectors](../api/vecs.md)

:::
