# Vectors

This guide covers vector objects in `coordinax`. A vector is _data_ plus a
representation (coordinate chart) and a role (Pos, Vel, PhysAcc, ...).

## Creating Vector Objects

```
import coordinax as cx
import unxt as u

q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    chart=cxc.cart3d,
    role=cxr.phys_disp,
)

v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    chart=cxc.cart3d,
    role=cxr.phys_vel,
)
```

If you already have array-valued data, you can construct vectors directly from
quantity-valued components:

```
import coordinax as cx
import unxt as u

q3 = cx.Vector(
    data={
        "x": u.Q([1.0, 2.0], "kpc"),
        "y": u.Q([3.0, 4.0], "kpc"),
        "z": u.Q([5.0, 6.0], "kpc"),
    },
    chart=cxc.cart3d,
    role=cxr.phys_disp,
)
v3 = cx.Vector(
    data={
        "x": u.Q([4.0, 5.0], "kpc/Myr"),
        "y": u.Q([6.0, 7.0], "kpc/Myr"),
        "z": u.Q([8.0, 9.0], "kpc/Myr"),
    },
    chart=cxc.cart3d,
    role=cxr.phys_vel,
)
```

## Arithmetic and Mathematical Operations

Vector objects are compatible with JAX primitives; see the operators guide for
examples.

```
import coordinax as cx
import unxt as u

q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    chart=cxc.cart3d,
    role=cxr.phys_disp,
)
v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    chart=cxc.cart3d,
    role=cxr.phys_vel,
)

# q and v are ready for use in JAX operations.
```

## Dimensionality: 1D, 2D, 3D

Representations are available for multiple dimensions:

- 1D: `cxc.cart1d`, `cxc.radial1d`
- 2D: `cxc.cart2d`, `cxc.polar2d`
- 3D: `cxc.cart3d`, `cxc.cyl3d`, `cxc.sph3d`

## Conversion Between Representations

Vectors can be converted between coordinate systems:

```
import coordinax as cx
import unxt as u

q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    chart=cxc.cart3d,
    role=cxr.phys_disp,
)
v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    chart=cxc.cart3d,
    role=cxr.phys_vel,
)

q_sph = q.vconvert(cxc.sph3d)
v_sph = v.vconvert(cxc.sph3d, q)
```

## Batch and Broadcast Operations

Component values can be arrays, enabling batch and broadcast behavior:

```
import coordinax as cx
import unxt as u

arr = cx.Vector(
    data={
        "x": u.Q([1.0, 4.0], "kpc"),
        "y": u.Q([2.0, 5.0], "kpc"),
        "z": u.Q([3.0, 6.0], "kpc"),
    },
    chart=cxc.cart3d,
    role=cxr.phys_disp,
)
# arr can be used in batched JAX computations.
```

## Space Objects: Grouping Related Vectors

A {class}`~coordinax.PointedVector` object collects related vectors (e.g.,
position, velocity, acceleration) anchored at a common base point:

```
import coordinax as cx
import unxt as u

q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    chart=cxc.cart3d,
    role=cxr.phys_disp,
)
v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    chart=cxc.cart3d,
    role=cxr.phys_vel,
)

space = cx.PointedVector(base=q, speed=v)
```

---

:::{seealso}

[API Documentation for Objects](../api/objs.md)

:::
