# `coordinax.roles`

The `coordinax.roles` module provides role flags that define the physical
meaning of vectors.

## Overview

A **role** defines what a vector represents mathematically and physically:

- **Point**: A location in space (affine, not a vector space element)
- **PhysDisp**: A physical displacement (tangent) vector in an orthonormal frame
- **PhysVel**: A physical velocity (tangent) vector in an orthonormal frame
- **PhysAcc**: A physical acceleration (tangent) vector in an orthonormal frame
- **CoordDisp / CoordVel / CoordAcc**: Coordinate-basis tangent components in a
  chart basis

Roles determine transformation laws: positions transform as points, while
tangent vectors transform using the Jacobian (and, for physical roles, the local
orthonormal frame), evaluated at a base point `at=`.

## Quick Start

```python
import coordinax as cx
import coordinax.roles as cxr

# Use predefined role instances
point = cxr.point  # Point role
pos = cxr.phys_disp  # Physical displacement role
vel = cxr.phys_vel  # Velocity role
acc = cxr.phys_acc  # Acceleration role
coord_vel = cxr.coord_vel  # Coordinate-basis velocity role

# Or instantiate classes directly
point_role = cxr.Point()
pos_role = cxr.PhysDisp()
```

## Role Hierarchy

```
AbstractRole
├── Point         # Affine location (transforms via point_transform)
├── AbstractPhysRole
│   ├── PhysDisp       # Physical displacement (physical_tangent_transform; needs at=)
│   ├── PhysVel        # Physical velocity (physical_tangent_transform; needs at=)
│   └── PhysAcc        # Physical acceleration (physical_tangent_transform; needs at=)
└── AbstractCoordRole
    ├── CoordDisp      # Coordinate-basis displacement (coord_transform; needs at=)
    ├── CoordVel       # Coordinate-basis velocity (coord_transform; needs at=)
    └── CoordAcc       # Coordinate-basis acceleration (coord_transform; needs at=)
```

## Role Semantics

### Point vs PhysDisp

- **Point**: An absolute location; cannot be added to other points
- **PhysDisp**: A physical displacement vector in a tangent space; can translate
  a point

Use `as_disp(point)` to convert a Point to a PhysDisp (interpreting the point as
a displacement from the origin).

### Physical Roles (Vel, PhysAcc)

Velocity and acceleration are **tangent vectors**. They transform using the
Jacobian of the coordinate transformation, not by simple coordinate conversion.

```python
import coordinax.charts as cxc
import coordinax.roles as cxr
from coordinax.objs import Vector, vconvert
import unxt as u
```

```python
p = Vector.from_(
    {"x": u.Q(1, "kpc"), "y": u.Q(2, "kpc"), "z": u.Q(3, "kpc")},
    cxc.cart3d,
    cxr.point,
)
v = Vector.from_(
    {"x": u.Q(10, "km/s"), "y": u.Q(20, "km/s"), "z": u.Q(30, "km/s")},
    cxc.cart3d,
    cxr.phys_vel,
)

# Velocity requires the base point for correct transformation
v_sph = v.vconvert(cxc.sph3d, p)
```

### Coordinate-Basis Roles

Coordinate-basis tangent roles (`CoordDisp`, `CoordVel`, `CoordAcc`) transform
by the Jacobian pushforward and also require a base point `at=`:

```python
at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}
v = {"x": u.Q(2.0, "m/s"), "y": u.Q(0.0, "m/s")}

v_pol = vconvert(cxr.coord_vel, cxc.polar2d, cxc.cart2d, v, at=at)
```

```{eval-rst}

.. currentmodule:: coordinax.roles

.. automodule:: coordinax.roles
    :exclude-members: aval, default, materialise, enable_materialise

```
