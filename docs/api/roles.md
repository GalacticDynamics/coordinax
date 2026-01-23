# `coordinax.roles`

The `coordinax.roles` module provides role flags that define the physical
meaning of vectors.

## Overview

A **role** defines what a vector represents mathematically and physically:

- **Point**: A location in space (affine, not a vector space element)
- **Pos** (Position): A displacement vector from an origin
- **Vel** (Velocity): A tangent vector representing rate of change of position
- **PhysAcc** (Acceleration): A tangent vector representing rate of change of
  velocity

Roles determine transformation laws: positions transform as points, while
velocities transform using the Jacobian of the coordinate transformation.

## Quick Start

```python
import coordinax as cx
import coordinax.roles as cxr

# Use predefined role instances
point = cxr.point  # Point role
pos = cxr.phys_disp  # Position role
vel = cxr.phys_vel  # Velocity role
acc = cxr.phys_acc  # Acceleration role

# Or instantiate classes directly
point_role = cxr.Point()
pos_role = cxr.PhysDisp()
```

## Role Hierarchy

```
AbstractRole
├── Point         # Affine location (transforms via point_transform)
└── AbstractPhysicalRole
    ├── PhysDisp       # Position/displacement (transforms via point_transform)
    ├── PhysVel       # Velocity (transforms via physical_tangent_transform)
    └── PhysAcc       # Acceleration (transforms via physical_tangent_transform)
```

## Role Semantics

### Point vs Pos

- **Point**: An absolute location; cannot be added to other points
- **Pos**: A displacement from an origin; forms a vector space

Use `as_pos(point)` to convert a Point to a PhysDisp (interpreting the point as a
displacement from the origin).

### Physical Roles (Vel, PhysAcc)

Velocity and acceleration are **tangent vectors**. They transform using the
Jacobian of the coordinate transformation, not by simple coordinate conversion.

```python
import coordinax as cx
import unxt as u

q = cx.Vector.from_(
    {"x": u.Q(1, "kpc"), "y": u.Q(2, "kpc"), "z": u.Q(3, "kpc")},
    cx.charts.cart3d,
    cx.roles.phys_disp,
)
v = cx.Vector.from_(
    {"x": u.Q(10, "km/s"), "y": u.Q(20, "km/s"), "z": u.Q(30, "km/s")},
    cx.charts.cart3d,
    cx.roles.phys_vel,
)

# Position converts directly
q_sph = q.vconvert(cx.charts.sph3d)

# Velocity requires the base point for correct transformation
v_sph = v.vconvert(cx.charts.sph3d, q)
```

```{eval-rst}

.. currentmodule:: coordinax.roles

.. automodule:: coordinax.roles
    :exclude-members: aval, default, materialise, enable_materialise

```
