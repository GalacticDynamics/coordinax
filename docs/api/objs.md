# `coordinax` (objects)

The top-level `coordinax` module provides high-level container objects for
vectors and coordinates.

## Overview

This module contains the main user-facing objects:

- **Vector**: A geometric vector with data, chart, and role
- **PointedVector**: A collection of related vectors anchored at a common point
- **Coordinate**: A vector attached to a reference frame

## Quick Start

```python
import coordinax as cx
import unxt as u

# Create a position vector
q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    chart=cx.charts.cart3d,
    role=cx.roles.point,
)

# Create a velocity vector
v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    chart=cx.charts.cart3d,
    role=cx.roles.phys_vel,
)

# Convert to spherical coordinates
q_sph = q.vconvert(cx.charts.sph3d)
v_sph = v.vconvert(cx.charts.sph3d, q)

# Group related vectors
space = cx.PointedVector(base=q, speed=v)

# Attach to a frame
import coordinax.frames as cxf

coord = cx.Coordinate({"base": q, "speed": v}, frame=cxf.ICRS())
```

## Vector

The `Vector` class is the primary object for representing geometric vectors:

```python
# From explicit components
q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    chart=cx.charts.cart3d,
    role=cx.roles.point,
)

# From array (auto-detect chart and role)
q = cx.Vector.from_([1, 2, 3], "kpc")

# From array with explicit chart and role
v = cx.Vector.from_([10, 20, 30], "km/s", cx.charts.cart3d, cx.roles.phys_vel)

a = cx.Vector.from_([0.1, 0.2, 0.3], "m/s^2", cx.charts.cart3d, cx.roles.phys_acc)

# Access components
print(q["x"])  # Quantity
print(q.data)  # Full data dict
print(q.chart)  # Chart instance
print(q.role)  # Role instance
```

### Vector Methods

- `vconvert(to_chart, at=None)`: Convert to a different chart
- `uconvert(units)`: Convert units of components

## PointedVector

The `PointedVector` class groups related vectors (position, velocity,
acceleration) at a common point:

```python
space = cx.PointedVector(base=q, speed=v, acceleration=a)

# Access vectors
space.base  # position
space["speed"]  # velocity
```

## Coordinate

The `Coordinate` class attaches vectors to a reference frame:

```python
import coordinax.frames as cxf

coord = cx.Coordinate({"base": q, "speed": v}, frame=cxf.ICRS())

# Transform to another frame
coord_gc = coord.to_frame(cxf.Galactocentric())

# Convert representation within frame
coord_sph = coord.vconvert(cx.charts.sph3d)
```

## vconvert Function

The top-level `vconvert` function provides flexible coordinate conversion:

```python
# Convert Vector
q_sph = cx.vconvert(cx.charts.sph3d, q)

# Convert velocity (needs base point)
v_sph = cx.vconvert(cx.charts.sph3d, v, q)
```

## as_pos Function

Convert a Point role to a PhysDisp role (interpret location as displacement from
origin):

```python
pos = cx.as_pos(q)  # q is a Point vector defined above
```

```{eval-rst}

.. currentmodule:: coordinax

.. automodule:: coordinax
    :exclude-members: aval, default, materialise, enable_materialise
    :noindex:

```
