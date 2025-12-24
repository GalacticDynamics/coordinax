# `coordinax.transforms`

The `coordinax.transforms` module provides coordinate transformation functions.

## Overview

This module contains the low-level transformation functions that implement
mathematically precise coordinate conversions:

- **Point transforms**: Convert coordinates between different charts
- **Physical tangent transforms**: Convert velocity/acceleration vectors
- **Frame transforms**: Get orthonormal frames at points

## Quick Start

```python
import coordinax.charts as cxc
import coordinax.transforms as cxt
import unxt as u

# Transform a point from Cartesian to spherical (with Quantities)
p_cart = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
p_sph = cxt.point_transform(cxc.sph3d, cxc.cart3d, p_cart)

# Transform a point (with bare arrays - no units)
p_cart_arr = {"x": 1.0, "y": 2.0, "z": 3.0}
p_sph_arr = cxt.point_transform(
    cxc.sph3d, cxc.cart3d, p_cart_arr, usys=u.unitsystems.galactic
)

# Transform a velocity vector (requires base point)
v_cart = {"x": u.Q(10, "km/s"), "y": u.Q(20, "km/s"), "z": u.Q(30, "km/s")}
v_sph = cxt.physical_tangent_transform(cxc.sph3d, cxc.cart3d, v_cart, at=p_cart)
```

## Core Functions

### Point Transform

Transform point coordinates from one chart to another:

```python
p_new = cxt.point_transform(to_chart, from_chart, p)
```

The transformation is a coordinate-wise map that preserves the geometric
location while changing the coordinate description.

**Input types:**

The `CsDict` may contain either:

- **Quantities** (`unxt.Quantity`): Values with explicit units
- **Bare arrays**: JAX arrays, NumPy arrays, or Python scalars (dimensionless)

**Examples:**

```python
import coordinax as cx
import unxt as u

# With Quantities (explicit units)
p_qty = {"r": u.Q(10, "m")}
p_cart = cx.transforms.point_transform(cx.charts.cart1d, cx.charts.radial1d, p_qty)
# Result: {'x': Quantity(Array(10, dtype=int64, ...), unit='m')}

# With bare values (no units)
p_arr = {"r": 5}
p_cart = cx.transforms.point_transform(
    cx.charts.cart1d, cx.charts.radial1d, p_arr, usys=u.unitsystems.si
)
# Result: {'x': 5}
```

### Physical Tangent Transform

Transform tangent vectors (velocities, accelerations) between charts:

```python
v_new = cxt.physical_tangent_transform(to_chart, from_chart, v, at=p)
```

This uses the Jacobian of the coordinate transformation to correctly transform
physical components in orthonormal frames.

**Important**: The `at=` parameter specifies the base point where the tangent
vector is attached. This is required because the transformation depends on the
position.

### Frame at Point

Get the orthonormal frame vectors at a point:

```python
frame = cxt.frame_cart(chart, at=p)
```

Returns the basis vectors of the orthonormal frame expressed in Cartesian
coordinates.

### Cartesian Chart

Get the canonical Cartesian chart for a given chart:

```python
cart_chart = cxt.cartesian_chart(chart)
# e.g., cartesian_chart(sph3d) returns cart3d
```

## Transformation Laws

### Position (Point/Pos)

Point transforms are direct coordinate maps:

$$q'^i = \phi^i(q^1, \ldots, q^n)$$

### Velocity (Vel)

Velocity transforms use the Jacobian:

$$v'^i = \frac{\partial q'^i}{\partial q^j} v^j$$

### Acceleration (Acc)

Acceleration transforms include connection terms for correct geometric behavior.

```{eval-rst}

.. currentmodule:: coordinax.transforms

.. automodule:: coordinax.transforms
    :exclude-members: aval, default, materialise, enable_materialise

```
