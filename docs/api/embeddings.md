# `coordinax.embeddings`

The `coordinax.embeddings` module provides tools for working with embedded
manifolds.

## Overview

An **embedded manifold** is a lower-dimensional surface embedded in a
higher-dimensional ambient space. This module provides:

- Embedding and projection of points
- Tangent vector embedding and projection
- Induced metrics from the ambient space

## Quick Start

```python
import coordinax as cx
import coordinax.charts as cxc
import coordinax.embeddings as cxe
import unxt as u

# Create an embedded 2-sphere in 3D space
embed = cxe.EmbeddedManifold(
    intrinsic_chart=cxc.twosphere,
    ambient_chart=cxc.cart3d,
    params={"R": u.Q(1.0, "km")},
)

# Embed a point from intrinsic coordinates to ambient space
p_intrinsic = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.5, "rad")}
p_ambient = cxe.embed_point(embed, p_intrinsic)

# Project back to intrinsic coordinates
p_back = cxe.project_point(embed, p_ambient)

# Embed tangent vectors
v_intrinsic = {"theta": u.Q(1.0, "rad/s"), "phi": u.Q(0.5, "rad/s")}
v_ambient = cxe.embed_tangent(embed, v_intrinsic, at=p_intrinsic)
```

## Embedded Manifold

The `EmbeddedManifold` class represents a manifold embedded in an ambient space:

```python
embed = cxe.EmbeddedManifold(
    intrinsic_chart=cxc.twosphere,  # Chart on the manifold
    ambient_chart=cxc.cart3d,  # Chart in the ambient space
    params={"R": u.Q(1.0, "km")},  # Parameters (e.g., radius)
)
```

## Core Functions

### Point Operations

- `embed_point(manifold, p)`: Map intrinsic coordinates to ambient space
- `project_point(manifold, p)`: Map ambient coordinates to intrinsic space

### Tangent Vector Operations

- `embed_tangent(manifold, v, *, at)`: Pushforward tangent vectors to ambient
- `project_tangent(manifold, v, *, at)`: Project ambient tangents to manifold

## Example: 2-Sphere Embedding

```python
import coordinax.charts as cxc
import coordinax.embeddings as cxe
import unxt as u

# Create a unit sphere
sphere = cxe.EmbeddedManifold(
    intrinsic_chart=cxc.twosphere,
    ambient_chart=cxc.cart3d,
    params={"R": u.Q(1.0, "km")},
)

# Point on the sphere (theta=pi/4, phi=0)
p = {"theta": u.Angle(0.785, "rad"), "phi": u.Angle(0.0, "rad")}

# Embed to 3D Cartesian
p_3d = cxe.embed_point(sphere, p)
# Result: x ≈ 0.707 km, y = 0, z ≈ 0.707 km
```

```{eval-rst}

.. currentmodule:: coordinax.embeddings

.. automodule:: coordinax.embeddings
    :exclude-members: aval, default, materialise, enable_materialise

```
