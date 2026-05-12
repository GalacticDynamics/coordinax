# `coordinax.vectors`

The `coordinax.vectors` module provides vector objects for storing and transforming coordinate data with explicit chart and representation semantics.

## Overview

A **vector** stores three pieces: data (component values), chart (coordinate system), and representation (transformation law).

For design philosophy, practical patterns, and worked examples, see [Working With Vectors](../guides/vectors.md). For mathematical foundations, see [spec § Vectors](../spec.md#vectors).

## Quick Start

```python
import coordinax.main as cx
import coordinax.charts as cxc
import coordinax.representations as cxr
import unxt as u

# ── Point ──────────────────────────────────────────────────
p = cx.Point.from_([1, 2, 3], "m")
p_sph = cx.cconvert(p, cxc.sph3d)

# ── Tangent ────────────────────────────────────────────────
# A velocity vector — transforms by Jacobian pushforward
v = cx.Tangent.from_(
    {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    cxc.cart3d,
    cxr.coord_vel,
)
# Convert a Tangent — must supply the base Point via `at=`
v_sph = v.cconvert(cxc.sph3d, at=p)

# ── Coordinate ─────────────────────────────────────────────
# Bundle: base Point + named Tangent fibre fields
pv = cx.Coordinate(point=p, velocity=v)
pv_sph = pv.cconvert(cxc.sph3d)  # converts point AND velocity together
```

See [Working With Vectors](../guides/vectors.md) and [Working With Tangent Vectors](../guides/tangents.md) for all construction patterns and design rationale.

## Functional API

### Constructors & Conversion

- `Point.from_`: flexible multiple-dispatch constructor from arrays, quantities, or dictionaries
- `cconvert`: coordinate conversion between charts (representation-aware)
- `uconvert`: unit conversion

### Shape & Structure

- `flatten()`: flatten all components to a 1D view
- `reshape(*shape)`: reshape components while preserving chart semantics
- `__getitem__()`: slice and index vectors

### Arithmetic & Operations

Vectors support JAX-style arithmetic via `quax` operators:

- `+`, `-`: vector addition and subtraction
- `*`, `/`: scalar multiplication and division
- `norm()`: Euclidean norm of components
- `copy()`: create a copy (via `dataclass.replace`)

Additional utilities:

- `astype(dtype)`: cast components to a new dtype
- `round(decimals)`: round components
- `to_device(device)`: move to a new device

## Available Objects

- **`Point`**: a geometric point storing data + chart + representation (always `PointGeometry`)
- **`Tangent`**: a tangent-space vector with explicit basis and semantic kind (velocity, displacement, acceleration)
- **`Coordinate`**: a vector bundle — a `Point` paired with named `Tangent` fibre fields anchored at that point
- **`AbstractVector`**: base class defining the vector interface
- **`ToUnitsOptions`**: configuration for unit conversion behavior

## Design & Integration

For design philosophy, architecture, and immutability details, see [Working With Vectors](../guides/vectors.md#architecture--design-philosophy). For tangent-vector patterns (basis, semantic kind, Jacobian pushforward), see [Working With Tangent Vectors](../guides/tangents.md). For JAX integration patterns (PyTree, scalar-first design, vmap/jit/grad), see [Working With Vectors](../guides/vectors.md#jax-integration--scaling).

```{automodule} coordinax.vectors
:members:
:undoc-members:
:exclude-members: aval, default, materialise, enable_materialise
```
