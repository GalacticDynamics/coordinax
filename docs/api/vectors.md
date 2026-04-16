# `coordinax.vectors`

The `coordinax.vectors` module provides vector objects for storing and transforming coordinate data with explicit chart and representation semantics.

## Overview

A **vector** stores three pieces: data (component values), chart (coordinate system), and representation (transformation law).

For design philosophy, practical patterns, and worked examples, see [Working With Vectors](../guides/vectors.md). For mathematical foundations, see [spec § Vectors](../spec.md#vectors).

## Quick Start

```python
import coordinax.main as cx
import coordinax.charts as cxc

# Construct and convert
v = cx.Point.from_([1, 2, 3], "m")
v_sph = cx.cconvert(v, cxc.sph3d)

# Arithmetic. Technically points are NOT displacements,
# but for convenience we allow subtraction to yield a new point.
v2 = cx.Point.from_([4, 5, 6], "m")
difference = v - v2

# Inspect converted components
components = cx.cdict(v_sph)
```

See [Working With Vectors](../guides/vectors.md#constructor-patterns) for all construction patterns and design rationale.

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

- **`Point`**: the primary vector class storing data + chart + representation
- **`AbstractVector`**: base class defining the vector interface
- **`Coordinate`**: a vector placed in a reference frame (see `coordinax.frames`)
- **`ToUnitsOptions`**: configuration for unit conversion behavior

## Design & Integration

For design philosophy, architecture, and immutability details, see [Working With Vectors](../guides/vectors.md#architecture--design-philosophy). For JAX integration patterns (PyTree, scalar-first design, vmap/jit/grad), see [Working With Vectors](../guides/vectors.md#jax-integration--scaling).

```{automodule} coordinax.vectors
:members:
:undoc-members:
:exclude-members: aval, default, materialise, enable_materialise
```
