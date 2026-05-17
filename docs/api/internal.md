# `coordinax.internal`

```{warning}
Everything in `coordinax.internal` is **semi-public**. The APIs exposed
here are usable by downstream packages but are **not** covered by the
same stability guarantees as the top-level `coordinax` API. Names,
signatures, and behaviour may change **at any time without warning** in
minor or patch releases. Pin to an exact version if you depend on
anything here.
```

`coordinax.internal` re-exports selected internal utilities that are useful for advanced users and downstream library authors, but whose interfaces are not yet stable enough for the main public API.

## Overview

The module currently provides two kinds of semi-public helpers:

- heterogeneous unit containers for vectors and matrices
- packing helpers for converting component dictionaries to arrays and back

These utilities are primarily useful when implementing downstream transforms, Jacobians, metric-like objects, or other chart-aware machinery that needs to preserve per-component physical units.

## Quick Start

```python
import jax.numpy as jnp
import unxt as u
from coordinax.internal import QMatrix

J = QMatrix(
    value=jnp.eye(3),
    unit=(
        (u.unit("m/m"), u.unit("m/rad"), u.unit("m/rad")),
        (u.unit("rad/m"), u.unit("rad/rad"), u.unit("rad/rad")),
        (u.unit("rad/m"), u.unit("rad/rad"), u.unit("rad/rad")),
    ),
)
```

`QMatrix` supports both 1-D and 2-D cases. This makes it suitable for heterogeneous vectors as well as Jacobians and metric tensors whose entries do not all share the same unit.

## Packing Helpers

```python
import unxt as u
import coordinax.charts as cxc
from coordinax.internal import pack_nonuniform_unit, pack_uniform_unit

p = {"x": u.Q(1, "km"), "y": u.Q(200, "m"), "z": u.Q(3, "km")}

vals, unit = pack_uniform_unit(p, ("x", "y", "z"))
restored = cxc.cdict(vals, unit, ("x", "y", "z"))

vals2, units2 = pack_nonuniform_unit(p, ("x", "y", "z"))
```

Use `pack_uniform_unit` when all components should be expressed in a shared unit before stacking into an array. Use `pack_nonuniform_unit` when each component should retain its own unit metadata.

## Functional API

- `cdict_units`: extract per-key units from a component dictionary
- `pack_uniform_unit`: stack a component dictionary into an array using a shared reference unit
- `pack_nonuniform_unit`: stack a component dictionary into an array while preserving a per-component unit tuple

## Available Objects

### Heterogeneous Unit Containers

- `QMatrix`: N-D quantity container with per-element units; currently supports 1-D vectors and 2-D matrices
- `UnitsMatrix`: immutable nested tuple of units with tuple-style indexing and shape metadata

### Packing Utilities

- `cdict_units`: unit introspection helper for component dictionaries
- `pack_uniform_unit`: pack values into an array with one shared unit
- `pack_nonuniform_unit`: pack values into an array with per-component units

## Notes

- This module is intended for advanced use and downstream integration, not as a stable top-level user API.
- The exported helpers are especially useful when chart components do not all share the same physical dimension.
- For stable end-user coordinate functionality, prefer the top-level `coordinax` API and its public submodules.

```{eval-rst}

.. currentmodule:: coordinax.internal

.. automodule:: coordinax.internal
    :exclude-members: aval, default, materialise, enable_materialise

```
