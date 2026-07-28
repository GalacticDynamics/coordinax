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

The module currently provides packing helpers for converting component dictionaries to arrays and back.

These utilities are primarily useful when implementing downstream transforms, Jacobians, metric-like objects, or other chart-aware machinery that needs to preserve per-component physical units.

## Packing Helpers

```python
import unxt as u
import coordinax as cx
import coordinax.charts as cxc
from coordinax.internal import pack_uniform_unit

p = {"x": u.Q(1, "km"), "y": u.Q(200, "m"), "z": u.Q(3, "km")}

vals, unit = pack_uniform_unit(p, ("x", "y", "z"))
restored = cxc.cdict(vals, unit, ("x", "y", "z"))

qm = cx.carray(p, ("x", "y", "z"))
```

Use `pack_uniform_unit` when all components should be expressed in a shared unit before stacking into an array. Use `cx.carray` when each component should retain its own unit metadata (it returns a `QuantityMatrix` carrying a per-component unit tuple).

## Functional API

- `pack_uniform_unit`: stack a component dictionary into an array using a shared reference unit
- `cx.carray`: pack a component dictionary into a `QuantityMatrix` preserving per-component units

## Available Objects

- `pack_uniform_unit`: pack values into an array with one shared unit

## Notes

- This module is intended for advanced use and downstream integration, not as a stable top-level user API.
- The exported helpers are especially useful when chart components do not all share the same physical dimension.
- For stable end-user coordinate functionality, prefer the top-level `coordinax` API and its public submodules.

```{eval-rst}

.. currentmodule:: coordinax.internal

.. automodule:: coordinax.internal
    :exclude-members: aval, default, materialise, enable_materialise

```
