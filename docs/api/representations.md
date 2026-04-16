# `coordinax.representations`

The `coordinax.representations` module defines representation descriptors for geometric data and the representation-aware conversion API.

## Overview

A representation is the triple $R = (K, B, S)$:

- $K$: geometry kind (what sort of geometric object the data represents)
- $B$: basis kind (how components are interpreted as basis components)
- $S$: semantic kind (what the object means physically/mathematically)

This is separate from charts:

- Charts define coordinate systems.
- Representations define geometric meaning and transformation law category.

In the current design, the primary built-in representation is point data.

## Quick Start

```python
import coordinax.charts as cxc
import coordinax.representations as cxr

# Canonical point representation.
rep = cxr.point

# Convert point data between charts while preserving representation.
p = {"x": 1.0, "y": 2.0, "z": 3.0}
q = cxr.cconvert(p, cxc.cart3d, rep, cxc.sph3d, rep)

# Build a reusable conversion map.
to_sph = cxr.cmap(cxc.cart3d, cxr.point, cxc.sph3d)
q2 = to_sph(p)
```

## Functional API

- `cconvert`: representation-aware coordinate conversion API
- `cmap`: partial-function builder around `cconvert`
- `guess_basis_kind`: infer basis kind from dimensions/data
- `guess_geometry_kind`: infer geometric kind from dimensions/data
- `guess_rep`: infer full representation from dimensions/data
- `guess_semantic_kind`: infer semantic kind from dimensions/data

For point data, `cconvert` dispatches through chart-level point conversion laws.

## Available Objects

### Conversion Functions

- `cconvert`: convert data across charts/representations
- `cmap`: build reusable conversion callables
- `guess_basis_kind`: infer a basis kind
- `guess_geometry_kind`: infer a geometry kind
- `guess_rep`: infer a full representation
- `guess_semantic_kind`: infer a semantic kind

### Representation Descriptor

- `Representation`: immutable descriptor `(geom_kind, basis, semantic_kind)`
- `point`: canonical point representation

`point` is equivalent to `(point_geom, no_basis, loc)`.

## Geometry Kind

- `AbstractGeometry`: base class for geometric kind descriptors
- `PointGeometry` / `point_geom`: point geometric kind

## Basis Kind

- `AbstractBasis`: base class for basis descriptors
- `NoBasis` / `no_basis`: basis kind used for affine point data

## Semantic Kind

- `AbstractSemanticKind`: base class for semantic descriptors
- `Location` / `loc`: location semantic kind

## Notes

- Representations are orthogonal to charts: chart choice and representation choice are independent concerns.
- The current built-in flow is point-first, centered on `(point_geom, no_basis, loc)`.

```{eval-rst}

.. currentmodule:: coordinax.representations

.. automodule:: coordinax.representations
    :exclude-members: aval, default, materialise, enable_materialise

```
