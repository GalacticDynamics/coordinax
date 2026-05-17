# `coordinax.charts`

The `coordinax.charts` module provides chart objects and chart-level coordinate maps for representing points on manifolds.

## Overview

A **chart** defines how point coordinates are represented:

- Component names (for example `x, y, z` or `r, theta, phi`)
- Coordinate dimensions (for example `length`, `angle`)
- Transition and realization behavior through the functional API

Use chart instances (for example `cart3d`, `sph3d`) when transforming concrete coordinate data.

For a step-by-step walkthrough, see [Working With Charts](../guides/charts.md).

## Quick Start

```python
import coordinax.charts as cxc
import unxt as u

# Use predefined chart instances.
cart = cxc.cart3d
sph = cxc.sph3d

# Same-manifold chart transition.
p = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
p_sph = cxc.pt_map(p, cart, sph)

# General realization map (defaults to transition map for same manifold charts).
p_sph2 = cxc.pt_map(p, cart, sph)

# Pick canonical Cartesian chart without transforming coordinates.
cart_of_sph = cxc.cartesian_chart(sph)
```

## Functional API

- `cartesian_chart`: return a chart's canonical Cartesian chart
- `guess_chart`: infer a chart from keys or array/quantity trailing shape
- `cdict`: normalize inputs to component dictionaries
- `pt_map`: transform points between charts on the same or different manifolds
- `jac_pt_map`: compute the Jacobian of a point map between charts on the same or different manifolds

## Available Objects

### Chart Families

The module exports both concrete chart classes and predefined instances.

### 0D Charts

- `Cart0D` / `cart0d`: Zero-dimensional Cartesian (scalar)

### 1D Charts

- `Cart1D` / `cart1d`: 1D Cartesian
- `Radial1D` / `radial1d`: Radial distance
- `Time1D` / `time1d`: 1D time chart

### 2D Charts

- `Cart2D` / `cart2d`: 2D Cartesian
- `Polar2D` / `polar2d`: Polar coordinates
- `SphericalTwoSphere` / `sph2`: 2-sphere (`theta`, `phi`)
- `LonLatSphericalTwoSphere` / `lonlat_sph2`: 2-sphere (`lon`, `lat`)
- `LonCosLatSphericalTwoSphere` / `loncoslat_sph2`: 2-sphere (`lon_coslat`, `lat`)
- `MathSphericalTwoSphere` / `math_sph2`: mathematical 2-sphere convention

Intrinsic two-sphere charts do not have a global Cartesian 2D chart; requesting `cartesian_chart(...)` on this family raises `NoGlobalCartesianChartError`.

### 3D Charts

- `Cart3D` / `cart3d`: 3D Cartesian
- `Cylindrical3D` / `cyl3d`: Cylindrical coordinates
- `Spherical3D` / `sph3d`: Spherical coordinates (physics convention)
- `LonLatSpherical3D` / `lonlat_sph3d`: Longitude/latitude spherical
- `LonCosLatSpherical3D` / `loncoslat_sph3d`: Lon/cos(lat) spherical
- `MathSpherical3D` / `math_sph3d`: Mathematical spherical convention
- `ProlateSpheroidal3D`: Prolate spheroidal chart with required `Delta` parameter

`ProlateSpheroidal3D` does not export a predefined instance because chart instances depend on the focal parameter `Delta`.

### 4D Charts

- `MinkowskiCT` / `minkowskict`: 4D Minkowski spacetime chart with the conventional 3+1 split. Components are `(ct, x, y, z)`, all with dimension `"length"`. This is a single (non-product) flat chart whose metric is $\eta = \operatorname{diag}(-1, 1, 1, 1)$. The chart is self-Cartesian: `cartesian_chart(minkowskict) is minkowskict`.

### 6D Charts

- `PoincarePolar6D` / `poincarepolar6d`: 6D Poincare polar chart family

### N-D Charts

- `CartND` / `cartnd`: N-dimensional Cartesian

### Product Charts

- `CartesianProductChart`: namespace-prefixed product chart with dot-delimited component keys (for example `q.x`, `q.y`, `p.x`, ...)

Product-chart transitions are factorwise: each factor chart transforms independently and then components are merged.

## Notes

- Use chart instances (for example `cart3d`, `sph3d`) as conversion arguments.
- Intrinsic two-sphere charts do not define a global Cartesian 2D chart.

```{eval-rst}

.. currentmodule:: coordinax.charts

.. automodule:: coordinax.charts
    :exclude-members: aval, default, materialise, enable_materialise

```
