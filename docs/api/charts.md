# `coordinax.charts`

The `coordinax.charts` module provides coordinate charts for representing points
in different coordinate systems.

## Overview

A **chart** is a coordinate system that describes how to represent points using
a specific set of coordinates. Charts define:

- Component names (e.g., `x, y, z` for Cartesian; `r, theta, phi` for spherical)
- Coordinate dimensions (e.g., `length`, `angle`)
- Transformation rules between charts

## Quick Start

```python
import coordinax.charts as cxc
import coordinax.transforms as cxt
import unxt as u

# Use predefined chart instances
cart = cxc.cart3d  # Cartesian3D
sph = cxc.sph3d  # Spherical3D

# Transform coordinates between charts
p = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
p_sph = cxt.point_transform(sph, cart, p)
```

## Available Charts

### 0D Charts

- `Cart0D` / `cart0d`: Zero-dimensional Cartesian (scalar)

### 1D Charts

- `Cart1D` / `cart1d`: 1D Cartesian
- `Radial1D` / `radial1d`: Radial distance

### 2D Charts

- `Cart2D` / `cart2d`: 2D Cartesian
- `Polar2D` / `polar2d`: Polar coordinates
- `TwoSphere` / `twosphere`: 2-sphere (θ, φ)

### 3D Charts

- `Cart3D` / `cart3d`: 3D Cartesian
- `Cylindrical3D` / `cyl3d`: Cylindrical coordinates
- `Spherical3D` / `sph3d`: Spherical coordinates (physics convention)
- `LonLatSpherical3D` / `lonlatsph3d`: Longitude/latitude spherical
- `LonCosLatSpherical3D` / `loncoslatsph3d`: Lon/cos(lat) spherical
- `MathSpherical3D` / `mathsph3d`: Mathematical spherical convention

### N-D Charts

- `CartND` / `cartnd`: N-dimensional Cartesian
- `SpaceTimeCT`: Spacetime with c\*t convention
- `SpaceTimeEuclidean`: Euclidean spacetime

```{eval-rst}

.. currentmodule:: coordinax.charts

.. automodule:: coordinax.charts
    :exclude-members: aval, default, materialise, enable_materialise

```
