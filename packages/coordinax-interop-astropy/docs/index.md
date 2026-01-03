# coordinax-interop-astropy

Astropy interoperability for coordinax.

## Overview

`coordinax-interop-astropy` provides seamless conversion between coordinax
vectors, frames, distances and Astropy coordinate representations, frames, and
quantities. This package enables you to work with both libraries
interchangeably.

## Installation

```bash
pip install coordinax-interop-astropy
```

Or with uv:

```bash
uv add coordinax-interop-astropy
```

## Features

- **Position vectors**: Convert between coordinax position vectors and Astropy
  `BaseRepresentation` objects
- **Velocity vectors**: Convert between coordinax velocity vectors and Astropy
  `BaseDifferential` objects
- **Distance types**: Convert between coordinax distance types (`Distance`,
  `Parallax`, `DistanceModulus`) and Astropy `Quantity` objects
- **Reference frames**: Convert between coordinax-astro frames (`ICRS`,
  `Galactocentric`) and Astropy coordinate frames
- **Quantity arrays**: Create coordinax vectors from Astropy `Quantity` arrays

## Quick Start

### Converting Astropy to coordinax

```python
from plum import convert
import astropy.coordinates as apyc
import coordinax.vecs as cxv

# Create an Astropy representation
cart = apyc.CartesianRepresentation(1, 2, 3, unit="km")

# Convert to coordinax
vec = convert(cart, cxv.CartesianPos3D)
print(vec)
# <CartesianPos3D: (x, y, z) [km]
#     [1. 2. 3.]>
```

### Converting coordinax to Astropy

```python
from plum import convert
import astropy.coordinates as apyc
import coordinax.vecs as cxv

# Create a coordinax vector
vec = cxv.CartesianPos3D.from_([1, 2, 3], "km")

# Convert to Astropy
apy_vec = convert(vec, apyc.CartesianRepresentation)
print(apy_vec)
# <CartesianRepresentation (x, y, z) in km
#     (1., 2., 3.)>
```

### Distance Conversions

```python
import coordinax as cx
import coordinax.distance as cxd
import astropy.units as u
from plum import convert

# Convert Astropy Quantity to coordinax Distance
q = 10 * u.kpc
dist = convert(q, cxd.Distance)

# Convert coordinax Distance to Astropy Quantity
apy_q = convert(dist, u.Quantity)

# Works with specialized distance types
parallax = cxd.Parallax.from_(5 * u.mas)
dist_from_parallax = convert(5 * u.mas, cxd.Parallax)

dist_mod = cxd.DistanceModulus.from_(15 * u.mag)
dist_from_distmod = convert(15 * u.mag, cxd.DistanceModulus)
```

### Frame Conversions

```python
import coordinax as cx
import coordinax.vecs as cxv
import coordinax_astro as cxa
import astropy.coordinates as apyc
from plum import convert
import unxt as u

# Convert ICRS frames (simple case)
cx_icrs = cxa.ICRS()
apy_icrs = convert(cx_icrs, apyc.ICRS)
back_to_cx = convert(apy_icrs, cxa.ICRS)

# Convert Galactocentric frames with custom parameters
galcen = cxv.LonLatSphericalPos(
    lon=u.Quantity(0, "deg"),
    lat=u.Quantity(0, "deg"),
    distance=u.Quantity(8.122, "kpc"),
)
galcen_v_sun = cxv.CartesianVel3D(
    x=u.Quantity(11.1, "km/s"),
    y=u.Quantity(244, "km/s"),
    z=u.Quantity(7.25, "km/s"),
)

cx_galcen = cxa.Galactocentric(
    galcen=galcen,
    z_sun=u.Quantity(20.8, "pc"),
    roll=u.Quantity(0, "deg"),
    galcen_v_sun=galcen_v_sun,
)

# Convert to Astropy (all parameters are preserved)
apy_galcen = convert(cx_galcen, apyc.Galactocentric)

# Round-trip back to coordinax
cx_result = convert(apy_galcen, cxa.Galactocentric)
```

## Supported Representations

### Position Types

- `CartesianPos3D` ↔ `CartesianRepresentation`
- `CylindricalPos` ↔ `CylindricalRepresentation`
- `SphericalPos` ↔ `PhysicsSphericalRepresentation`
- `LonLatSphericalPos` ↔ `SphericalRepresentation`
- `TwoSpherePos` ↔ `UnitSphericalRepresentation`

### Velocity Types

- `CartesianVel3D` ↔ `CartesianDifferential`
- `CylindricalVel` ↔ `CylindricalDifferential`
- `SphericalVel` ↔ `PhysicsSphericalDifferential`
- `LonLatSphericalVel` ↔ `SphericalDifferential`
- `LonCosLatSphericalVel` ↔ `SphericalCosLatDifferential`
- `TwoSphereVel` ↔ `UnitSphericalDifferential`

### Distance Types

- `Distance` ↔ `Quantity` (length units)
- `Parallax` ↔ `Quantity` (angle units, inverse relationship to distance)
- `DistanceModulus` ↔ `Quantity` (magnitude units)

### Reference Frames

- `coordinax_astro.ICRS` ↔ `astropy.coordinates.ICRS`
- `coordinax_astro.Galactocentric` ↔ `astropy.coordinates.Galactocentric`

## Guides

- [Frame Transformations](frame-transformations.md) - Detailed examples of
  transforming between ICRS and Galactocentric frames

## API Reference

See the [API documentation](api.md) for detailed information about all functions
and conversions.
