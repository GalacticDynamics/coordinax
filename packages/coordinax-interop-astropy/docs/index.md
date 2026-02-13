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

- **Position vectors**: Convert between coordinax vectors and Astropy
  `BaseRepresentation` objects
- **Velocity vectors**: Convert between coordinax vectors (role=vel) and Astropy
  `BaseDifferential` objects
- **Distance types**: Convert between coordinax distance types (`Distance`,
  `Parallax`, `DistanceModulus`) and Astropy `Quantity` objects
- **Reference frames**: Convert between coordinax-astro frames (`ICRS`,
  `Galactocentric`) and Astropy coordinate frames
- **Quantity arrays**: Create coordinax vectors from Astropy `Quantity` arrays

## Quick Start

### Converting Astropy to coordinax

```
from plum import convert
import astropy.coordinates as apyc
import coordinax as cx

# Create an Astropy representation
cart = apyc.CartesianRepresentation(1, 2, 3, unit="km")

# Convert to coordinax Vector
vec = convert(cart, cx.Vector)
print(vec)
# <Vector: chart=Cart3D, role=Pos (x, y, z) [km]
#     [1. 2. 3.]>
```

### Converting coordinax to Astropy

```
from plum import convert
import astropy.coordinates as apyc
import coordinax as cx

# Create a coordinax vector
vec = cx.Vector.from_([1, 2, 3], "km")

# Convert to Astropy
apy_vec = convert(vec, apyc.CartesianRepresentation)
print(apy_vec)
# <CartesianRepresentation (x, y, z) in km
#     (1., 2., 3.)>
```

### Distance Conversions

```
import coordinax as cx
import coordinax.distances as cxd
import astropy.units as u
from plum import convert

# Convert Astropy Quantity to coordinax Distance
q = 10 * u.kpc
dist = convert(q, cxd.Distance)

# Convert coordinax Distance to Astropy Quantity
apy_q = convert(dist, u.Q

# Works with specialized distance types
parallax = cxd.Parallax.from_(5 * u.mas)
dist_from_parallax = convert(5 * u.mas, cxd.Parallax)

dist_mod = cxd.DistanceModulus.from_(15 * u.mag)
dist_from_distmod = convert(15 * u.mag, cxd.DistanceModulus)
```

### Frame Conversions

```
import coordinax as cx
import coordinax_astro as cxa
import astropy.coordinates as apyc
from plum import convert
import unxt as u

# Convert ICRS frames (simple case)
cx_icrs = cxa.ICRS()
apy_icrs = convert(cx_icrs, apyc.ICRS)
back_to_cx = convert(apy_icrs, cxa.ICRS)

# Convert Galactocentric frames with custom parameters
galcen = cx.Vector.from_(
    {"lon": u.Q(0, "deg"), "lat": u.Q(0, "deg"), "distance": u.Q(8.122, "kpc")},
    cxc.lonlatsph3d,
    cxr.point,
)
galcen_v_sun = cx.Vector.from_(u.Q([11.1, 244, 7.25], "km/s"))

cx_galcen = cxa.Galactocentric(
    galcen=galcen,
    z_sun=u.Q(20.8, "pc"),
    roll=u.Q(0, "deg"),
    galcen_v_sun=galcen_v_sun,
)

# Convert to Astropy (all parameters are preserved)
apy_galcen = convert(cx_galcen, apyc.Galactocentric)

# Round-trip back to coordinax
cx_result = convert(apy_galcen, cxa.Galactocentric)
```

## Supported Representations

### Position Types (coordinax Vector → Astropy)

- `Vector(chart=cart3d, role=pos)` ↔ `CartesianRepresentation`
- `Vector(chart=cyl3d, role=pos)` ↔ `CylindricalRepresentation`
- `Vector(chart=sph3d, role=pos)` ↔ `PhysicsSphericalRepresentation`
- `Vector(chart=lonlatsph3d, role=pos)` ↔ `SphericalRepresentation`
- `Vector(chart=twosphere, role=pos)` ↔ `UnitSphericalRepresentation`

### Velocity Types (coordinax Vector → Astropy)

- `Vector(chart=cart3d, role=vel)` ↔ `CartesianDifferential`
- `Vector(chart=cyl3d, role=vel)` ↔ `CylindricalDifferential`
- `Vector(chart=sph3d, role=vel)` ↔ `PhysicsSphericalDifferential`
- `Vector(chart=lonlatsph3d, role=vel)` ↔ `SphericalDifferential`
- `Vector(chart=loncoslatsph3d, role=vel)` ↔ `SphericalCosLatDifferential`

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
