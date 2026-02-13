# coordinax-interop-astropy

Astropy interoperability for coordinax.

This package provides converters and constructors to seamlessly work with
Astropy coordinate representations, differentials, frames, and quantities.

## Installation

```bash
pip install coordinax-interop-astropy
```

Or with uv:

```bash
uv add coordinax-interop-astropy
```

## Features

- **Vector Conversions**: Convert between coordinax vectors and Astropy
  representations (Cartesian, Spherical, Cylindrical)
- **Distance Conversions**: Convert between coordinax distance types
  (`Distance`, `Parallax`, `DistanceModulus`) and Astropy quantities
- **Frame Conversions**: Convert between coordinax-astro frames (`ICRS`,
  `Galactocentric`) and Astropy coordinate frames
- **Full Velocity Support**: Convert position and velocity vectors with proper
  differential handling

## Usage

### Vector Conversions

```
import coordinax as cx
from astropy.coordinates import CartesianRepresentation
from plum import convert

# Convert Astropy to coordinax
cart = CartesianRepresentation(1, 2, 3, unit="m")
vec = cx.vector(cart)

# Convert coordinax to Astropy
apy_vec = convert(vec, CartesianRepresentation)
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
apy_q = convert(dist, u.Q)

# Works with Parallax and DistanceModulus too
parallax = cxd.Parallax.from_(5 * u.mas)
dist_mod = cxd.DistanceModulus.from_(15 * u.mag)
```

### Frame Conversions

```
import astropy.coordinates as apyc
import coordinax as cx
import coordinax_astro as cxa
from plum import convert
import unxt as u

# Convert ICRS frames
cx_icrs = cxa.ICRS()
apy_icrs = convert(cx_icrs, apyc.ICRS)

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
apy_galcen = convert(cx_galcen, apyc.Galactocentric)

# Round-trip conversions preserve values
cx_result = convert(apy_galcen, cxa.Galactocentric)
```
