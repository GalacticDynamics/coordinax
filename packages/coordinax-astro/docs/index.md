# coordinax-astro

```{toctree}
:maxdepth: 1
:hidden:

api
```

Astronomy-specific reference frames for
[coordinax](https://github.com/GalacticDynamics/coordinax).

This package provides astronomical reference frames like ICRS and Galactocentric
for use with `coordinax`, enabling transformations between different
astronomical coordinate systems.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinax[astro]
```

:::

:::{tab-item} uv

```bash
uv add coordinax --extra astro
```

:::

::::

## Quick Start

```python
import coordinax as cx
import coordinax_astro as cxa
import unxt as u

# Create a position in ICRS frame
pos = cx.SphericalPos(r=u.Q(10, "kpc"), theta=u.Q(45, "deg"), phi=u.Q(30, "deg"))
icrs_coord = cx.Coordinate({"length": pos}, frame=cxa.ICRS())

# Transform to Galactocentric frame
galactocentric = icrs_coord.to_frame(cxa.Galactocentric())
```

## Available Frames

### ICRS

The International Celestial Reference System (ICRS) is the standard celestial
reference frame.

```python
frame = cxa.ICRS()
```

### Galactocentric

A reference frame centered on the Galactic center with configurable parameters.

```python
frame = cxa.Galactocentric(
    galcen={
        "lon": u.Q(266, "deg"),
        "lat": u.Q(-29, "deg"),
        "distance": u.Q(8.122, "kpc"),
    },
    z_sun=u.Q(20.8, "pc"),
)
```

## Frame Transformations

The package provides frame transformation functions that work with coordinax's
coordinate system:

```python
# Create a coordinate in one frame
coord_icrs = cx.Coordinate({"length": pos}, frame=cxa.ICRS())

# Transform to another frame
coord_gal = coord_icrs.to_frame(cxa.Galactocentric())
```

## Integration with Astropy

When `astropy` is installed, `coordinax-astro` can interoperate with astropy's
coordinate frames:

```python
from astropy.coordinates import SkyCoord
import coordinax_astro as cxa

# Convert from astropy SkyCoord (requires astropy)
# skycoord = SkyCoord(ra=10*u.deg, dec=20*u.deg, distance=100*u.pc)
# coord = cxa.ICRS.from_skycoord(skycoord)
```

## API Reference

See the [API Reference](api.md) for complete documentation of all frames and
functions.

## License

MIT License. See [LICENSE](../../LICENSE) for details.
