# coordinax-astro

Astronomy-specific reference frames for coordinax.

This package provides astronomical reference frames like ICRS and Galactocentric
for use with coordinax, enabling transformations between different astronomical
coordinate systems.

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

```
import coordinax as cx
import coordinax_astro as cxa
import unxt as u

# Create a position in ICRS frame
pos = cx.Spherical3D(
    r=u.Q(10, "kpc"),
    theta=u.Q(45, "deg"),
    phi=u.Q(30, "deg"),
)
icrs_coord = cx.Coordinate(pos, frame=cxa.ICRS())

# Transform to Galactocentric frame
galactocentric = icrs_coord.to_frame(cxa.Galactocentric())
```

## Available Frames

### ICRS

The International Celestial Reference System (ICRS):

```
frame = cxa.ICRS()
```

### Galactocentric

A reference frame centered on the Galactic center with configurable parameters:

```
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

Transform coordinates between reference frames:

```
# Create a coordinate in one frame
coord_icrs = cx.Coordinate(pos, frame=cxa.ICRS())

# Transform to another frame
coord_gal = coord_icrs.to_frame(cxa.Galactocentric())
```

## Documentation

For detailed usage examples and API documentation, see the
[full documentation](https://coordinax.readthedocs.io/).

## License

MIT License. See [LICENSE](../../LICENSE) for details.
