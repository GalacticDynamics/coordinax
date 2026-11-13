# coordinaxs.astro

```{toctree}
:maxdepth: 1
:hidden:

api.md
spec.md
```

Astronomy-specific reference frames for [coordinax](https://github.com/GalacticDynamics/coordinax).

This package provides astronomical reference frames like ICRS and Galactocentric for use with `coordinax`, enabling transformations between different astronomical coordinate systems.

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
import coordinaxs.astro as cxastro
import unxt as u

# Create a position in ICRS frame
crd_icrs = cx.Point.from_(
    {"r": u.Q(10, "kpc"), "theta": u.Q(45, "deg"), "phi": u.Q(30, "deg")},
    cxastro.ICRS(),
)

# Transform to Galactocentric frame
crd_gc = crd_icrs.to_frame(cxastro.Galactocentric())
```

## Available Frames

### ICRS

The International Celestial Reference System (ICRS) is the standard celestial reference frame.

```python
frame = cxastro.ICRS()
```

### Galactic

The (heliocentric) IAU 1958 Galactic coordinate frame, related to ICRS by a fixed rotation (matching Astropy's Galactic frame, including the ICRS/FK5 frame bias). Parameter-free.

```python
frame = cxastro.Galactic()
```

### Galactocentric

A reference frame centered on the Galactic center with configurable parameters.

```python
frame = cxastro.Galactocentric(
    galcen={
        "lon": u.Q(266, "deg"),
        "lat": u.Q(-29, "deg"),
        "distance": u.Q(8.122, "kpc"),
    },
    z_sun=u.Q(20.8, "pc"),
)
```

## Frame Transformations

The package provides frame transformation functions that work with coordinax's coordinate system:

```python
# Create a point in one frame
crd_icrs = cx.Point.from_([1.0, 2.0, 3.0], "kpc", cxastro.ICRS())

# Transform to another frame
crd_gc = crd_icrs.to_frame(cxastro.Galactocentric())
```

## API Reference

See the {doc}`API Reference </packages/coordinaxs.astro/api>` for complete documentation of all frames and functions.

## License

MIT License. See [LICENSE](../../../LICENSE) for details.
