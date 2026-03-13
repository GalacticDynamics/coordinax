---
sd_hide_title: true
---

<h1> <code> coordinax.interop.astropy </code> </h1>

`coordinax.interop.astropy` provides seamless conversion between `coordinax` and `astropy` objects. This package enables you to work with both libraries interchangeably.

## Installation

[![PyPI version][pypi-version]][pypi-link] [![PyPI platforms][pypi-platforms]][pypi-link]

::::{tab-set}

:::{tab-item} uv

```bash
uv add coordinax.interop.astropy
```

:::

:::{tab-item} pip

```bash
pip install coordinax.interop.astropy
```

:::

::::

### Quick Start

<!-- invisible-code-block: python
import importlib.util
-->
<!-- skip: start if(importlib.util.find_spec('coordinax.interop.astropy') is None, reason="coordinax.interop.astropy not installed") -->

```{code-block} python
>>> import jax.numpy as jnp
>>> import plum

>>> import coordinax.astro as cxastro  # enables functionality
>>> import coordinax.main as cx

>>> import astropy.coordinates as apyc
```

```{code-block} python
>>> angle = cx.Angle(jnp.array([1, 2, 3]), "rad")
>>> angle_apy = plum.convert(angle, apyc.Angle)
>>> plum.convert(angle_apy, cx.Angle)
Angle([1., 2., 3.], 'rad')
```

```{code-block} python
>>> distance = cx.Distance(jnp.array([1, 2, 3]), "km")
>>> distance_apy = plum.convert(distance, apyc.Distance)
>>> cx.Distance.from_(distance_apy)
Distance([1., 2., 3.], 'km')
```

For a full walkthrough of Angle, Distance, DistanceModulus, and Parallax conversions, see [Quantities](quantities.md).

## Guides

- [Quantities](quantities.md) - Detailed examples of converting angles and distance-like quantities

## See Also

- [Astropy coordinates documentation](https://docs.astropy.org/en/stable/coordinates/) for background on astronomical coordinate systems

<!-- LINKS -->

[pypi-link]: https://pypi.org/project/coordinax/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/coordinax
[pypi-version]: https://img.shields.io/pypi/v/coordinax
