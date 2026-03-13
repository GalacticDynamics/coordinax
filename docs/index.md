---
sd_hide_title: true
---

<h1> <code> coordinax </code> </h1>

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📦 Packages

coordinax.api <packages/coordinax.api/index>
coordinax <self>
coordinax.astro <packages/coordinax.astro/index>
coordinax.hypothesis <packages/coordinax.hypothesis/index>
coordinax.interop.astropy <packages/coordinax.interop.astropy/index>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📚 Guides

guides/quantities.md
packages/coordinax.hypothesis/testing-guide
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 📘 API Reference

coordinax.api <packages/coordinax.api/api>
coordinax <api/index.md>
coordinax.astro <packages/coordinax.astro/api>
coordinax.hypothesis <packages/coordinax.hypothesis/api>
coordinax.interop.astropy <packages/coordinax.interop.astropy/api>
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: More

glossary.md
conventions.md
contributing.md
dev.md
```

# 🚀 Get Started

`coordinax` enables working with coordinates and reference frames with [JAX][jax].

`coordinax` supports JAX's main features:

- JIT compilation ({func}`~jax.jit`)
- vectorization ({func}`~jax.vmap`, etc.)
- auto-differentiation ({func}`~jax.grad`, {func}`~jax.jacobian`, {func}`jax.hessian`)
- GPU/TPU/multi-host acceleration

And best of all, `coordinax` doesn't force you to use special unit-compatible re-exports of JAX libraries. You can use `coordinax` with existing JAX code, and with one simple decorator ({func}`quax.quaxify`), JAX will work with `coordinax` objects.

---

## Installation

[![PyPI version][pypi-version]][pypi-link] [![PyPI platforms][pypi-platforms]][pypi-link]

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinax
```

:::

:::{tab-item} uv

```bash
uv add coordinax
```

:::

:::{tab-item} source, via pip

To install the latest development version of `coordinax` directly from the GitHub repository, use pip:

```bash
uv add git+https://https://github.com/GalacticDynamics/coordinax.git@main
```

You can customize the branch by replacing `main` with any other branch name.

:::

:::{tab-item} building from source

To build `coordinax` from source, clone the repository and install it with uv:

```bash
cd /path/to/parent
git clone https://https://github.com/GalacticDynamics/coordinax.git
cd coordinax
uv pip install -e .  # editable mode
```

:::

::::

## Quickstart

The `coordinax` package has powerful tools for representing, using, and transforming coordinate objects, such as:

- specific {class}`~unxt.quantity.Quantity` subclasses like {class}`~coordinax.angles.Angle` and {class}`~coordinax.distances.Distance`
- and more!

This functionality is organized into submodules available under the top-level `coordinax` namespace. You can import them directly, or for many objects use the `coordinax.main` namespace to access them.

<!-- invisible-code-block: python
import coordinax.angles
import coordinax.api
import coordinax.astro
import coordinax.distances
import coordinax.hypothesis
import coordinax.interop
import coordinax.main
-->

```{code-block} python
>>> import coordinax

>>> from inspect import ismodule
>>> [name for name in dir(coordinax) if ismodule(getattr(coordinax, name))]
['angles', 'api', 'astro', 'distances', 'hypothesis', 'interop', 'main']
```

We recommend importing as needed:

- `coordinax.main` as `cx` : probably everything you need!
- `coordinax.angles` as `cxa` : further angle-specific functionality.
- `coordinax.distances` as `cxd` : further distance-specific functionality.

### Angles and Distances

`coordinax` is built on top of [`unxt`](http://unxt.readthedocs.io), which provides support for quantity objects that represent a data array with an associated unit with the {class}`unxt.quantity.Quantity` class. These {class}`~unxt.quantity.Quantity` objects can be used throughout `coordinax`, but `coordinax` also provides specific classes that offer additional functionality.

Let's start with angles, which are represented by the {class}`~coordinax.angles.Angle` class. This class enforces that the inputted units have angular dimensions and provides some other useful utilities for working with angles. For example, the resulting {class}`~coordinax.angles.Angle` (a re-export of `unxt.Angle`) object can be wrapped to a specific range to conform to a branch cut (e.g., 0 to $2\pi$ or $-180^\circ$ to $180^\circ$).

```{code-block} python
>>> import coordinax.main as cx
>>> import unxt as u

>>> a = cx.Angle(370, "deg")
>>> a
Angle(370, 'deg')

>>> a.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
Angle(10, 'deg')
```

Similarly, the {class}`~coordinax.distances.Distance` class represents distances in `coordinax`:

```{code-block} python
>>> d = cx.Distance(10, "kpc")
>>> d
Distance(10, 'kpc')
```

but other distance-like objects can be represented with the {class}`~coordinax.distances.Parallax` and {class}`~coordinax.distances.DistanceModulus` classes. These classes check that the units have distance dimensions, and they provide useful properties for converting between different distance representations.

```{code-block} python
>>> d.parallax
Parallax(4.84813681e-10, 'rad')

>>> d.distance_modulus
DistanceModulus(15., 'mag')
```

## Ecosystem

### `coordinax`'s Dependencies

- [unxt][unxt]: Quantities in JAX.
- [Equinox][equinox]: one-stop JAX library, for everything that isn't already in core JAX.
- [Quax][quax]: JAX + multiple dispatch + custom array-ish objects.
- [Quaxed][quaxed]: pre-`quaxify`ed Jax.
- [plum][plum]: multiple dispatch in python

### `coordinax`'s Dependents

- [galax][galax]: Galactic dynamics in JAX.

<!-- LINKS -->

[unxt]: https://github.com/GalacticDynamics/unxt
[equinox]: https://docs.kidger.site/equinox/
[galax]: https://github.com/GalacticDynamics/galax
[jax]: https://jax.readthedocs.io/en/latest/
[plum]: https://pypi.org/project/plum-dispatch/
[quax]: https://github.com/patrick-kidger/quax
[quaxed]: https://quaxed.readthedocs.io/en/latest/
[pypi-link]: https://pypi.org/project/coordinax/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/coordinax
[pypi-version]: https://img.shields.io/pypi/v/coordinax
