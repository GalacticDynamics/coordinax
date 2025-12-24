---
sd_hide_title: true
---

<h1> <code> coordinax </code> </h1>

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📦 Packages

coordinax-api <packages/coordinax-api/index>
coordinax <self>
coordinax-astro <packages/coordinax-astro/index>
coordinax-hypothesis <packages/coordinax-hypothesis/index>
coordinax-interop-astropy <packages/coordinax-interop-astropy/index>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📚 Guides

guides/quantities.md
guides/charts.md
guides/metrics.md
guides/vectors.md
guides/vector_algebra.md
guides/operators.md
guides/coordinates_and_frames.md
guides/embedded_manifolds.md
packages/coordinax-hypothesis/testing-guide
```

<!--
```{toctree}
:maxdepth: 1
:hidden:
:caption: 🤝 Interoperability
:glob:

interop/*
```
-->

```{toctree}
:maxdepth: 2
:hidden:
:caption: 📘 API Reference

coordinax-api <packages/coordinax-api/api>
coordinax <api/index.md>
coordinax-astro <packages/coordinax-astro/api>
coordinax-hypothesis <packages/coordinax-hypothesis/api>
coordinax-interop-astropy <packages/coordinax-interop-astropy/api>
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

`coordinax` enables working with coordinates and reference frames with
[JAX][jax].

`coordinax` supports JAX's main features:

- JIT compilation ({func}`~jax.jit`)
- vectorization ({func}`~jax.vmap`, etc.)
- auto-differentiation ({func}`~jax.grad`, {func}`~jax.jacobian`,
  {func}`jax.hessian`)
- GPU/TPU/multi-host acceleration

And best of all, `coordinax` doesn't force you to use special unit-compatible
re-exports of JAX libraries. You can use `coordinax` with existing JAX code, and
with one simple decorator ({func}`quax.quaxify`), JAX will work with `coordinax`
objects.

---

## Installation

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

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

To install the latest development version of `coordinax` directly from the
GitHub repository, use pip:

```bash
pip install git+https://https://github.com/GalacticDynamics/coordinax.git
```

:::

:::{tab-item} building from source

To build `coordinax` from source, clone the repository and install it with pip:

```bash
cd /path/to/parent
git clone https://https://github.com/GalacticDynamics/coordinax.git
cd coordinax
pip install -e .  # editable mode
```

:::

::::

## Quickstart

The `coordinax` package has powerful tools for representing, using, and
transforming coordinate objects, such as:

- specific {class}`~unxt.quantity.Quantity` subclasses like
  {class}`~coordinax.angle.Angle` and {class}`~coordinax.distance.Distance`
- representation charts exposed in {mod}`coordinax.charts` (Cartesian,
  cylindrical, spherical, manifolds, spacetime)
- vector objects ({class}`~coordinax.Vector`, {class}`~coordinax.FiberPoint`)
  that pair data with reps and roles (Pos, Vel, Acc, ...)
- coordinate and physical conversions
  ({func}`~coordinax.transforms.point_transform`,
  {func}`~coordinax.transforms.physical_tangent_transform`,
  {func}`~coordinax.vconvert`)
- operations on vectors ({mod}`~coordinax.ops`)
- reference frames and coordinate systems ({mod}`~coordinax.frames`)
- coordinates that combine vectors and frames
  ({class}`~coordinax.frames.Coordinate`)
- and more!

This functionality is organized into submodules, which are imported into the
top-level `coordinax` namespace. You can import them directly, or use the
`coordinax` namespace to access them.

```{code-block} python
>>> import coordinax as cx

>>> from inspect import ismodule
>>> [name for name in cx.__all__ if ismodule(getattr(cx, name))]
['angles', 'charts', 'distances', 'frames', 'metrics',
 'ops', 'roles', 'transforms', 'objects']
```

We recommend importing as needed:

- `coordinax` as `cx`
- `coordinax.angles` as `cxa`
- `coordinax.distance` as `cxd`
- `coordinax.charts` as `cxc`
- `coordinax.ops` as `cxo`
- `coordinax.frames` as `cxf`

### Angles and Distances

`coordinax` is built on top of [`unxt`](http://unxt.readthedocs.io), which
provides support for quantity objects that represent a data array with an
associated unit with the {class}`unxt.quantity.Quantity` class. These
{class}`~unxt.quantity.Quantity` objects can be used throughout `coordinax`, but
`coordinax` also provides specific classes that offer additional functionality.

Let's start with angles, which are represented by the
{class}`~coordinax.angle.Angle` class. This class enforces that the inputted
units have angular dimensions and provides some other useful utilities for
working with angles. For example, the resulting {class}`~coordinax.angle.Angle`
(a re-export of `unxt.Angle`) object can be wrapped to a specific range to
conform to a branch cut (e.g., 0 to 2π or -180º to 180º).

```{code-block} python
>>> import unxt as u

>>> a = u.Angle(370, "deg")
>>> a
Angle(Array(370, dtype=int32, weak_type=True), unit='deg')

>>> a.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
Angle(Array(10, dtype=int32, weak_type=True), unit='deg')
```

Similarly, the {class}`~coordinax.distance.Distance` class represents distances
in `coordinax`:

```{code-block} python
>>> d = cx.distances.Distance(10, "kpc")
>>> d
Distance(Array(10, dtype=int32, weak_type=True), unit='kpc')
```

but other distance-like objects can be represented with the
{class}`~coordinax.distance.Parallax` and
{class}`~coordinax.distance.DistanceModulus` classes. These classes check that
the units have distance dimensions, and they provide useful properties for
converting between different distance representations.

```{code-block} python
>>> d.parallax
Parallax(Array(4.848137e-10, dtype=float32, weak_type=True), unit='rad')

>>> d.distance_modulus
DistanceModulus(Array(15., dtype=float32), unit='mag')
```

### Creating and Working with Vector Objects

Vectors combine data, a representation, and a role (Pos, Vel, Acc). Here's a
Cartesian 3D position and velocity:

```
import coordinax as cx
import unxt as u

q = cx.Vector(
    data={"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")},
    chart=cx.charts.cart3d,
    role=cx.roles.Pos(),
)
v = cx.Vector(
    data={"x": u.Q(4.0, "kpc/Myr"), "y": u.Q(5.0, "kpc/Myr"), "z": u.Q(6.0, "kpc/Myr")},
    chart=cx.charts.cart3d,
    role=cx.roles.Vel(),
)
```

#### Vector Conversion

```
q_sph = q.vconvert(cx.charts.sph3d)
v_sph = v.vconvert(cx.charts.sph3d, q)
```

#### Creating an `FiberPoint` Object

```
space = cx.FiberPoint(base=q, speed=v)
space_sph = space.vconvert(cx.charts.sph3d)
```

### Operators on Vectors

The {mod}`coordinax.ops` module (shorthand `cxo`) provides a framework for and
set of vector operations that work seamlessly with all `coordinax` vector types.

```{code-block} text
>>> import coordinax.ops as cxo

>>> op = cxo.GalileanOp.from_([10, 10, 10], "kpc")

>>> print(op(q))
<Cart3D: (x, y, z) [kpc]
    [11 12 13]>

```

### Reference Frames and Coordinates

{mod}`coordinax.frames` (shorthand `cxf`) provides a framework for defining and
working with reference frames and coordinate systems.

```{code-block} text
>>> import coordinax.frames as cxf

>>> alice = cxf.Alice()
>>> alice
Alice()

>>> bob = cxf.Bob()
>>> bob
Bob()

```

Frames can be used to define coordinate transformations. For example, you can
transform a position vector from the Alice frame to the Bob frame:

```{code-block} text

>>> op = cxf.frame_transform_op(alice, bob)
>>> t = u.Q(1, "yr")
>>> print(op(t, q)[1])
<Cart3D: (x, y, z) [kpc]
    [1. 2. 3.]>

```

Coordinate objects can also be created to represent positions in a specific
frame:

```{code-block} text

>>> coord = cxf.Coordinate(q, frame=alice)
>>> print(coord)
Coordinate(
    { 'base': <Cart3D: (x, y, z) [kpc]
                    [1 2 3]> },
    frame=Alice()
)

>>> coord.to_frame(bob, t)
Coordinate(
    FiberPoint({
        'base': Cart3D(x=Q(f32[], 'kpc'), y=Q(f32[], 'kpc'),
                                 z=Q(f32[], 'kpc')) }),
    frame=Bob()
)

>>> coord.vconvert(cx.charts.sph3d)
Coordinate(
    FiberPoint({
        'base': Spherical3D( r=Distance(f32[], 'kpc'), theta=Angle(f32[], 'rad'),
                                phi=Angle(f32[], 'rad') )
    }),
    frame=Alice()
)

```

## Ecosystem

### `coordinax`'s Dependencies

- [unxt][unxt]: Quantities in JAX.
- [Equinox][equinox]: one-stop JAX library, for everything that isn't already in
  core JAX.
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
