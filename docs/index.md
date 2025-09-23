---
sd_hide_title: true
---

<h1> <code> coordinax </code> </h1>

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📚 Guides

guides/quantities.md
guides/vectors.md
guides/operators.md
guides/coordinates_and_frames.md
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
:caption: 🔌 API Reference

api/index.md
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
- vector objects
  - 1D, 2D, 3D and N-dimensional vector classes in many different representation
    types (e.g., {class}`~coordinax.vecs.CartesianPos1D`,
    {class}`~coordinax.vecs.CartesianPos2D`,
    {class}`~coordinax.vecs.CartesianPos3D`,
    {class}`~coordinax.vecs.SphericalPos`,
    {class}`~coordinax.vecs.ProlateSpheroidalPos`, etc.)
  - time-differential vector objects, like velocities and accelerations (e.g.,
    {class}`~coordinax.vecs.CartesianVel3D`,
    {class}`~coordinax.vecs.SphericalVel`, etc.)
  - collections of vector objects
- transformations on vectors ({func}`~coordinax.vecs.vconvert`)
- reference frames <!-- TODO: add e.g., links to classes -->

This functionality is organized into submodules, which are imported into the
top-level `coordinax` namespace. You can import them directly, or use the
`coordinax` namespace to access them.

```{code-block} python
>>> import coordinax as cx

>>> from inspect import ismodule
>>> [x for x in cx.__all__ if ismodule(getattr(cx, x))]
['angle', 'distance', 'vecs', 'ops', 'frames']
```

We recommend importing as needed:

- `coordinax` as `cx`
- `coordinax.vecs` as `cxv`
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

>>> a.wrap_to(u.Quantity(0, "deg"), u.Quantity(360, "deg"))
Angle(Array(10, dtype=int32, weak_type=True), unit='deg')
```

Similarly, the {class}`~coordinax.distance.Distance` class represents distances
in `coordinax`:

```{code-block} python
>>> d = cx.distance.Distance(10, "kpc")
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

Vector objects in `coordinax` represent positions, velocities, and accelerations
in a variety of coordinate systems and dimensions. Here are some common
operations:

#### Constructing Vector Objects

You can create a vector by specifying its components and units:

```{code-block} python
>>> import coordinax.vecs as cxv

>>> q = cxv.CartesianPos3D.from_([1, 2, 3], "kpc")
>>> print(q)
<CartesianPos3D: (x, y, z) [kpc]
    [1 2 3]>
```

The {meth}`~coordinax.vecs.AbstractVector.from_` method is a flexible
constructor that allows you to create vectors from various input formats, such
as lists, tuples, or NumPy arrays. Direct construction is also possible by
specifying values for all components:

```{code-block} python
>>> q = cxv.CartesianPos3D(x=u.Quantity(1, "kpc"), y=u.Quantity(2, "kpc"), z=u.Quantity(3, "kpc"))
>>> print(q)
<CartesianPos3D: (x, y, z) [kpc]
    [1 2 3]>
```

#### Vector Conversion

Vectors can be converted between different coordinate representations using the
{meth}`~coordinax.vecs.AbstractVector.vconvert` method. For example, to convert
a Cartesian position vector to spherical coordinates:

```{code-block} python
>>> sph = q.vconvert(cxv.SphericalPos)
>>> print(sph)
<SphericalPos: (r[kpc], theta[rad], phi[rad])
    [3.742 0.641 1.107]>
```

#### Transforming Velocities

Velocity vectors can also be converted to other representations, but require
specifying the corresponding position:

```{code-block} python
>>> v = cxv.CartesianVel3D.from_([4, 5, 6], "kpc/Myr")
>>> v_sph = v.vconvert(cxv.SphericalVel, q)
>>> print(v_sph)
<SphericalVel: (r[kpc / Myr], theta[rad / Myr], phi[rad / Myr])
    [ 8.552  0.383 -0.6  ]>
```

#### Creating a `KinematicSpace` Object

A {class}`~coordinax.vecs.KinematicSpace` object collects related vectors (e.g.,
position, velocity, acceleration) into a single container:

```{code-block} python
>>> import coordinax as cx

>>> space = cx.KinematicSpace(length=q, speed=v)
>>> print(space)
KinematicSpace({
   'length': <CartesianPos3D: (x, y, z) [kpc]
       [1 2 3]>,
   'speed': <CartesianVel3D: (x, y, z) [kpc / Myr]
       [4 5 6]>
})
```

You can convert all vectors in a {class}`~coordinax.vecs.KinematicSpace` to a
different representation at once:

```{code-block} python
>>> space_sph = space.vconvert(cxv.SphericalPos)
>>> print(space_sph)
KinematicSpace({
    'length': <SphericalPos: (r[kpc], theta[rad], phi[rad])
                [3.742 0.641 1.107]>,
    'speed': <SphericalVel: (r[kpc / Myr], theta[rad / Myr], phi[rad / Myr])
                [ 8.552  0.383 -0.6  ]>
})
```

### Operators on Vectors

The {mod}`coordinax.ops` module (shorthand `cxo`) provides a framework for and
set of vector operations that work seamlessly with all `coordinax` vector types.

```{code-block} python
>>> import coordinax.ops as cxo

>>> op = cxo.GalileanSpatialTranslation.from_([10, 10, 10], "kpc")

>>> print(op(q))
<CartesianPos3D: (x, y, z) [kpc]
    [11 12 13]>

```

### Reference Frames and Coordinates

{mod}`coordinax.frames` (shorthand `cxf`) provides a framework for defining and
working with reference frames and coordinate systems.

```{code-block} python
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

```{code-block} python

>>> op = cxf.frame_transform_op(alice, bob)
>>> t = u.Quantity(1, "yr")
>>> print(op(t, q)[1])
<CartesianPos3D: (x, y, z) [kpc]
    [1. 2. 3.]>

```

Coordinate objects can also be created to represent positions in a specific
frame:

```{code-block} python

>>> coord = cxf.Coordinate(q, frame=alice)
>>> print(coord)
Coordinate(
    { 'length': <CartesianPos3D: (x, y, z) [kpc]
                    [1 2 3]> },
    frame=Alice()
)

>>> coord.to_frame(bob, t)
Coordinate(
    KinematicSpace({ 'length': CartesianPos3D(
        x=Quantity(f32[], unit='kpc'),
        y=Quantity(f32[], unit='kpc'),
        z=Quantity(f32[], unit='kpc')
    ) }),
    frame=Bob()
)

>>> coord.vconvert(cxv.SphericalPos)
Coordinate(
    KinematicSpace({ 'length': SphericalPos(
          r=Distance(weak_f32[], unit='kpc'),
          theta=Angle(f32[], unit='rad'),
          phi=Angle(weak_f32[], unit='rad')
        ) }),
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
