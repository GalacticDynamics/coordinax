# Vectors

This guide covers the creation and use of vector objects in `coordinax`,
including positions, velocities, accelerations, arithmetic, dimensionality,
spaces, and vector functions.

## Creating Vector Objects

You can create vectors for positions, velocities, and accelerations in many
supported dimension:

```{code-block} python
>>> import coordinax.vecs as cxv
>>> q1 = cxv.CartesianPos1D.from_(1, "kpc")
>>> q2 = cxv.CartesianPos2D.from_([1, 2], "kpc")
>>> q3 = cxv.CartesianPos3D.from_([1, 2, 3], "kpc")
>>> v3 = cxv.CartesianVel3D.from_([4, 5, 6], "kpc/Myr")
>>> a3 = cxv.CartesianAcc3D.from_([0.1, 0.2, 0.3], "kpc/Myr^2")
```

You can also create N-D vectors:

```{code-block} python
>>> qn = cxv.CartesianPosND.from_([1, 2, 3, 4], "kpc")
```

All vector types support flexible input: scalars, lists, arrays, or
{class}`~unxt.quantity.Quantity`.

The component values can be multidimensional arrays, allowing for batch
operations:

```{code-block} python
>>> arr = cxv.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")

```

## Arithmetic and Mathematical Operations

Vector objects support arithmetic and mathematical operations:

```{code-block} python
>>> q3 + q3
CartesianPos3D(
    x=Quantity(2, unit='kpc'), y=Quantity(4, unit='kpc'), z=Quantity(6, unit='kpc')
)

>>> 2 * q3
CartesianPos3D(
    x=Quantity(2, unit='kpc'), y=Quantity(4, unit='kpc'), z=Quantity(6, unit='kpc')
)

>>> v3 - v3
CartesianVel3D(
    x=Quantity(0, unit='kpc / Myr'),
    y=Quantity(0, unit='kpc / Myr'),
    z=Quantity(0, unit='kpc / Myr')
)
```

## Dimensionality: 1,N-D

`coordinax` provides vector classes for many dimensions:

- {class}`~coordinax.vecs.CartesianPos1D`,
  {class}`~coordinax.vecs.CartesianPos2D`,
  {class}`~coordinax.vecs.CartesianPos3D`,
  {class}`~coordinax.vecs.CartesianPosND`
- Similar classes for velocities (`CartesianVel*`), accelerations
  (`CartesianAcc*`), etc.
- Spacetime vectors {class}`~coordinax.vecs.FourVector`

## Conversion Between Representations

Vectors can be converted between coordinate systems:

```{code-block} python
>>> sph = q3.vconvert(cxv.SphericalPos)
>>> print(sph)
<SphericalPos: (r[kpc], theta[rad], phi[rad])
    [3.742 0.641 1.107]>
```

## Batch and Broadcast Operations

All vector types support batch operations and broadcasting:

```{code-block} python
>>> arr = cxv.CartesianPos3D.from_([[1,2,3],[4,5,6]], "kpc")
>>> arr * 2
CartesianPos3D(
    x=Quantity([2, 8], unit='kpc'),
    y=Quantity([ 4, 10], unit='kpc'),
    z=Quantity([ 6, 12], unit='kpc')
)
```

## Space Objects: Grouping Related Vectors

A {class}`~coordinax.vecs.Space` object collects related vectors (e.g.,
position, velocity, acceleration):

```{code-block} python
>>> space = cxv.Space(length=q3, speed=v3, acceleration=a3)
>>> print(space)
Space({
    'length':
    <CartesianPos3D: (x, y, z) [kpc]
         [1 2 3]>,
    'speed':
    <CartesianVel3D: (x, y, z) [kpc / Myr]
        [4 5 6]>,
    'acceleration':
    <CartesianAcc3D: (x, y, z) [kpc / Myr2]
        [0.1 0.2 0.3]>
})
```

You can convert all vectors in a {class}`~coordinax.vecs.Space` at once:

```{code-block} python
>>> space_sph = space.vconvert(cxv.SphericalPos)
>>> print(space_sph)
Space({
       'length':
       <SphericalPos: (r[kpc], theta[rad], phi[rad])
           [3.742 0.641 1.107]>,
       'speed':
       <SphericalVel: (r[kpc / Myr], theta[rad / Myr], phi[rad / Myr])
           [ 8.552  0.383 -0.6  ]>,
       'acceleration':
       <SphericalAcc: (r[kpc / Myr2], theta[rad / Myr2], phi[rad / Myr2])
           [ 3.742e-01 -9.355e-09  1.639e-09]>
    })
```

---

:::{seealso}

[API Documentation for Vectors](../api/vecs.md)

:::
